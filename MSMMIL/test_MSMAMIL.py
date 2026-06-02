import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchmetrics
import numpy as np

# Import necessary components from the project
from utils.utils import Struct, set_seed, MetricLogger
from datasets.datasets import build_HDF5_feat_dataset
from architecture.msma import MSMAMIL
from timm.utils import accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):
    """
    Evaluates the model on a given dataset and computes overall and per-class metrics.
    """
    net.eval()
    y_pred_list = []
    y_true_list = []
    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        
        _, slide_preds, _ = net(image_patches)
        loss = criterion(slide_preds, labels)
        pred_probs = torch.softmax(slide_preds, dim=-1)
        
        acc1 = accuracy(pred_probs, labels, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred_list.append(pred_probs)
        y_true_list.append(labels)

    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)

    # --- Calculate Overall and Per-Class Metrics ---

    # Overall metrics
    overall_auroc = torchmetrics.AUROC(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).item()
    overall_f1 = torchmetrics.F1Score(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).item()
    overall_recall = torchmetrics.Recall(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).item()
    overall_specificity = torchmetrics.Specificity(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).item()

    # Per-class metrics
    per_class_auroc = torchmetrics.AUROC(num_classes=conf.n_class, task='multiclass', average=None).to(device)(y_pred, y_true).cpu().numpy()
    per_class_acc = torchmetrics.Accuracy(num_classes=conf.n_class, task='multiclass', average=None).to(device)(y_pred, y_true).cpu().numpy()

    print(f'* Overall Acc@1 {metric_logger.acc1.global_avg:.3f} | Loss {metric_logger.loss.global_avg:.3f} | AUROC {overall_auroc:.3f} | F1 {overall_f1:.3f}')

    return (overall_auroc, metric_logger.acc1.global_avg, overall_f1, overall_recall, overall_specificity, metric_logger.loss.global_avg,
            per_class_auroc, per_class_acc)


def get_arguments():
    """
    Parses command-line arguments for the evaluation script.
    """
    parser = argparse.ArgumentParser('WSI classification evaluation', add_help=False)
    parser.add_argument('--config', dest='config', default='config/four_natural_supervised_config.yml',
                        help='Path to the dataset settings in YAML format.')
    parser.add_argument('--ckpt_path', required=True, type=str,
                        help='Path to the pre-trained model checkpoint file (.pth).')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed for reproducibility.')
    # Arguments to define the model architecture, should match the trained model
    parser.add_argument('--n_token', type=int, default=4,
                        help="Number of attention branches (MBA).")
    parser.add_argument('--n_masked_patch', type=int, default=10,
                        help="Number of top-K instances for STKIM.")
    parser.add_argument('--mask_drop', type=float, default=1.0,
                        help="Masking ratio for STKIM.")
    parser.add_argument('--arch', type=str, default='msma',
                        help="Architecture type (e.g., 'msma').")
    parser.add_argument('--pretrain', default='natural_supervised',
                        help='Pretrained backbone used for feature extraction.')

    args = parser.parse_args()
    return args

def main():
    # 1. Load Configuration from YAML and Command-Line Arguments
    args = get_arguments()
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    set_seed(conf.seed)
    print("--- Evaluation Script ---")
    print("Used config:")
    pprint(vars(conf))

    # 2. Set Feature Dimensions Based on Pre-trained Model
    if conf.pretrain == 'medical_ssl':
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain == 'natural_supervised':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain in ['path-clip-B', 'openai-clip-B', 'plip', 'quilt-net', 'path-clip-B-AAAI', 'biomedclip']:
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain in ['path-clip-L-336', 'openai-clip-L-336']:
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'UNI':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat = 1536
        conf.D_inner = 768

    # 3. Prepare the Test Dataset
    h5_path = os.path.join(conf.data_dir, f'{conf.dataset}_patch_feats_pretrain_{conf.pretrain}.h5')
    _, _, test_data = build_HDF5_feat_dataset(h5_path, conf)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    print(f"\nTest dataset loaded from {h5_path}.")
    print(f"Number of test samples: {len(test_data)}")

    # 4. Initialize Model and Load Checkpoint
    model = MSMAMIL(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model.to(device)

    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint path '{args.ckpt_path}' does not exist.")
        sys.exit(1)

    print(f"Loading model from checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    
    # Extract model state dictionary, handling different checkpoint formats
    model_state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    # Remove 'module.' prefix if model was saved using DataParallel
    new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    # 5. Run Evaluation
    criterion = torch.nn.CrossEntropyLoss()
    print("\n--- Starting Evaluation on Test Set ---")
    
    (test_auc, test_acc, test_f1, test_recall, test_specificity, test_loss,
     per_class_auc, per_class_acc) = evaluate(
        model, criterion, test_loader, device, conf, 'Test'
    )

    print("\n--- Overall Evaluation Finished ---")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test Specificity: {test_specificity:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    print("\n--- Per-Class Metrics ---")
    for i in range(conf.n_class):
        print(f"  Class {i}:")
        print(f"    - AUC:      {per_class_auc[i]:.4f}")
        print(f"    - Accuracy: {per_class_acc[i]:.4f}")


if __name__ == '__main__':
    main()