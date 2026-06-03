'''
baseline, 20x,mamba+atten
'''
import argparse
import os
import yaml

# internal imports
from datasets.dataset_module import PLDataModule
from models.model import Mamatten
from models.fusion import Fusion_Block

# pytorch impoRTs
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, AUROC, Recall, Specificity

# pytorch lightning imports
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule, Trainer, utilities
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
#from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class PLModule(LightningModule):
    def __init__(self, config_dict):
        super(PLModule, self).__init__()
        self.config_dict = config_dict
        self.automatic_optimization = False
        
        if config_dict['model_arguments']['bag_loss'] == 'ce':
            self.criterion2 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion0 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_fusion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise NotImplementedError

        self.n_cluster = self.config_dict['n_cluster']
        self.k_sample = self.config_dict['k_sample']
        self.model2 = Mamatten(**config_dict['model_arguments'], k_sample=self.k_sample)
        self.model1 = Mamatten(**config_dict['model_arguments'], k_sample=self.k_sample)
        self.model0 = Mamatten(**config_dict['model_arguments'], k_sample=self.k_sample)
        #self.fusion = Fusion_Block(in_channel=3*self.n_cluster*self.k_sample)
        self.fusion = Fusion_Block(in_channel=512)

        self.bag_weight = self.config_dict['model_arguments']['bag_weight']
        self.contra_weight = config_dict['model_arguments']['contra_weight']
        
        self.acc = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.auc = AUROC(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.f1 = F1Score(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.sensitivity = Recall(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.specificity = Specificity(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.log_f = 'logs/main_9_cm16_single.txt'

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
    
    def training_step(self, batch, batch_index):
        x2,x1,x0, y,y_, _ = batch
        opt = self.optimizers()
        opt.zero_grad()

        x2,x1,x0, y,y_, _ = batch
        y_ = y_[0]
        # logits, Y_prob, _, A_raw, results_dict2 = self.model2(x2, label=y, label_c=y_[0])
        # loss2 = self.bag_weight * self.criterion2(logits, y)
        # loss2 += self.contra_weight * results_dict2['contra_loss']
        
        # logits, Y_prob, _, A_raw, results_dict1 = self.model1(x1, label=y, label_c=y_[1])
        # loss1 = self.bag_weight * self.criterion1(logits, y)
        # loss1 += self.contra_weight * results_dict1['contra_loss']

        logits, Y_prob, _, A_raw, results_dict0 = self.model0(x0, label=y, label_c=y_[2])
        loss0 = self.bag_weight * self.criterion0(logits, y)
        loss0 += self.contra_weight * results_dict0['contra_loss']

        #concat
        # slides = torch.cat((results_dict2['slides'], results_dict1['slides'], results_dict0['slides']), dim=0)
        slides = torch.cat((results_dict0['slides'],), dim=0)
        # patches = torch.cat((results_dict2['patches'], results_dict1['patches'], results_dict0['patches']), dim=0)
        patches = torch.cat((results_dict0['patches'],), dim=0)
        
        #fusion && classify
        logits, Y_prob = self.fusion(slides, patches)
        loss_fusion = self.criterion_fusion(logits, y)

        #total loss
        # loss = loss2+loss1+loss0+loss_fusion
        loss = loss0+loss_fusion
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        self.log('t_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1, logger=True)
        self.log('t_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}
    
    def epoch_end_log(self, outputs, prefix):
        
        assert prefix == 't_' or prefix == 'v_' or prefix == 'test_'

        y = torch.cat([x["true"] for x in outputs])
        preds = torch.cat([x['pred'] for x in outputs])
        auc = self.auc(preds, y)
        f1 = self.f1(preds, y)
        self.log(prefix+'auc', auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log(prefix+'f1', f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)

        if prefix == 'v_':
            combined_metric = auc + f1
            self.log('v_auc_plus_f1', combined_metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)

    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 't_')

    def validation_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'v_')
    
    def test_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'test_')

        y = torch.cat([x["true"] for x in outputs])
        preds = torch.cat([x['pred'] for x in outputs])
        acc = self.acc(preds, y)
        auc = self.auc(preds, y)
        f1 = self.f1(preds, y)
        se = self.sensitivity(preds, y)
        sp = self.specificity(preds, y)
        print('testing...\n')
        print(f'acc: {acc:.4f}\n')
        print(f'auc: {auc:.4f}\n')
        print(f'f1_score: {f1:.4f}\n')
        print(f'se: {se:.4f}\n')
        print(f'sp: {sp:.4f}\n')

        # fold = self.config_dict['fold']
        k_sample = self.config_dict['k_sample']
        contra_weight = self.config_dict['contra_weight']
        seed = self.config_dict['seed']
        n_cluster = self.config_dict['n_cluster']

        with open(self.log_f, 'a') as f:
            f.write(f"main_9_folds_single: \n")
            f.write(f'seed: {seed}\n')
            f.write(f'n_cluster: {n_cluster}\n')
            f.write(f'k_sample: {k_sample}\n')
            f.write(f'contra_weight: {contra_weight}\n')
            f.write(f"Accuracy: {acc.item():.4f}\n")
            f.write(f"AUC: {auc.item():.4f}\n")
            f.write(f"F1 Score: {f1.item():.4f}\n")
            f.write(f"Sensitivity: {se.item():.4f}\n")
            f.write(f"Specificity: {sp.item():.4f}\n")
            f.write("\n")

    def validation_step(self, batch, batch_index):
        x2,x1,x0, y,y_, _ = batch
        y_ = y_[0]
        # logits, Y_prob, _, A_raw, results_dict2 = self.model2(x2, label=y, label_c=y_[0])
        # loss2 = self.bag_weight * self.criterion2(logits, y)
        # loss2 += self.contra_weight * results_dict2['contra_loss']
        
        # logits, Y_prob, _, A_raw, results_dict1 = self.model1(x1, label=y, label_c=y_[1])
        # loss1 = self.bag_weight * self.criterion1(logits, y)
        # loss1 += self.contra_weight * results_dict1['contra_loss']

        logits, Y_prob, _, A_raw, results_dict0 = self.model0(x0, label=y, label_c=y_[2])
        loss0 = self.bag_weight * self.criterion0(logits, y)
        loss0 += self.contra_weight * results_dict0['contra_loss']

        #concat
        # slides = torch.cat((results_dict2['slides'], results_dict1['slides'], results_dict0['slides']), dim=0)
        # patches = torch.cat((results_dict2['patches'], results_dict1['patches'], results_dict0['patches']), dim=0)
        slides = torch.cat((results_dict0['slides'],), dim=0)
        patches = torch.cat((results_dict0['patches'],), dim=0)
        
        #fusion && classify
        logits, Y_prob = self.fusion(slides, patches)
        loss_fusion = self.criterion_fusion(logits, y)

        #total loss
        # loss = loss2+loss1+loss0+loss_fusion
        loss =  loss0+loss_fusion
        
            
        self.log('v_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log('v_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}

    def test_step(self, batch, batch_index):
        x2,x1,x0, y,y_, _ = batch
        y_ = y_[0]
        # logits, Y_prob, _, A_raw, results_dict2 = self.model2(x2, label=y, label_c=y_[0])
        # loss2 = self.bag_weight * self.criterion2(logits, y)
        # loss2 += self.contra_weight * results_dict2['contra_loss']
        
        # logits, Y_prob, _, A_raw, results_dict1 = self.model1(x1, label=y, label_c=y_[1])
        # loss1 = self.bag_weight * self.criterion1(logits, y)
        # loss1 += self.contra_weight * results_dict1['contra_loss']

        logits, Y_prob, _, A_raw, results_dict0 = self.model0(x0, label=y, label_c=y_[2])
        loss0 = self.bag_weight * self.criterion0(logits, y)
        loss0 += self.contra_weight * results_dict0['contra_loss']

        #concat
        # slides = torch.cat((results_dict2['slides'], results_dict1['slides'], results_dict0['slides']), dim=0)
        # patches = torch.cat((results_dict2['patches'], results_dict1['patches'], results_dict0['patches']), dim=0)
        slides = torch.cat((results_dict0['slides'],), dim=0)
        patches = torch.cat((results_dict0['patches'],), dim=0)
        
        #fusion && classify
        logits, Y_prob = self.fusion(slides, patches)
        loss_fusion = self.criterion_fusion(logits, y)

        #total loss
        # loss = loss2+loss1+loss0+loss_fusion
        loss = loss0+loss_fusion
        
        
        self.log('test_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.config_dict['hyperparams_arguments']['lr'], 
                                weight_decay=self.config_dict['hyperparams_arguments']['reg'])
        return [optimizer]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Configurations for IAT Training')
    parser.add_argument('--n-cluster', type=int, default=2)
    parser.add_argument('--k-sample', type=float, default=0.05)
    parser.add_argument('--contra-weight', type=float, default=0.05)
    parser.add_argument(
        "--num_runs", type=int, default=5, help="number of runs with different random seeds"
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--config-path', type=str, default='configs/main_9_cm16_single.yaml',
                        help='Configuration path used in training the model')
    args = parser.parse_args()

    config_dict = yaml.safe_load(open(args.config_path, 'r'))
    num_runs = args.num_runs
    config_dict['contra_weight'] = args.contra_weight
    config_dict['n_cluster'] = args.n_cluster
    config_dict['k_sample'] = args.k_sample

    for i in range(num_runs):
        current_seed = args.seed + i
        config_dict['seed'] = current_seed
        
        utilities.seed.seed_everything(current_seed)

        model_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                    config_dict['storage_arguments']['task'],
                                    config_dict['storage_arguments']['model_dir'],
                                    config_dict['storage_arguments']['exp_code'])
        log_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                    config_dict['storage_arguments']['task'],
                                    config_dict['storage_arguments']['log_dir'])
                                
        # logger = pl_loggers.TensorBoardLogger(log_dir, config_dict['storage_arguments']['exp_code']+'_single')
        logger = pl_loggers.TensorBoardLogger(log_dir, config_dict['storage_arguments']['exp_code'])
        model = PLModule(config_dict)
        all_datasets = PLDataModule(config_dict) # Already initialized

        model_checkpoint = ModelCheckpoint(monitor='v_auc_plus_f1', 
                                            dirpath=os.path.join(model_dir,f'version_{logger.version}'),
                                            filename=config_dict['model_arguments']['model_type'] + '-{epoch:03d}-{v_auc_plus_f1:.3f}',
                                            save_top_k=config_dict['storage_arguments']['save_top_k'],verbose=True,mode='max'
                                            )
        #early_stopping = EarlyStopping(monitor="v_loss", min_delta=0.00, 
        #                            patience=config_dict['hyperparams_arguments']['early_stop'], verbose=True, mode="min")


        early_stopping = EarlyStopping(monitor="v_auc_plus_f1", min_delta=0.00, 
                                patience=config_dict['hyperparams_arguments']['early_stop'],verbose=True, mode="max")                   

        trainer = Trainer(
                        gpus=config_dict['hyperparams_arguments']['gpus'],
                        max_epochs=config_dict['hyperparams_arguments']['max_epoch'],
                        callbacks=[model_checkpoint, early_stopping],
                        logger=logger,
                        strategy=DDPPlugin(find_unused_parameters=True),
                        #strategy=DDPStrategy(find_unused_parameters=True),
                        log_every_n_steps=10,
                        )

        trainer.fit(model, all_datasets)
        trainer.test(ckpt_path="best",datamodule=all_datasets)