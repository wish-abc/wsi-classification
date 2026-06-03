import argparse
import os
import yaml

# internal imports
from datasets.dataset_module import PLDataModule
from models_latest.model import IATModel
from models_latest.fusion import Fusion_Block

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

# loss function imports
#from topk.svm import SmoothTop1SVM

def expand_indices(indices):
    expanded = []
    for index in indices:
        expanded.append(index.item() * 4)
        expanded.append(index.item() * 4 + 1)
        expanded.append(index.item() * 4 + 2)
        expanded.append(index.item() * 4 + 3)
    expanded = torch.tensor(expanded).long().to(indices.device)
    return expanded

class PLModule(LightningModule):
    def __init__(self, config_dict, k_sample):
        super(PLModule, self).__init__()
        self.config_dict = config_dict
        self.automatic_optimization = False

        if config_dict['model_arguments']['inst_loss'] == 'svm':
            pass
            #instance_loss_fn = SmoothTop1SVM(n_classes=config_dict['model_arguments']['n_classes'])
        elif config_dict['model_arguments']['inst_loss'] == 'ce':
            instance_loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        if config_dict['model_arguments']['bag_loss'] == 'ce':
            self.criterion_level_2 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_level_1 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_level_0 = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.criterion_slides = nn.CrossEntropyLoss()
        elif config_dict['model_arguments']['bag_loss'] == 'svm':
            pass
            #self.criterion = SmoothTop1SVM(n_classes=config_dict['model_arguments']['n_classes'])
        else:
            raise NotImplementedError
        instance_loss_fn = nn.CrossEntropyLoss()
        self.k_sample = k_sample
        self.model0 = IATModel(**config_dict['model_arguments'], k_sample=self.k_sample*4, instance_loss_fn=instance_loss_fn)
        self.model1 = IATModel(**config_dict['model_arguments'], k_sample=self.k_sample*2, instance_loss_fn=instance_loss_fn)
        self.model2 = IATModel(**config_dict['model_arguments'], k_sample=self.k_sample, instance_loss_fn=instance_loss_fn)
        #TODO
        self.fusion = Fusion_Block()
        
        self.acc = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.acc_avg = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'])
        self.acc_list = Accuracy(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average=None)
        self.auc = AUROC(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.f1 = F1Score(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.sensitivity = Recall(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')
        self.specificity = Specificity(task='multiclass', num_classes=config_dict['model_arguments']['n_classes'], average='macro')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
    
    def training_step(self, batch, batch_index):
        x2, x1, x0, y, _ = batch #x2 118 x1 472 x0 1888
        opt = self.optimizers()
        opt.zero_grad()

        x2, x1, x0, y, _ = batch
        logits, results_dict, h2, patches2, A_raw2, key_instances2 = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss2 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_2(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss2 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss2 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h1, patches1, A_raw1, key_instances1 = self.model1(x1, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss1 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_1(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss1 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss1 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h0, patches0, A_raw0, key_instances0 = self.model0(x0, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss0 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss0 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss0 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        #concatenate key patches
        #derive key patches from single level
        input_patches = torch.cat((key_instances2,key_instances1,key_instances0),dim=0)
        input_slides = torch.cat((h2,h1,h0),dim=0)

        try:
            logits, Y_prob = self.fusion(input_slides, input_patches)
            loss_patch_slide = self.criterion_slides(logits, y)
            loss = loss0+loss1+loss2+config_dict['model_arguments']['bag_weight']*loss_patch_slide
            self.manual_backward(loss)
            opt.step()

            self.log('t_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1, logger=True)
            self.log('t_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
            return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}
        except:
            pass
        
    
    def epoch_end_log(self, outputs, prefix):
        
        assert prefix == 't_' or prefix == 'v_' or prefix == 'test_'

        y = torch.cat([x["true"] for x in outputs])
        preds = torch.cat([x['pred'] for x in outputs])
        auc = self.auc(preds, y)
        f1 = self.f1(preds, y)
        se = self.sensitivity(preds, y)
        sp = self.specificity(preds, y)
        
        acc_avg = self.acc_avg(preds, y)
        acc_list = self.acc_list(preds, y)
        print(f'acc_avg: {acc_avg}')
        print(f'auc: {auc}')
        #print(f'acc_list: {acc_list}')
        print(f'f1_score: {f1}')
        print(f'se: {se}')
        print(f'sp: {sp}')
        self.log(prefix+'auc', auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, logger=True)
        self.log(prefix+'f1', f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
        

    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 't_')

    def validation_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'v_')
    
    def test_epoch_end(self, outputs):
        self.epoch_end_log(outputs, 'test_')

    def validation_step(self, batch, batch_index):
        x2, x1, x0, y, _ = batch
        logits, results_dict, h2, patches2, A_raw2, key_instances2  = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss2 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_2(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss2 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss2 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h1, patches1, A_raw1, key_instances1  = self.model1(x1, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss1 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_1(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss1 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss1 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h0, patches0, A_raw0, key_instances0 = self.model0(x0, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss0 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss0 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss0 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']


        #concatenate key patches
        #derive key patches from single level
        input_patches = torch.cat((key_instances2,key_instances1,key_instances0),dim=0)
        input_slides = torch.cat((h2,h1,h0),dim=0)

        try:
            logits, Y_prob = self.fusion(input_slides, input_patches)
            loss_patch_slide = self.criterion_slides(logits, y)
            loss = loss0+loss1+loss2+config_dict['model_arguments']['bag_weight']*loss_patch_slide

            self.log('v_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1, logger=True)
            self.log('v_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
            return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}
        except:
            pass


    def test_step(self, batch, batch_index):
        x2, x1, x0, y, _ = batch
        logits, results_dict, h2, patches2, A_raw2, key_instances2 = self.model2(x2, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss2 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_2(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss2 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss2 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h1, patches1, A_raw1, key_instances1 = self.model1(x1, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss1 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_1(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss1 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss1 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        logits, results_dict, h0, patches0, A_raw0, key_instances0 = self.model0(x0, label=y, instance_eval=config_dict['model_arguments']['inst_cluster'])
        loss0 = config_dict['model_arguments']['bag_weight'] * self.criterion_level_0(logits, y)
        if config_dict['model_arguments']['inst_cluster']:
            loss0 += config_dict['model_arguments']['inst_weight'] * results_dict['instance_loss']
            loss0 += config_dict['model_arguments']['contra_weight'] * results_dict['contra_loss']

        #concatenate key patches
        #derive key patches from single level
        input_patches = torch.cat((key_instances2,key_instances1,key_instances0),dim=0)
        input_slides = torch.cat((h2,h1,h0),dim=0)

        try:
            logits, Y_prob = self.fusion(input_slides, input_patches)
            loss_patch_slide = self.criterion_slides(logits, y)
            loss = loss0+loss1+loss2+config_dict['model_arguments']['bag_weight']*loss_patch_slide

            self.log('test_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1, logger=True)
            self.log('test_acc', self.acc(Y_prob, y), on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True)
            return {'loss': loss, 'true': y, 'pred': Y_prob.detach()}
        except:
            pass
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.config_dict['hyperparams_arguments']['lr'], 
                                weight_decay=self.config_dict['hyperparams_arguments']['reg'])
        return [optimizer]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Configurations for IAT Training')
    parser.add_argument('--config-path', type=str, default='configs/main_cm16.yaml',
                        help='Configuration path used in training the model')
    parser.add_argument('--k-sample', type=int, default=8)
    parser.add_argument('--ckpt-path', type=str)

    args = parser.parse_args()

    k_sample = args.k_sample
    print(f'k_sample is {k_sample}')
    config_dict = yaml.safe_load(open(args.config_path, 'r'))

    utilities.seed.seed_everything(config_dict['hyperparams_arguments']['seed'])

    model_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                config_dict['storage_arguments']['task'],
                                config_dict['storage_arguments']['model_dir'],
                                config_dict['storage_arguments']['exp_code'])
    log_dir = os.path.join(config_dict['storage_arguments']['result_dir'], 
                                config_dict['storage_arguments']['task'],
                                config_dict['storage_arguments']['log_dir'])
                            
    logger = pl_loggers.TensorBoardLogger(log_dir, config_dict['storage_arguments']['exp_code'])
    #model = PLModule(config_dict, k_sample=k_sample)
    model = PLModule.load_from_checkpoint(args.ckpt_path, config_dict=config_dict,k_sample=k_sample).to('cuda:0')
    all_datasets = PLDataModule(config_dict)

    model_checkpoint = ModelCheckpoint(monitor='v_acc', 
                                        dirpath=os.path.join(model_dir,f'version_{logger.version}'),
                                        filename=config_dict['model_arguments']['model_type'] + '-{epoch:03d}-{v_acc:.3f}',
                                        save_top_k=config_dict['storage_arguments']['save_top_k'],verbose=True,mode='max'
                                        )
    #early_stopping = EarlyStopping(monitor="v_loss", min_delta=0.00, 
    #                            patience=config_dict['hyperparams_arguments']['early_stop'], verbose=True, mode="min")
    
    
    early_stopping = EarlyStopping(monitor="v_acc", min_delta=0.00, 
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

    #trainer.fit(model, all_datasets)
    trainer.test(model=model,datamodule=all_datasets)
