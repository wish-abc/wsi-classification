import pytorch_lightning as pl
from datasets.dataset_generic import Generic_MIL_Dataset
from utils import get_loader_no_sampler, get_not_shuffled_loader
import os

def get_dataset(config_dict):
    dataset = Generic_MIL_Dataset(csv_path = config_dict['data_arguments']['ground_truth_csv'],
                                data_dir= config_dict['data_arguments']['feature_dir'],
                                shuffle = config_dict['data_arguments']['shuffle_data'], 
                                seed = config_dict['hyperparams_arguments']['seed'], 
                                print_info = config_dict['data_arguments']['print_info'],
                                label_dict = config_dict['data_arguments']['label_dict'],
                                patient_strat=config_dict['data_arguments']['patient_strat'],
                                ignore=[])
    return dataset.return_splits(from_id=False, 
                csv_path=os.path.join(config_dict['data_arguments']['split_dir'], 'splits_0.csv'))


def get_dataset_HAG_tvt(config_dict):
    id_cat = {
        0:'海绵',1:'扁平苔藓',2:'银屑病'
    }

    if config_dict['exp_name']=='5x':
        from datasets.dataset_5x import OriginDataset, load_test
    elif config_dict['exp_name']=='10x':
        from datasets.dataset_10x import OriginDataset, load_test
    elif config_dict['exp_name']=='20x':
        from datasets.dataset_20x import OriginDataset, load_test
    elif config_dict['exp_name']=='fusion_slide_patch':
        from datasets.dataset_slide_patch import OriginDataset, load_test
    else:
        from datasets.dataset_new import OriginDataset, load_test

    train_ids, train_labels= load_test('splits/inflammation_train.csv')
    trainset = OriginDataset('data/clam_patch_1024', train_ids, train_labels, id_cat)

    val_ids, val_labels= load_test('splits/inflammation_val.csv')
    valset = OriginDataset('data/clam_patch_1024', val_ids, val_labels, id_cat)

    test_ids, test_labels = load_test('splits/inflammation_test.csv')
    testset = OriginDataset('data/clam_patch_1024', test_ids, test_labels, id_cat)
    
    
    return trainset,valset,testset



def get_dataset_HAG_fold_cm16(config_dict):
    #id_cat = {
        #0:'海绵',1:'扁平苔藓',2:'银屑病'
    #}

    if config_dict['exp_name']=='5x':
        from datasets.dataset_5x import OriginDataset, load_test
    elif config_dict['exp_name']=='10x':
        from datasets.dataset_10x import OriginDataset, load_test
    elif config_dict['exp_name']=='20x':
        from datasets.dataset_20x import OriginDataset, load_test
    elif config_dict['exp_name']=='fusion_slide_patch':
        from datasets.dataset_slide_patch import OriginDataset, load_test
    elif config_dict['exp_name']=='cm16':
        from datasets.dataset_cm16 import OriginDataset, load_test
    else:
        from datasets.dataset_new import OriginDataset, load_test

    split_path = 'splits/camelyon16/splits_0.csv'
    # split_path = 'splits/camelyon16/split_1.csv'
    label_path = 'csvs/camelyon16.csv'
    feat_path = 'feats_cm16/feats_ours'

    train_ids, train_labels= load_test(split_path, label_path, 'train')
    trainset = OriginDataset(feat_path, train_ids, train_labels)

    test_ids, test_labels = load_test(split_path, label_path, 'test')
    testset = OriginDataset(feat_path, test_ids, test_labels)

    val_ids, val_labels = load_test(split_path, label_path, 'val')
    valset = OriginDataset(feat_path, val_ids, val_labels)

    return trainset,valset,testset

class PLDataModule(pl.LightningDataModule):
    def __init__(self, config_dict):
        #self.train_dataset, self.val_dataset, self.test_dataset = get_dataset_HAG_tvt(config_dict)
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset_HAG_fold_cm16(config_dict)
        
        self.val_dataset = get_not_shuffled_loader(self.val_dataset)
        self.test_dataset = get_not_shuffled_loader(self.test_dataset)
        self._log_hyperparams = None

    def train_dataloader(self):
        return get_loader_no_sampler(self.train_dataset)

    def prepare_data_per_node(self):
        pass
    
    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
    
