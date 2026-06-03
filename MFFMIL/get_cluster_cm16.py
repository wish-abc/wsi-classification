from sklearn.cluster import KMeans
import torch
import tqdm
import os
import numpy as np
import warnings
import pandas as pd
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def load_test(split_path, label_path, split_type):
    # Load splits
    splits_df = pd.read_csv(split_path)
    split_ids = splits_df[split_type].dropna().tolist()

    # Load labels
    labels_df = pd.read_csv(label_path)
    # Create a map for faster lookup
    label_map = {row['slide_id']: 1 if row['label'] == 'tumor' else 0 for index, row in labels_df.iterrows()}

    # Get labels for the current split
    ids = []
    labels = []
    for slide_id in split_ids:
        if slide_id in label_map:
            ids.append(slide_id)
            labels.append(label_map[slide_id])

    return ids, labels

class OriginDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels):
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        label = self.labels[index]
        label = torch.tensor([label])

        f3_path = os.path.join(self.feature_root,'level3','pt_files',self.slide_ids[index]+'.pt')
        features_level_3 = torch.load(f3_path)

        return features_level_3, self.slide_ids[index]


split_path = 'splits/splits_0.csv'
label_path = 'splits/camelyon16.csv'
feat_path = 'feats_ours'

train_ids, train_labels= load_test(split_path, label_path, 'train')
trainset = OriginDataset(feat_path, train_ids, train_labels)

test_ids, test_labels = load_test(split_path, label_path, 'test')
testset = OriginDataset(feat_path, test_ids, test_labels)

val_ids, val_labels = load_test(split_path, label_path, 'val')
valset = OriginDataset(feat_path, val_ids, val_labels)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, drop_last=False)
valloader=torch.utils.data.DataLoader(valset,batch_size=1,drop_last=False)


for data in tqdm.tqdm(trainloader, total=len(trainloader)):
    inputs, name=data
    f_1 = inputs[0] #shape 109*256
    kmeans = KMeans(n_clusters=4, random_state=0).fit(f_1.cpu())
    labels = kmeans.labels_

    root = os.path.join(feat_path, 'level3', 'cluster_4')
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, name[0]), labels)

for data in tqdm.tqdm(testloader, total=len(testloader)):
    inputs, name=data
    f_1 = inputs[0] #shape 109*256
    kmeans = KMeans(n_clusters=4, random_state=0).fit(f_1.cpu())
    labels = kmeans.labels_

    root = os.path.join(feat_path, 'level3', 'cluster_4')
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, name[0]), labels)

       
for data in tqdm.tqdm(valloader, total=len(valloader)):
    inputs, name=data
    f_1 = inputs[0] #shape 109*256
    kmeans = KMeans(n_clusters=4, random_state=0).fit(f_1.cpu())
    labels = kmeans.labels_

    root = os.path.join(feat_path, 'level3', 'cluster_4')
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, name[0]), labels)

       