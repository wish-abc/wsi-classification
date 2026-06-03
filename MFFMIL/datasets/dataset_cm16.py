import csv
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset

class OriginDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels):
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        self.labels = labels

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        label = self.labels[index]
        label = torch.tensor([label])

        f3_path = os.path.join(self.feature_root,'level3','pt_files',self.slide_ids[index]+'.pt')
        f2_path = os.path.join(self.feature_root,'level2','pt_files',self.slide_ids[index]+'.pt')
        f1_path = os.path.join(self.feature_root,'level1','pt_files',self.slide_ids[index]+'.pt')
        features_level_3 = torch.load(f3_path)
        features_level_2 = torch.load(f2_path)
        features_level_1 = torch.load(f1_path)

        f3_path = os.path.join(self.feature_root,'level3','cluster_4',self.slide_ids[index]+'.npy')
        f2_path = os.path.join(self.feature_root,'level2','cluster_4',self.slide_ids[index]+'.npy')
        f1_path = os.path.join(self.feature_root,'level1','cluster_4',self.slide_ids[index]+'.npy')
        y3_ = np.load(f3_path)
        y2_ = np.load(f2_path)
        y1_ = np.load(f1_path)

        return features_level_3, features_level_2,features_level_1, label,[y3_,y2_,y1_], self.slide_ids[index]

    def __len__(self):
        return len(self.slide_ids)

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