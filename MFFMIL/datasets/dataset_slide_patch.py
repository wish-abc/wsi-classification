import csv
import os
import random
import re
import joblib
import torch
import numpy as np
import torch.nn.functional as F
import glob
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class FeatDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels):
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        self.labels = labels

    def __getitem__(self, index):
        return (torch.from_numpy(torch.load(os.path.join(self.feature_root, self.slide_ids[index] + '.pth'))),
                self.labels[index])

    def __len__(self):
        return len(self.slide_ids)


class OriginDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels, id_cat):
        self.feature_list = []
        self.bag_all = []
        
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        #self.slide_ids = slide_ids[:10]
        self.labels = labels
        self.id_cat = id_cat
    

    
    def __getitem__(self, index):
        cat_id = self.labels[index]
        cat = self.id_cat[cat_id]
        label = torch.tensor([self.labels[index]])

        f3_path = os.path.join(self.feature_root,cat+'_level3','feats_level3','pt_files',self.slide_ids[index]+'.pt')
        f2_path = os.path.join(self.feature_root,cat+'_level2','feats_level2','pt_files',self.slide_ids[index]+'.pt')
        f1_path = os.path.join(self.feature_root,cat+'_level1','feats_level1','pt_files',self.slide_ids[index]+'.pt')
        features_level_3 = torch.load(f3_path)
        features_level_2 = torch.load(f2_path)
        features_level_1 = torch.load(f1_path)

        #f3_path = os.path.join(self.feature_root,cat+'_level3','cluster_4',self.slide_ids[index]+'.npy')
        #f2_path = os.path.join(self.feature_root,cat+'_level2','cluster_4',self.slide_ids[index]+'.npy')
        #f1_path = os.path.join(self.feature_root,cat+'_level1','cluster_4',self.slide_ids[index]+'.npy')

        f3_path = os.path.join(self.feature_root,cat+'_level3','cluster_8',self.slide_ids[index]+'.npy')
        f2_path = os.path.join(self.feature_root,cat+'_level2','cluster_8',self.slide_ids[index]+'.npy')
        f1_path = os.path.join(self.feature_root,cat+'_level1','cluster_8',self.slide_ids[index]+'.npy')
        y3_ = np.load(f3_path)
        y2_ = np.load(f2_path)
        y1_ = np.load(f1_path)

        #name = self.slide_ids[index]
        #match = re.search(r'\d{4,5}', name)
        #case = match.group()
        return features_level_3, features_level_2,features_level_1, label,[y3_,y2_,y1_], self.slide_ids[index]



    def __len__(self):
        return len(self.slide_ids)


def load_test(csvfile):
    ids, labels = [], []

    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        for idx, content in enumerate(reader):
            slide_id = content[0]
            label = int(content[1])

            ids.append(slide_id)
            labels.append(label)

    return ids, labels

