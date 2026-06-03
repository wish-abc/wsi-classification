from dataset.dataset_cluster import load_test, OriginDataset
from sklearn.cluster import KMeans
import torch
import tqdm
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

dataset
id_cat = {
   0:'海绵',1:'扁平苔藓',2:'银屑病'
}

train_ids, train_labels= load_test('splits/inflammation_trainval_fold.csv')
train_dataset = OriginDataset('data/clam_patch_1024', train_ids, train_labels, id_cat)

test_ids, test_labels = load_test('splits/inflammation_test.csv')
test_dataset = OriginDataset('data/clam_patch_1024', test_ids, test_labels, id_cat)


trainloader=torch.utils.data.DataLoader(train_dataset, batch_size=1, drop_last=False)
testloader=torch.utils.data.DataLoader(test_dataset, batch_size=1, drop_last=False)


for data in tqdm.tqdm(trainloader, total=len(trainloader)):
    inputs, _, f1_path=data
    f_1 = inputs[0][0] #shape 109*256
    kmeans = KMeans(n_clusters=8, random_state=0).fit(f_1.cpu())
    labels = kmeans.labels_

    cat, name = f1_path[0][0], f1_path[1][0]
    root = os.path.join('data/clam_patch_1024',f'{cat}_level3','cluster_8')
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, name), labels)

for data in tqdm.tqdm(testloader, total=len(testloader)):
    inputs, _, f1_path=data
    f_1 = inputs[0][0] #shape 109*256
    kmeans = KMeans(n_clusters=8, random_state=0).fit(f_1.cpu())
    labels = kmeans.labels_

    cat, name = f1_path[0][0], f1_path[1][0]
    root = os.path.join('data/clam_patch_1024',f'{cat}_level3','cluster_8')
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, name), labels)

       