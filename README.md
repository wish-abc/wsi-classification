# Key Feature Fusion for Pathological Image Multi-Instance Learning Classification
This repository contains the implementation of two multi-instance learning (MIL) methods for pathological image classification, which are the core contributions of the master's thesis "Research on Multi-Instance Learning Classification Method of Pathological Images Based on Key Feature Fusion" (University of Chinese Academy of Sciences, 2026).

## Method 1: Multi-Scale Key Feature Fusion (MFF-MIL)

To address the insufficient recognition of key lesion regions at each level in existing multi-scale methods, we propose a multi-scale key feature fusion method that simulates the hierarchical diagnostic process of pathologists by leveraging the hierarchical structure of pathological images. At each scale, global features are first learned through an attention-based aggregator; subsequently, diverse key features are directly selected using instance feature clustering labels derived from attention scores at the current stage; finally, multi-scale key features and global features are fused via a cross-attention mechanism.

## Method 2: Multi-Stage Key Feature Mining (MSM-MIL)

To alleviate the problem that existing attention aggregators focus on only a few salient features, we propose a classification method with multi-stage key feature mining. This method introduces a multi-stage instance aggregation process, uses a masking mechanism to occlude the identified key features, guides the model to mine different key features in the next stage, and thus learns the representations of different pathological patterns. Finally, the multi-stage pathological pattern representations are aggregated through the attention mechanism for final classification.

## Datasets
- **ISDI**: An internal inflammatory skin disease dataset from Beijing Hospital containing 1,091 WSIs across three categories (spongiosis: 362, lichen planus: 357, psoriasis: 437).
- **Camelyon16**: A public benchmark dataset of 397 breast cancer lymph node WSIs for tumor metastasis detection (normal: 157 train + test, tumor: 111 train + test). Dataset could be available from **[site](
https://camelyon16.grand-challenge.org)**.

## Experimental Steps
### steps for MFFMIL
### step1 create patches && extract features
refer to **[CLAM](https://github.com/mahmoodlab/CLAM)** to run the follow scripts. For ISDI dataset, we use multi-scale features at level 1,2,3. For Camelyon16 dataset, we use single-scale features at level 1.
```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch --patch_level 1
```
```bash
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .mrxs

```
### step2 get cluster labels for features
run script get_cluster_n.py for dataset ISDI and run script get_cluster_cm16.py for dataet Camelyon16.
```bash
python get_cluster_n.py
```

### step3 train && test
After obtaining features and feature clustering labels, run script main_ISDI.py for dataset ISDI and main_cm16.py for dataset Camelyon16.
```bash
python main_ISDI.py
```


### steps for MSMMIL
We create patches and extract features at level 1 for both two dataset.
### step1 create patches for each wsi
```bash
python Step1_create_patches_fp.py
```
### step2 extract features for patches
Features for dataset Camelyon16 could be available from method **[ACMIL(https://github.com/dazhangyu123/ACMIL)**
```bash
python Step2_feature_extract.py
```
### step3 train&&test
```bash
python Step3_WSI_classification_MSMAMIL.py
```

## Acknowledgments

This work builds upon and extends several open-source projects. We gratefully acknowledge the following:

- **[ACMIL](https://github.com/dazhangyu123/ACMIL)**
- **[HAGMIL](https://github.com/BearCleverProud/HAG-MIL)**


