# Key Feature Fusion for Pathological Image Multi-Instance Learning Classification
This repository contains the implementation of two multi-instance learning (MIL) methods for pathological image classification.

## Method 1: Multi-Scale Key Feature Fusion (MFF-MIL)

To address the insufficient recognition of key lesion regions at each level in existing multi-scale methods, we propose a multi-scale key feature fusion method that simulates the hierarchical diagnostic process of pathologists by leveraging the hierarchical structure of pathological images. At each scale, global features are first learned through an attention-based aggregator; subsequently, diverse key features are directly selected using instance feature clustering labels derived from attention scores at the current stage; finally, multi-scale key features and global features are fused via a cross-attention mechanism.

## Method 2: Multi-Stage Key Feature Mining (MSM-MIL)

To alleviate the problem that existing attention aggregators focus on only a few salient features, we propose a classification method with multi-stage key feature mining. This method introduces a multi-stage instance aggregation process, uses a masking mechanism to occlude the identified key features, guides the model to mine different key features in the next stage, and thus learns the representations of different pathological patterns. Finally, the multi-stage pathological pattern representations are aggregated through the attention mechanism for final classification.

## Datasets
- **ISDI**: An internal inflammatory skin disease dataset from Beijing Hospital containing 1,091 WSIs across three categories (spongiosis: 362, lichen planus: 357, psoriasis: 437).
- **Camelyon16**: A public benchmark dataset of 397 breast cancer lymph node WSIs for tumor metastasis detection (normal: 157 train + test, tumor: 111 train + test).

## Experimental Steps

## Acknowledgments

This work builds upon and extends several open-source projects. We gratefully acknowledge the following:

- **[ACMIL](https://github.com/dazhangyu123/ACMIL)**


