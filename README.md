# wsi-classification
A repository containing two AI models for whole-slide image (WSI) classification.
- This thesis proposes a pathological image classification method based on 
multi-scale key feature fusion. To address the insufficient recognition of key lesion 
regions at each level in existing multi-scale methods, this thesis proposes a multi-scale 
key feature fusion method that simulates the hierarchical diagnostic process of 
pathologists by leveraging the hierarchical structure of pathological images. At each 
scale, global features are first learned through an attention-based aggregator; 
subsequently, diverse key features are directly selected using instance feature 
clustering labels derived from attention scores at the current stage; finally, multi-scale 
key features and global features are fused via a cross-attention mechanism. 
Experiments are conducted, and the results demonstrate that this method can 
effectively identify diverse lesion regions at various hierarchical levels of pathological 
images and improve classification performance.
- This thesis proposes a pathological image classification method based 
on multi-stage key feature mining. To alleviate the problem that existing attenti
on aggregators focus on only a few salient features, this thesis proposes a classification method with multi-stage key feature mining. This method introduces 
a multi-stage instance aggregation process, uses a masking mechanism to occlu
de the identified key features, guides the model to mine different key features 
in the next stage, and thus learns the representations of different pathological p
atterns. Finally, the multi-stage pathological pattern representations are aggregate
d through the attention mechanism for final classification. Experiments of this 
method on two datasets and with two feature extractors prove its effectiveness. 
