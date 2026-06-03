
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
# from mamba_ssm import BiMamba

from models.model_utils import Attn_Net, Attn_Net_Gated, TransLayer
from models.loss import SupConLoss
from models.memory import Memory

def create_attention_net(feat_in, sz, dropout, gate):
    fc = [nn.Linear(feat_in, sz[1]), nn.GELU()]
    if dropout:
        fc.append(nn.Dropout(dropout))
    if gate:
        attention_net = Attn_Net_Gated(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    else:
        attention_net = Attn_Net(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    fc.append(attention_net)
    return nn.Sequential(*fc)

class IAMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim, size, dropout, gate):
        super(IAMBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.attention_net = create_attention_net(out_dim, size, dropout, gate)
        self.layer = TransLayer(dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer_map_to_final = nn.Sequential(nn.Linear(out_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())
        
    def aggregate(self, h, norm_func, attn_net):
        _h = norm_func(h).squeeze() #shape _h 118*1024
        A, _h = attn_net(_h)  # shape A 118*1, shape _h 118*1024
        A = torch.transpose(A, 1, 0)  # 1*118
        A_raw = A #shape A 1*118
        A = F.softmax(A, dim=1)  # softmax over N

        result = (torch.mul(h.squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0) #shape 1*1024
        slide_level = norm_func(result) #shape 1*1024
        return A, A_raw, slide_level, _h
    
    def forward(self, h):
        h = self.fc(h) #shape h 1*81*1024 -> 1*81*1024
        h = self.layer(h) #shape h 1*81*1024
        A, A_raw, slide_level, _h = self.aggregate(h, self.norm, self.attention_net)
        slide_level = self.layer_map_to_final(slide_level) #shape 1*1024
        return h, A, A_raw, slide_level, _h


class Mamatten(nn.Module):
    def __init__(self, dropout=0.25, n_classes=2, feat_in=1024, k_sample=2, **kwargs):
        super(Mamatten, self).__init__()
        self.n_classes = n_classes
        self.layer = 2
        self.k_sample = k_sample

        self.norm = nn.LayerNorm(512)
        #projection
        self.proj = nn.Sequential(
            nn.Linear(feat_in, 512),
            nn.ReLU(),
            nn.Dropout1d(dropout)
        )
        #mamba block
        self.mamba_layers = nn.ModuleList()
        for _ in range(self.layer):
            self.mamba_layers.append(
                nn.Sequential(
                    self.norm,
                    Mamba(
                        d_model=512,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )
        self.attention = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=1)
        self.atten = Attn_Net(L=512, D=256, dropout=dropout, n_classes=1)
        self.classifier = nn.Linear(512, n_classes)
        
        #add for supcon
        self.pos_mems = [Memory(512,40) for i in range(n_classes)]
        self.sup_loss = SupConLoss()

    def create_positive_targets_contrast(self, length, label, device):
        return torch.full((length,), label, device=device).long()

    def forward(self, h, label=None, label_c=None, instance_eval=False):

        h = h.unsqueeze(0) # 1*118*1024
        h = h.float().cuda() # 1*118*1024

        # encode
        h = self.proj(h) # 1*118*512
        for layer in self.mamba_layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h)
            h = h + h_
        h = self.norm(h)
        
        # aggregate
        h = h.squeeze(0) #shape h 
        h0 = h
        A, h = self.attention(h)  # shape A 118*1, shape _h 118*512
        A = torch.transpose(A, 1, 0)  # 1*118
        A_raw = A #shape A 1*118
        A = F.softmax(A, dim=1)  # softmax over N

        h = (torch.mul(h.squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0) #shape 1*512
        h = self.norm(h)

        # select patches
        label_c = torch.from_numpy(label_c).cuda()
        unique_clusters = torch.unique(label_c)
        selected_patches = []

        #for cluster in unique_clusters:
        #    cluster_indices = torch.nonzero(label_c == cluster, as_tuple=True)[0]
        #    cluster_attention_scores = A[0, cluster_indices]
        #    if len(cluster_indices) < self.k_sample:
        #        repeat_times = (self.k_sample + len(cluster_indices) - 1) // len(cluster_indices)
        #        cluster_indices = cluster_indices.repeat(repeat_times)[:self.k_sample]
        #        cluster_attention_scores = cluster_attention_scores.repeat(repeat_times)[:self.k_sample]
        #    topk_indices = torch.topk(cluster_attention_scores, k=self.k_sample, dim=0).indices
        #    selected_patch_indices = cluster_indices[topk_indices]
        #    selected_patches.append(h0[selected_patch_indices])
        for cluster in unique_clusters:
            cluster_indices = torch.nonzero(label_c == cluster, as_tuple=False).squeeze(1)
            cluster_attention_scores = A[0, cluster_indices]

            k = max(1, int(len(cluster_indices)*self.k_sample))
            topk_indices = torch.topk(cluster_attention_scores, k=k, dim=0).indices
            selected_patch_indices = cluster_indices[topk_indices]
            selected_patches.append(h0[selected_patch_indices])

        selected_patches = torch.cat(selected_patches, dim=0)

        #contrast
        device = h0.device
        A1, h1 = self.atten(selected_patches)  # shape A 118*1, shape _h 118*512
        A1 = torch.transpose(A1, 1, 0)  # 1*118
        A1 = F.softmax(A1, dim=1)  # softmax over N
        pa_region = (torch.mul(h1.squeeze().T, A1.squeeze())).sum(dim=1).unsqueeze(0) #shape 1*512
        pa_region = self.norm(pa_region)
        pa_target = self.create_positive_targets_contrast(1, label.item(), device)  # [1]
        
        all_samples = [F.normalize(pa_region, dim=1)]
        all_labels = [pa_target] 
        for i in range(self.n_classes):
            pos_sample = F.normalize(self.pos_mems[i]._return_queue().detach(),dim=1)
            pos_label = self.create_positive_targets_contrast(pos_sample.shape[0], i, device)
            all_labels.append(pos_label.to(device))
            all_samples.append(pos_sample.to(device))
        contra_samples = torch.cat(all_samples, dim=0).unsqueeze(dim=1)
        contra_labels = torch.cat(all_labels, dim=0)

        contra_loss = self.sup_loss(contra_samples, contra_labels)
        self.pos_mems[i]._dequeue_and_enqueue(pa_region)
        #end contrast

        #classifier
        logits = self.classifier(h) #[B, n_classes]   1*3
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {
            'slides': h,
            'patches': selected_patches,
            'contra_loss': contra_loss
        }

        
        return logits, Y_prob, Y_hat, A_raw, results_dict
    