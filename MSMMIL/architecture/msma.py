import math
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from einops import repeat
from .nystrom_atten import NystromAttention
from modules.emb_position import *

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class Attention_Aggregator(nn.Module):
    def __init__(self, L=512, D=128, K=1, dropout_rate=0.25):
        super(Attention_Aggregator, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),
        )


    def forward(self, x):
        A_U = self.attention(x)  # NxK 
        A = torch.transpose(A_U, 1, 0)  # KxN

        return A  ### K x N

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MHA(nn.Module):
    def __init__(self, conf):
        super(MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        feat, attn = self.attention(q, k, v)
        output = self.classifier(feat)

        return output


class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape,device=attn.device), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c,device=attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout= d
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

# Simple masked attention module 
class MAM(nn.Module):
    def __init__(self, L, dim=128, d=0.25, n_masked_patch=5, mask_drop=1):
        #Linear, ReLU, Dropout, 
        #LayerNorm,NystromAttention, 
        #Attention_Gated
        #mask 
        super(MAM, self).__init__()
        self.L = L
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.encoder = nn.Sequential(
            TransLayer(dim=self.L,d=d),
            nn.LayerNorm(self.L),
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.L,self.L),
            nn.ReLU(),
            nn.Linear(self.L,self.L),
            nn.Dropout(d),
        )
        self.attn = Attention_Gated(self.L, dim, 1) 
    
    def forward(self, x, mask0):
        #x: N x L 1000*256,   mask0: 1*1000
        x = self.ffn(x) # 1000*256
        x = self.encoder(x.unsqueeze(0)).squeeze(0) # 1*1000*256 #N x L 
        A = self.attn(x) #1*1000

        k, n = A.shape
        mask1 = torch.ones(k, n, device=A.device)
        
        if self.n_masked_patch > 0 and self.training:
            #mask 
            A = A.masked_fill(mask0 == 0, -1e9)

            # Get the indices of the top-k largest values
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape,device=A.device), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n,device=A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            # A = A.masked_fill(random_mask == 0, -1e9)

            mask1 = mask0 * random_mask # 1*1000

        bag = torch.mm(F.softmax(A, dim=1), x)  ## K x L 1*256
        return x, A, bag, mask1



#multi satge masked attention mil
class MSMAMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token=2, n_masked_patch=5, mask_drop=1):
        super(MSMAMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.mask_nets = nn.ModuleList()
        self.classifier = nn.ModuleList()

        self.attention0 = Attention_Gated(conf.D_inner, D, 1)

        #one ABMIL + n_token MAMs
        for i in range(n_token+1):
            if i==0:
                self.mask_nets.append(self.attention0)
            else:
                self.mask_nets.append(MAM(conf.D_inner, D, droprate, n_masked_patch, mask_drop))
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))

        self.last_attention_aggregator = Attention_Aggregator(conf.D_inner, D, 1, dropout_rate=droprate)
        self.n_masked_patch = n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = mask_drop


    def forward(self, x): ## x: N x L
        x = x[0] #x: 1000*512
        x = self.dimreduction(x) #x: 1000*256

        # Multi-stage masked attention
        feats_pattern = []
        outputs = []
        for i, net in enumerate(self.mask_nets):
            if i==0: #first ABMIL
                A0 = F.softmax(net(x), dim=1)## K x N 1*1000
                bag0 = torch.mm(A0, x)  ## K x L 1*256
                k, n = A0.shape # k=1, n=1000
                mask = torch.ones(k, n, device=A0.device)
                
                if self.n_masked_patch > 0 and self.training:
                    _, indices = torch.topk(A0, self.n_masked_patch, dim=-1)
                    rand_selected = torch.argsort(torch.rand(*indices.shape,device=A0.device), dim=-1)[:,:int(self.n_masked_patch * self.mask_drop)]
                    masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
                    random_mask = torch.ones(k, n, device=A0.device)
                    random_mask.scatter_(-1, masked_indices, 0)
                    mask = random_mask # 1* 1000
            
                feats_pattern.append(bag0)
                outputs.append(self.classifier[i](bag0))
            else: #MAMs
                x, A, bag_mam, mask = net(x, mask)
                # x = x_ + x
                feats_pattern.append(bag_mam)
                outputs.append(self.classifier[i](bag_mam))

        feats_patterns = torch.concat(feats_pattern, dim=0) # (1+n_token)*L
        #last Attention aggregation
        A_last = self.last_attention_aggregator(feats_patterns)  ## K x N  1*(1+n_token)
        bag_last = torch.mm(F.softmax(A_last, dim=1), feats_patterns)  ## K x L

        return torch.concat(outputs, dim=0), self.Slide_classifier(bag_last), A.unsqueeze(0)
    
    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0] #x: 1000*512
        x = self.dimreduction(x) #x: 1000*256

        # Multi-stage masked attention
        feats_pattern = []
        outputs = []
        attns = []
        for i, net in enumerate(self.mask_nets):
            if i==0: #first ABMIL
                A_ = net(x)
                A0 = F.softmax(A_, dim=1)## K x N 1*1000
                bag0 = torch.mm(A0, x)  ## K x L 1*256
                k, n = A0.shape # k=1, n=1000
                mask = torch.ones(k, n, device=A0.device)
                
                if self.n_masked_patch > 0 and self.training:
                    _, indices = torch.topk(A0, self.n_masked_patch, dim=-1)
                    rand_selected = torch.argsort(torch.rand(*indices.shape,device=A0.device), dim=-1)[:,:int(self.n_masked_patch * self.mask_drop)]
                    masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
                    random_mask = torch.ones(k, n, device=A0.device)
                    random_mask.scatter_(-1, masked_indices, 0)
                    mask = random_mask # 1* 1000
            
                feats_pattern.append(bag0)
                outputs.append(self.classifier[i](bag0))
                attns.append(A_)
            else: #MAMs
                x, A, bag_mam, mask = net(x, mask)
                # x = x_ + x
                feats_pattern.append(bag_mam)
                outputs.append(self.classifier[i](bag_mam))
                attns.append(A)

        attns = torch.stack(attns, dim=0)
        feats_patterns = torch.concat(feats_pattern, dim=0) # (1+n_token)*L
        #last Attention aggregation
        A_last = self.last_attention_aggregator(feats_patterns)  ## K x N  1*(1+n_token)
        bag_last = torch.mm(F.softmax(A_last, dim=1), feats_patterns)  ## K x L

        return torch.concat(outputs, dim=0), self.Slide_classifier(bag_last), attns.unsqueeze(0)

if __name__ == "__main__":
    model = MAM(256, dim=128, d=0.25)
    x = torch.randn(1000, 256)
    mask = torch.ones(1, 1000)
    mask[0,:10] = 0
    x, A, bag_mam, mask = model(x, mask)
    print(x.shape)