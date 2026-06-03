from torch import nn, Tensor
import torch
import torch.nn.functional as F
from models.linearatt import MultiheadLinearAttention


class CrossLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.attn = MultiheadLinearAttention(embed_dim=dim,num_heads=8,dropout=d)

    def forward(self,q,k,v):

        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        x,attention= self.attn(q,k,v)

        return x.permute(1,0,2), attention
        
class Fusion_Block(nn.Module):
    def __init__(
        self,
        in_channel: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.crossattention =  CrossLayer(dim=512,d=0.25)
        #self.conv_11 = nn.Sequential(
        #    nn.Conv2d(in_channel, 1, 1, 1, 1//2, bias=False),
        #    nn.BatchNorm2d(1),
        #    nn.ReLU(inplace=True),
        #)
        self.conv_11 = nn.Sequential(
            nn.Conv1d(in_channel,in_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            # nn.Linear(512, 3)
            nn.Linear(512, 2)
        )
        
    def forward(self, slides: Tensor, patches: Tensor):
        
        '''
        slides: 3*512
        patches: 56*512
        '''
        slides0 = slides

        slides, _ = self.crossattention(slides.unsqueeze(0), patches.unsqueeze(0),patches.unsqueeze(0))
        slides = slides.squeeze(0)+slides0
        slides = self.norm(slides)
        slides_ = torch.mean(slides, dim=0, keepdim=True)

        patches_ = self.conv_11(patches.unsqueeze(0).transpose(1,2)).transpose(1,2).squeeze(0)
        patches_ = torch.mean(patches_,dim=0,keepdim=True)

        h_ = slides_ + patches_

        logits = self.classifier(h_)
        prob = F.softmax(logits, dim=1)

        return logits, prob

        