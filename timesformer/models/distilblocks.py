from timesformer.models.modules import Attention, Mlp
from timesformer.models.vit_utils import DropPath, trunc_normal_

from einops import rearrange
from torch import nn
import torch

class FeatureDistillationBlock(nn.Module):
    '''
    For the distilled feature auxtask
    '''
    def __init__(self, norm_layer=nn.LayerNorm, embed_dim=768, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0, 
                 drop=0, act_layer=nn.GELU, mlp_ratio=4, drop_path=0.1, num_joints=13, temporal_pooling=False):
        super().__init__()

        self.proj = nn.Linear(embed_dim, 216)
        self.temporal_pooling = temporal_pooling

    def forward(self, x, B, T, H, W):
        '''
        Arguments
        x : torch.Tensor
            The input tensor of shape (B, TN+1, 768)
        B : int
            The batch size
        T : int
            The number of frames
        H : int
            The height of the input
        W : int
            The width of the input
        '''
        x = x[:, 1:, :]

        x = rearrange(x, 'b (t h w) d -> b t (h w) d', h=H, w=W) # (B, T, HW, 768)

        if self.temporal_pooling:
            x = x.mean(dim=2).mean(dim=1) # (B, 768)
        else:
            x = x.mean(dim=2) # (B, T, 768)

        x = self.proj(x) # (B, 216) or (B, T, 216)

        return x

class FeatureDistillationBlock_classifier(nn.Module):
    '''
    For the distilled feature classification
    '''
    def __init__(self, norm_layer=nn.LayerNorm, embed_dim=768, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0, 
                 drop=0, act_layer=nn.GELU, mlp_ratio=4, drop_path=0.1, num_joints=13, temporal_pooling=False, num_classes=-1):
        super().__init__()

        self.proj = nn.Linear(embed_dim, 216)
        # print('Debugging feature: Initialize projection weights from truncated normal distribution to attempt to fix exploding loss in contrastive distillation')
        # nn.init.constant_(self.proj.bias, 0)
        # trunc_normal_(self.proj.weight, std=.02)

        self.temporal_pooling = temporal_pooling

        self.classifier = nn.Linear(216, num_classes)

    def forward(self, x, B, T, H, W):
        '''
        Arguments
        x : torch.Tensor
            The input tensor of shape (B, TN+1, 768)
        B : int
            The batch size
        T : int
            The number of frames
        H : int
            The height of the input
        W : int
            The width of the input
        '''
        x = x[:, 1:, :]

        x = rearrange(x, 'b (t h w) d -> b t (h w) d', h=H, w=W) # (B, T, HW, 768)
        
        if self.temporal_pooling: # for global distillation
            x = x.mean(dim=2).mean(dim=1) # (B, 768)
        else: # for temporal distillation
            x = x.mean(dim=2) # (B, T, 768)

        x = self.proj(x) # (B, 216) or (B, T, 216)

        if self.temporal_pooling:
            dist_logits = self.classifier(x) # (B, num_classes)
        else:
            dist_logits = self.classifier(x.mean(dim=1)) # (B, num_Classes)

        return x, dist_logits


class FeatureDistillationBlock_classifier_Joints(nn.Module):
    '''
    For the distilled feature classification
    '''
    def __init__(self, norm_layer=nn.LayerNorm, embed_dim=768, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0, 
                 drop=0, act_layer=nn.GELU, mlp_ratio=4, drop_path=0.1, num_joints=13):
        super().__init__()

        self.proj = nn.Linear(embed_dim, 216)
        # print('Debugging feature: Initialize projection weights from truncated normal distribution to attempt to fix exploding loss in contrastive distillation')
        # nn.init.constant_(self.proj.bias, 0)
        # trunc_normal_(self.proj.weight, std=.02)

        self.temporal_pooling = temporal_pooling

        NUM_CLASSES = 60
        self.classifier = nn.Linear(216, NUM_CLASSES)

    def forward(self, x, B, T, H, W):
        '''
        Arguments
        x : torch.Tensor
            The input tensor of shape (B, TN+1, 768)
        B : int
            The batch size
        T : int
            The number of frames
        H : int
            The height of the input
        W : int
            The width of the input
        '''
        x = x[:, 1:, :]

        x = rearrange(x, 'b (t h w) d -> b t (h w) d', h=H, w=W) # (B, T, HW, 768)
        
        if self.temporal_pooling: # for global distillation
            x = x.mean(dim=2).mean(dim=1) # (B, 768)
        else: # for temporal distillation
            x = x.mean(dim=2) # (B, T, 768)

        x = self.proj(x) # (B, 216) or (B, T, 216)

        if self.temporal_pooling:
            dist_logits = self.classifier(x) # (B, 60)
        else:
            dist_logits = self.classifier(x.mean(dim=1)) # (B, 60)

        return x, dist_logits