from timesformer.models.modules import Attention, Mlp
from timesformer.models.vit_utils import DropPath

from einops import rearrange
from torch import nn
import torch

class PAAT(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, embed_dim=768, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0, 
                 drop=0, act_layer=nn.GELU, mlp_ratio=4, drop_path=0.1, num_joints=13):
        super().__init__()
        # multi-label multi-class prediction of joints
        latent_dim = 256

        self.learned_joint_proj1 = nn.Linear(embed_dim, latent_dim)
        self.learned_joint_proj2 = nn.Linear(latent_dim, num_joints)
        self.S = nn.Sigmoid()

        nn.init.constant_(self.learned_joint_proj1.weight, 0)
        nn.init.constant_(self.learned_joint_proj1.bias, 0)
        nn.init.constant_(self.learned_joint_proj2.weight, 0)
        nn.init.constant_(self.learned_joint_proj2.bias, 0)


    '''
    PoseBlock does additional spatial attention over patches containing poses

    Expects the input, x, to be shape (B, TN+1, 768)
    Output will be shape (B, TN+1, 768)
    '''
    def forward(self, x, B, T, H, W):
        # Convert shape of input from (B, TN+1, 768) to (B, TN, 768). i.e., take everything except cls_token
        x = x[:, 1:, :]

        # Process mask
        learned_mask = self.S(self.learned_joint_proj2(self.learned_joint_proj1(x)))

#        learned_mask = learned_mask[:, :, 0] # Only for 2D joint mask

        return x, learned_mask
