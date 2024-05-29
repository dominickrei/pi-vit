from torch import nn
from einops import rearrange

class Module_2DSIM(nn.Module):
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

class Module_3DSIM(nn.Module):
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