# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class PiViT_Loss():
    def __init__(self, cls_scale=1, loss_3dsim_scale=1, loss_3dsim_cls_scale=1, loss_2dsim_scale=1.6, clsreduction="mean", num_joints=None, num_frames=None):
        self.cross_entropy = nn.CrossEntropyLoss(reduction=clsreduction)

        self.dist_loss_func_type = 'mse' # standard mse distillation loss
        if self.dist_loss_func_type == 'mse':
            self.lossfunc_3dsim = nn.MSELoss()

        self.lossfunc_2dsim = nn.BCELoss()
            
        self.cls_scale = cls_scale
        self.loss_3dsim_scale = loss_3dsim_scale
        self.loss_3dsim_cls_scale = loss_3dsim_cls_scale
        self.loss_2dsim_scale = loss_2dsim_scale
        self.num_joints = num_joints if num_joints != 13 else 15 # for smarthome (13 joints) hyperformer training, 15 joints were used following Di Yang's UNIK paper
        self.num_frames = num_frames
    
    def __call__(self, cls_logits, true_cls_label, outs_2d3dsim_all_layers, true_skeleton_features, keypoint_mask):
        '''
        Arguments
        cls_logits : torch.Tensor
            The logits for the classification task
        true_cls_label : torch.Tensor
            The true labels for the classification task
        outs_2d3dsim_all_layers : list[dict]
            The list of dictionaries that is the same length as the model depth. Each dictionary contains the distillation features and logits for that layer. 
            If there was no distillation performed in the layer, the dictionary will be empty. Otherwise it will have keys corresponding to the distillation level [global, temporal, joint].
        true_skeleton_features : torch.Tensor or None
            The true features from hyperformer (or other skeleton model) (Bx400x216). Or None if we do not pass distillation features (for baseline timesformer or only 2dsim)
        '''
        #########################################
        ### Classification loss
        #########################################
        cls_loss = self.cross_entropy(cls_logits, true_cls_label)


        #########################################
        ### 2D-SIM Loss
        #########################################
        num_2dsim_layers = 0
        loss_2dsim = 0
        for layer_idx, distillation_out in enumerate(outs_2d3dsim_all_layers):
            if 'learned_mask_2dsim' in distillation_out:
                loss_2dsim = loss_2dsim + self.lossfunc_2dsim(distillation_out['learned_mask_2dsim'], keypoint_mask)
                num_2dsim_layers += 1

        if num_2dsim_layers > 0:
            loss_2dsim = loss_2dsim / num_2dsim_layers


        #########################################
        ### 3D-SIM Loss
        #########################################
        # If true_skeleton_features is a tensor of -1's (i.e., the shape will be (B,)), then we are using then no hyperformer features were passed
        # We only transform the true_skeleton_features if they are passed
        if len(true_skeleton_features.shape) != 1:
            B, _, C = true_skeleton_features.shape # Bx400x216

            temporal_dist_features = rearrange(true_skeleton_features, 'b (t v) c -> b t v c', b=B, v=self.num_joints, c=C).mean(dim=2) # BxTx216
            T = temporal_dist_features.shape[1]

            if T != self.num_frames and self.num_frames == 8:
                temporal_dist_features = temporal_dist_features[:, 0::2, :]
            elif T == self.num_frames and self.num_frames == 16:
                pass
            else:
                raise ValueError('Num frames is not 8 or 16')

            joint_dist_features = -1

        # check outs_2d3dsim_all_layers keys, if any distillation is used (i.e., hyperformer features were passed) and if true_skeleton_features is None, raise an error
        if len(true_skeleton_features.shape) == 1:
            for layer_idx, distillation_out in enumerate(outs_2d3dsim_all_layers):
                if 'global' in distillation_out:
                    raise ValueError('Distillation was used but true_skeleton_features is None')
                if 'temporal' in distillation_out:
                    raise ValueError('Distillation was used but true_skeleton_features is None')
                if 'joint' in distillation_out:
                    raise ValueError('Distillation was used but true_skeleton_features is None')

        dist_loss = 0
        global_dist_loss = 0
        temporal_dist_loss = 0
        joint_dist_loss = 0

        # aggregate the distillation outputs from each layer into corresponding distillation level losses
        num_global_distillation_layers, num_temporal_distillation_layers, num_joint_distillation_layers = 0, 0, 0

        for layer_idx, distillation_out in enumerate(outs_2d3dsim_all_layers):
            if 'global' in distillation_out:
                # In this case, distillation features should be Bx216
                global_dist_loss = global_dist_loss + self.lossfunc_3dsim(distillation_out['global'][0], true_skeleton_features.mean(dim=1))
                num_global_distillation_layers += 1
            if 'temporal' in distillation_out:
                # In this case, distillation features should be BxTx216
                temporal_dist_loss = temporal_dist_loss + self.lossfunc_3dsim(distillation_out['temporal'][0], temporal_dist_features)
                temporal_dist_loss = temporal_dist_loss / T # normalize by the number of frames
                num_temporal_distillation_layers += 1
            if 'joint' in distillation_out:
                raise NotImplementedError()
                # joint_dist_loss = joint_dist_loss + self.dist_loss_func(distillation_out['joint'][0], true_skeleton_features)
                num_joint_distillation_layers += 1

        # normalize by the number of layers that used distillation
        if num_global_distillation_layers > 0:
            global_dist_loss = global_dist_loss / num_global_distillation_layers
        if num_temporal_distillation_layers > 0:
            temporal_dist_loss = temporal_dist_loss / num_temporal_distillation_layers
        if num_joint_distillation_layers > 0:
            joint_dist_loss = joint_dist_loss / num_joint_distillation_layers

        # aggregate the losses from each distillation level
        dist_loss = global_dist_loss + temporal_dist_loss + joint_dist_loss


        #########################################
        ### Distillation loss of the logits (from classifier)
        #########################################
        global_logits, temporal_logits, joint_logits = None, None, None
        for layer_idx, distillation_out in enumerate(outs_2d3dsim_all_layers):
            if 'global' in distillation_out:
                if global_logits is None:
                    global_logits = distillation_out['global'][1]
                else:
                    global_logits = global_logits + distillation_out['global'][1]
            if 'temporal' in distillation_out:
                if temporal_logits is None:
                    temporal_logits = distillation_out['temporal'][1]
                else:
                    temporal_logits = temporal_logits + distillation_out['temporal'][1]
            if 'joint' in distillation_out:
                raise NotImplementedError()
                if joint_logits is None:
                    joint_logits = distillation_out['joint'][1]
                else:
                    joint_logits = joint_logits + distillation_out['joint'][1]

        global_logit_loss, temporal_logit_loss, joint_logit_loss = 0, 0, 0
        if global_logits is not None:
            global_logit_loss = self.cross_entropy(global_logits, true_cls_label)
        if temporal_logits is not None:
            temporal_logit_loss = self.cross_entropy(temporal_logits, true_cls_label)
        if joint_logits is not None:
            joint_logit_loss = self.cross_entropy(joint_logits, true_cls_label)

        # aggregate the losses of the logits from each distillation level
        dist_logit_loss = global_logit_loss + temporal_logit_loss + joint_logit_loss

        # print all losses
        # print(f'DEBUG: cls_loss: {cls_loss}, dist_loss: {dist_loss}, loss_2dsim: {loss_2dsim}, dist_logit_loss: {dist_logit_loss}, global_dist_loss: {global_dist_loss}, temporal_dist_loss: {temporal_dist_loss}, joint_dist_loss: {joint_dist_loss}, global_logit_loss: {global_logit_loss}, temporal_logit_loss: {temporal_logit_loss}, joint_logit_loss: {joint_logit_loss}')

        total_loss = (self.cls_scale*cls_loss) + (self.loss_3dsim_scale*dist_loss) + (self.loss_3dsim_cls_scale*dist_logit_loss) + (self.loss_2dsim_scale*loss_2dsim)

        return total_loss, cls_loss, dist_loss, loss_2dsim, dist_logit_loss, global_dist_loss, temporal_dist_loss, joint_dist_loss, global_logit_loss, temporal_logit_loss, joint_logit_loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "pivit_loss": PiViT_Loss
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
