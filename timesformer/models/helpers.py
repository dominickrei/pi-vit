# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified model creation / weight loading / state_dict helpers

from ast import And
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timesformer.models.features import FeatureListNet, FeatureDictNet, FeatureHookNet
from timesformer.models.conv2d_same import Conv2dSame
from timesformer.models.linear import Linear


_logger = logging.getLogger(__name__)

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_pretrained(model, cfg=None, num_classes=1000, db_depth=None, db_human_depth=None, db_bg_depth=None, in_chans=3, filter_fn=None, 
                    img_size=224, num_frames=8, num_patches=196, attention_type='divided_space_time', pretrained_model="", strict=True,
                    merge_input='cls_token', pose_block_model=False):
    
    # boolean indicating whether model is dual-branch. if not it is base-timesformer
    is_dual_branch = ((db_depth is not None) and ((db_human_depth is not None) or (db_bg_depth is not None))) or pose_block_model

    if cfg is None:
        cfg = getattr(model, 'default_cfg') # contains url to download pretrained ViT weights
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    # Load ViT model weights if no model specified
    if len(pretrained_model) == 0:
        pretrain_with_vit = True
        state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu') # Load ViT weights if no video model passed

    # Load given pretrained video model weights
    else:
        pretrain_with_vit = False
        
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)


    if filter_fn is not None:
        # optional modification of state dict prior to loading
        state_dict = filter_fn(state_dict)

    # Can ignore this code block for now, we will only be using 3-color channel inputs
    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    # Load classifier head weights
    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        #print('Removing the last fully connected layer due to dimensions mismatch ('+str(num_classes)+ ' != '+str(state_dict[classifier_name + '.weight'].size(0))+').', flush=True)
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False


    ## Resizing the positional embeddings in case they don't match
    num_cls_tokens = 2 if 'uniform_hbg' in str(type(model)) else 1
    if num_patches + num_cls_tokens != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)

        if num_cls_tokens == 1:
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        elif num_cls_tokens == 2:
            new_pos_embed = torch.cat((cls_pos_embed, cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
    
    # For less parameter model, we use two cls tokens
    if num_cls_tokens == 2:
        state_dict['cls_token_human'] = state_dict['cls_token']
        state_dict['cls_token_bg'] = state_dict['cls_token']

    ## Resizing time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)

    ## Initializing spatial attention if using dual-branch-timesformer
    if db_depth is None: # base timesformer. do nothing (spatial weights already match)
        pass
    elif is_dual_branch: # only call if dual-branch-timesformer
        state_dict = init_spatial_attention(state_dict=state_dict, db_depth=db_depth, db_human_depth=db_human_depth, db_bg_depth=db_bg_depth, merge_input=merge_input, pretrain_with_vit=pretrain_with_vit)
    elif num_cls_tokens == 2:
        state_dict = init_spatial_attention_lessparammodel(state_dict=state_dict, db_depth=db_depth)
    else: # undefined architecture. cfg options are likely invalid
        raise ValueError('dual-branch-timesformer inputs are incorrect')

    ## Initializing temporal attention
    # Temporal attention and spatial attention are initialized with the same weights
    if attention_type == 'divided_space_time':
        state_dict = init_temporal_attention(state_dict=state_dict)

    ## Loading the weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    _logger.info(f'Loaded pretrained weights. Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}')

# If this is called, we are loading weights for a dual-branch-timesformer model with efficient attention computation
def init_spatial_attention_lessparammodel(state_dict, db_depth):
    new_state_dict = state_dict.copy()

    # Assign attn block weights
    for key in state_dict:
        if 'attn.proj' in key:
            new_state_dict[key.replace('proj', 'proj1')] = state_dict[key]
            new_state_dict[key.replace('proj', 'proj2')] = state_dict[key]

        if '.mlp.' in key:
            new_state_dict[key.replace('mlp', 'mlp_human')] = state_dict[key]
            new_state_dict[key.replace('mlp', 'mlp_bg')] = state_dict[key]

        if '.norm2.' in key:
            new_state_dict[key.replace('norm2', 'norm2_human')] = state_dict[key]
            new_state_dict[key.replace('norm2', 'norm2_bg')] = state_dict[key]

    # Assign merge block weights
    suffixes = ['proj.bias', 'proj.weight', 'qkv.bias', 'qkv.weight']

    for i in range(db_depth):
        ini_prefix = f'blocks.{i}.attn'
        new_key = f'blocks.{i}.merge'

        for suffix in suffixes:
            if 'proj' in suffix:
                new_suffix = suffix.replace('proj', 'proj1')
            else:
                new_suffix = suffix

            new_state_dict[f'{new_key}.{new_suffix}'] = state_dict[f'{ini_prefix}.{suffix}']

    return new_state_dict

# If this is called, we are loading weights for a dual-branch-timesformer model
def init_spatial_attention(state_dict, db_depth, db_human_depth, db_bg_depth, merge_input, pretrain_with_vit):
    new_state_dict = state_dict.copy()

    available_blocks = 12 # number of ViT weight blocks
    levels_per_block = max(db_human_depth, db_bg_depth) + 1 # how many levels in 1 dual-branch-timesformer block
    offset = 1 if merge_input == 'cls_token' else 0 # Used during merge block weight assignment

    prefix = 'blocks'

    if pretrain_with_vit:
        # ViT weights
        suffixes = [
            'norm1.bias', 'norm1.weight', 'norm2.bias', 'norm2.weight', 
            'mlp.fc1.bias', 'mlp.fc1.weight', 'mlp.fc2.bias', 'mlp.fc2.weight', 
            'attn.proj.bias', 'attn.proj.weight', 'attn.qkv.bias', 'attn.qkv.weight'
        ]
    else:
        # Base TimeSformer (trained on Kinetics) weights
        suffixes = [
            'norm1.bias', 'norm1.weight', 'norm2.bias', 'norm2.weight', 
            'mlp.fc1.bias', 'mlp.fc1.weight', 'mlp.fc2.bias', 'mlp.fc2.weight', 
            'attn.proj.bias', 'attn.proj.weight', 'attn.qkv.bias', 'attn.qkv.weight',
            'temporal_norm1.weight', 'temporal_norm1.bias', 'temporal_attn.qkv.weight', 'temporal_attn.qkv.bias',
            'temporal_attn.proj.weight', 'temporal_attn.proj.bias', 'temporal_fc.weight', 'temporal_fc.bias'
        ]

    # If merging with cls_tokens, our merge_block is instance of Attention() (rather than AttnBlock())
    # i.e., it wont have as many parameters to set
    suffixes_cls = ['attn.proj.bias', 'attn.proj.weight', 'attn.qkv.bias', 'attn.qkv.weight']

    # Assign weights to norm blocks in the case of cls_token merge (we want to norm human_x and bg_x individually)
    # In the case of 'all' merge, the norm block is named model.norm and is already contained in weights
    if merge_input == 'cls_token':
        new_state_dict['human_norm.weight'] = state_dict['norm.weight']
        new_state_dict['human_norm.bias'] = state_dict['norm.bias']
        new_state_dict['bg_norm.weight'] = state_dict['norm.weight']
        new_state_dict['bg_norm.bias'] = state_dict['norm.bias']

        del new_state_dict['norm.weight']
        del new_state_dict['norm.bias']
    elif merge_input == 'all':
        new_state_dict['merge_norm.weight'] = state_dict['norm.weight']
        new_state_dict['merge_norm.bias'] = state_dict['norm.bias']

    # Assign weights to dual-branch-blocks
    for d in range(db_depth):
        human_bg_height = levels_per_block - 1
        
        # Assign weights to others
        for i in range(human_bg_height):
            level = d * (levels_per_block - offset) + i

            if level < available_blocks: # if we have enough ViT weight blocks
                # Allocate weights to human blocks
                if i < db_human_depth:
                    for suffix in suffixes:
                        vit_key = f'{prefix}.{level}.{suffix}'
                        new_key = f'dual_branch_blocks.{d}.human_blocks.{i}.{suffix}'

                        new_state_dict[new_key] = state_dict[vit_key]

                # Allocate weights to background blocks
                if i < db_bg_depth:
                    for suffix in suffixes:
                        vit_key = f'{prefix}.{level}.{suffix}'
                        new_key = f'dual_branch_blocks.{d}.bg_blocks.{i}.{suffix}'

                        new_state_dict[new_key] = state_dict[vit_key]

        # Allocate weights to merge block
        # If merge with all tokens, we should up the level. Otherwise keep it the same as last highest human/bg block
        level = level + 1 - offset
        merge_suffixes = suffixes if merge_input == 'all' else suffixes_cls

        if level < available_blocks:
            for suffix in merge_suffixes:
                vit_key = f'{prefix}.{level}.{suffix}'

                suffix = suffix if merge_input == 'all' else suffix[5:] # clip 'attn.'
                new_key = f'dual_branch_blocks.{d}.merge_block.{suffix}'

                new_state_dict[new_key] = state_dict[vit_key]

    # Remove all used and unused ViT weight blocks from state_dict to avoid future errors
    for i in range(available_blocks):
        for suffix in suffixes:
            del new_state_dict[f'{prefix}.{i}.{suffix}']

    return new_state_dict
 
def init_temporal_attention(state_dict):
    new_state_dict = state_dict.copy()

    for key in state_dict:
        if 'blocks' in key and '.attn.' in key:
            new_key = key.replace('attn','temporal_attn')
            if not new_key in state_dict:
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[new_key] = state_dict[new_key]
        if 'blocks' in key and '.norm1.' in key:
            new_key = key.replace('norm1','temporal_norm1')
            if not new_key in state_dict:
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[new_key] = state_dict[new_key]

    return new_state_dict

def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)

    if pruned:
        model = adapt_model_from_file(model, variant)

    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn, strict=pretrained_strict)

    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg

    return model
