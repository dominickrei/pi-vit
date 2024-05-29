# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import numpy as np

import timesformer.utils.logging as logging

from . import decoder_ntu as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

from . import pose_utils
import h5py

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ntu(torch.utils.data.Dataset):
    """
    NTU video loader. Construct the NTU video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the NTU video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 path_to_pose_1 label_1
        path_to_video_2 path_to_pose_2 label_2
        ...
        path_to_video_N path_to_pose_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for NTU".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        # In the case of uniform sampling, we uniformly sample frames from the beginning
        # to the end of the video. In this case there is no reason to have multiple ensemble
        # views, so we set them to 1 if doing uniform sampling
        if self.cfg.DATA.UNIFORM_SAMPLING:
            logger.warning("Uniformly sampling frames for training and testing. This will nulify TEST.NUM_ENSEMBLE_VIEWS and DATA.SAMPLING_RATE")
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS = 1

        # If not uniform sampling:
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing NTU {}... Data path: {}".format(mode, cfg.DATA.PATH_TO_DATA_DIR))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._path_to_poses = []
        self._labels = []
        self._spatial_temporal_idx = []
        
        if self.cfg.EXPERIMENTAL.HYPERFORMER_FEATURES_PATH != '':
            logger.info('[For Distillation] Using hyperformer features from path: {}'.format(self.cfg.EXPERIMENTAL.HYPERFORMER_FEATURES_PATH))
            self._hyperformer_feature_path = self.cfg.EXPERIMENTAL.HYPERFORMER_FEATURES_PATH
        else:
            self._hyperformer_feature_path = ''
            logger.info('[Distillation Disabled] no hyperformer features path was provided in config.')

        # Load and store all video paths, pose paths, and labels
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 3
                )
                path, pose_path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._path_to_poses.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, pose_path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load NTU split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing NTU dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

        if self.cfg.EXPERIMENTAL.DEBUG.RANDOM_MASK:
            logger.warning('Using random keypoint mask for debugging (EXPERIMENTAL.DEBUG.RANDOM_MASK is True). Make sure this is what you want.')

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            keypoint_attention_mask (tensor): the indices of the PatchEmbedded video containing keypoints.
                The dimension is `num_frames * num_patches`
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            if self.cfg.DATA.UNIFORM_SAMPLING:
                # do uniform sampling for train and test
                temporal_sample_index = -2

            frames, sampled_frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            # Load keypoints before data augmentation
            keypoints, njts = pose_utils.npy_to_keypoints(self._path_to_poses[index])

            keypoints = keypoints[sampled_frames]

            '''
            Data augmentation            
            '''
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            
            # Perform data augmentation
            frames, keypoints = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                keypoints=keypoints
            )

            # Add noise to the keypoints
            # if self.cfg.EXPERIMENTAL.DEBUG.NOISE_LEVEL > 0:
            #     keypoints = keypoints + (np.random.rand(*keypoints.shape) * self.cfg.EXPERIMENTAL.DEBUG.NOISE_LEVEL)

            # Generate the attention mask from the keypoints

            granularity = '3d' # '2d' or '3d'. 2d is the presence of a joint in a patch and 3d is the presence of a specific joint in a patch. '2d' is 'Flat Variant' ablation in main paper

            # grab 2D joints
            if granularity == '2d':
                keypoint_attention_mask = pose_utils.keypoints_2_patch_idx(
                    keypoints=keypoints, 
                    patch_size=16, 
                    frame_height=self.cfg.DATA.TRAIN_CROP_SIZE, 
                    frame_width=self.cfg.DATA.TRAIN_CROP_SIZE,
                    inflation=self.cfg.DUAL_BRANCH_TIMESFORMER.POSE_INFLATION
                )
            # grab 3D joints
            elif granularity == '3d':
                keypoint_attention_mask = pose_utils.keypoints_2_patch_joint_labels(
                    keypoints=keypoints,
                    patch_size=16,
                    frame_height=self.cfg.DATA.TRAIN_CROP_SIZE,
                    frame_width=self.cfg.DATA.TRAIN_CROP_SIZE,
                    njts=njts,
                )

            # Random joints if desired
            if self.cfg.EXPERIMENTAL.DEBUG.RANDOM_MASK:
                keypoint_attention_mask = np.random.randint(0, 2, keypoint_attention_mask.shape)
                
            if self.cfg.EXPERIMENTAL.DEBUG.MASK_FILL is None:
                keypoint_attention_mask = torch.tensor(keypoint_attention_mask)
            elif self.cfg.EXPERIMENTAL.DEBUG.MASK_FILL == 0:
                keypoint_attention_mask = torch.zeros(keypoint_attention_mask.shape)
            elif self.cfg.EXPERIMENTAL.DEBUG.MASK_FILL == 1:
                keypoint_attention_mask = torch.ones(keypoint_attention_mask.shape)
            else:
                raise ValueError(f'{self.cfg.EXPERIMENTAL.DEBUG.MASK_FILL} is not a valid option for EXPERIMENTEL.DEBUG.MASK_FILL!')


            if not self.cfg.MODEL.ARCH in ['vit']:
                raise NotImplementedError('Only implemented for vit model architecture')
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                # Perform temporal sampling from the fast pathway.
                frame_idxs_to_sample = torch.linspace(0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES).long()
               
                frames = torch.index_select(
                     frames,
                     1,
                     frame_idxs_to_sample,
                )

            # Get logits from hyperformer
            if self.mode in ['train', 'val', 'test']:
                video_identifier = os.path.split(self._path_to_videos[index])[-1][:-8]
                # hyperformer_logits = self._hyperformer_logits[video_identifier]
                # return frames, keypoint_attention_mask, label, hyperformer_logits, index, {}
                # hyperformer_features = self._hyperformer_features[video_identifier]
                if self._hyperformer_feature_path is not '':
                    hyperformer_features = h5py.File(os.path.join(self._hyperformer_feature_path, video_identifier + '.h5'), 'r')['data'][:]
                else:
                    hyperformer_features = -1

                return frames, keypoint_attention_mask, label, hyperformer_features, index, {}

            return frames, keypoint_attention_mask, label, index, {}

        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
