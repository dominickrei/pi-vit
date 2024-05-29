# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter

import pandas as pd
import wandb

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    # Other evaluation protocols for testing. We can combine the classification logits with the distilled logits in different ways
    # In the paper, all results are reported with cls_only
    _EVAL_PROTOCOLS = {
        'cls_only': lambda c, d: c,
        'only_3dsim': lambda c, d: d,
        'cls_plus_3dsim': lambda c, d: c + d,
        'cls_plus_3dsim_sm': lambda c, d: torch.nn.functional.softmax(c, dim=1) - torch.nn.functional.softmax(d, dim=1),
    }

    for cur_iter, (inputs, keypoint_masks, labels, skeleton_logits, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.

            keypoint_masks = keypoint_masks.cuda()
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()


        # Get the predictions.
        cls_logits, outs_2d3dsim_all_layers = model(inputs, keypoint_masks)

        # get the distilled logits from all layers and all distillation levels. sum the logits at the distillation level
        global_logits, temporal_logits = None, None
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

        # add the summed logits from all distillation levels to get the final distilled logits
        all_distilled_logits = [logits for logits in [global_logits, temporal_logits] if logits is not None]

        if len(all_distilled_logits) > 0:
            distilled_logits = all_distilled_logits[0]
            for distilled_logit in all_distilled_logits[1:]:
                distilled_logits = distilled_logits + distilled_logit

            preds = _EVAL_PROTOCOLS[cfg.TEST.EVAL_PROTOCOL](cls_logits, distilled_logits)
        else:
            preds = cls_logits

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics(eval_protocol=cfg.TEST.EVAL_PROTOCOL)
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Do wandb setup if requested. Only run on main process
    if cfg.WANDB.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        cfg_dict = misc.cfg_2_dict(cfg)
        wandb_run = wandb.init(
            id = cfg.WANDB.RUN_ID,
            resume = 'allow',
            project = cfg.WANDB.PROJECT_NAME,
            name = cfg.WANDB.EXPERIMENT_NAME if cfg.WANDB.EXPERIMENT_NAME != '' else None,
            config = cfg_dict,
        )
    else:
        wandb_run = None

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        cfg,
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    # write to weights and biases if available.
    if wandb_run is not None:
        if cfg.TEST.DATASET == 'smarthome':
            if 'cv1' in cfg.DATA.PATH_TO_DATA_DIR or 'cv2' in cfg.DATA.PATH_TO_DATA_DIR:
                class_mapping_filename = 'cv_label_mappings.csv'
            elif 'cs' in cfg.DATA.PATH_TO_DATA_DIR:
                class_mapping_filename = 'cs_label_mappings.csv'
            else:
                raise NotImplementedError('Class mappings are not found for given dataset. Disable wandb logging')

            if cfg.DATA.PATH_TO_DATA_DIR[-1] == '/':
                head, _ = os.path.split(cfg.DATA.PATH_TO_DATA_DIR[:-1])
            else:
                head, _ = os.path.split(cfg.DATA.PATH_TO_DATA_DIR)

            class_mapping_path = f'{head}/{class_mapping_filename}'
            class_mapping = list(pd.read_csv(class_mapping_path, header=None)[0].values)

        elif cfg.TEST.DATASET == 'ntu':
            class_mapping_path = '/data/ntu/csvs/_LABEL_MAPPINGS/ntu60_label_mapping.csv'
            class_mapping = list(pd.read_csv(class_mapping_path, header=None)[0].values)
                

        # wandb should keep track of current run
        wandb_run.log(
            {
                "test_conf_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=np.argmax(test_meter.video_preds.cpu().numpy(), axis=1), 
                    y_true=test_meter.video_labels.cpu().numpy(), 
                    class_names=class_mapping),
                "Test": 
                {
                    "mCA": float(test_meter.stats['mCA']),
                    "Top1_acc": float(test_meter.stats['top1_acc']),
                    "Top5_acc": float(test_meter.stats['top5_acc']),
                }
            }
        )

        wandb.save(os.path.join(cfg.OUTPUT_DIR, 'stdout.log'))

        if cfg.WANDB.SAVE_LAST_CHECKPOINT and cu.has_checkpoint(cfg.OUTPUT_DIR):
            last_checkpoint_path = cu.get_last_checkpoint(cfg.OUTPUT_DIR)

            if not cfg.WANDB.SAVE_OPTIMIZER_STATE:
                state_dict = torch.load(last_checkpoint_path)
                del state_dict['optimizer_state']
                torch.save(state_dict, last_checkpoint_path)


            wandb.save(last_checkpoint_path)

    if writer is not None:
        writer.close()
