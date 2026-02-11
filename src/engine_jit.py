import copy
import csv
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

import util.lr_sched as lr_sched
import util.misc as misc


def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice coefficient for binary segmentation.
    pred: (B, C, H, W) - predictions in [-1, 1]
    target: (B, C, H, W) - ground truth in [-1, 1]
    threshold: threshold for binarization
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()

    dice = 2.0 * intersection / (union + 1e-8)
    return dice.item()


def compute_iou_score(pred, target, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    pred: (B, C, H, W) - predictions in [-1, 1]
    target: (B, C, H, W) - ground truth in [-1, 1]
    threshold: threshold for binarization
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = (pred_binary + target_binary - pred_binary * target_binary).sum()

    iou = intersection / (union + 1e-8)
    return iou.item()


def compute_sensitivity(pred, target, threshold=0.5):
    """
    Compute Sensitivity (Recall) for binary segmentation.
    Sensitivity = TP / (TP + FN)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    return sensitivity.item()


def compute_specificity(pred, target, threshold=0.5):
    """
    Compute Specificity for binary segmentation.
    Specificity = TN / (TN + FP)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    specificity = tn / (tn + fp + 1e-8)
    return specificity.item()


def compute_hausdorff_distance_95(pred, target, threshold=0.5):
    """
    Compute 95th percentile Hausdorff Distance (HD95) for binary segmentation.
    pred: (B, C, H, W) - predictions
    target: (B, C, H, W) - ground truth
    """
    pred_binary = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_binary = (target > threshold).cpu().numpy().astype(np.uint8)

    hd95_scores = []

    for b in range(pred_binary.shape[0]):
        pred_mask = pred_binary[b, 0]
        target_mask = target_binary[b, 0]

        # Get contours
        pred_dist = distance_transform_edt(1 - pred_mask)
        target_dist = distance_transform_edt(1 - target_mask)

        # Compute Hausdorff distance (95th percentile)
        hd_pred_to_target = np.percentile(pred_dist[target_mask > 0], 95) if target_mask.sum() > 0 else 0
        hd_target_to_pred = np.percentile(target_dist[pred_mask > 0], 95) if pred_mask.sum() > 0 else 0

        hd95 = max(hd_pred_to_target, hd_target_to_pred)
        hd95_scores.append(hd95)

    return np.mean(hd95_scores)


def calculate_metrics(pred, target, threshold=0.5):
    if type(pred) is list or type(target) is list:
        raise NotImplementedError("List of predictions not supported in this function.")
    elif type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    dice = compute_dice_score(pred, target, threshold=threshold)
    iou = compute_iou_score(pred, target, threshold=threshold)
    sensitivity = compute_sensitivity(pred, target, threshold=threshold)
    specificity = compute_specificity(pred, target, threshold=threshold)
    hd95 = compute_hausdorff_distance_95(pred, target, threshold=threshold)

    return {'dice': dice, 'iou': iou, 'sensitivity': sensitivity, 'specificity': specificity, 'hd95': hd95}


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, (images, masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize to [-1, 1] (already [0, 1] from ToTensor)
        masks = masks.to(device, non_blocking=True).to(torch.float32)
        images = images.to(device, non_blocking=True).to(torch.float32)
        images = images * 2.0 - 1.0

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, mask_pred = model(masks, images)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_images('train_images', (images + 1.0) / 2.0, epoch_1000x)
                log_writer.add_images('train_masks_gt', masks, epoch_1000x)
                log_writer.add_images('train_masks_pred', mask_pred, epoch_1000x)


def evaluate(model_without_ddp, data_loader, device, epoch, log_writer=None, threshold=0.5):
    """
    Evaluate model on validation set using Dice, IoU, Sensitivity, Specificity, and HD95 metrics.

    Args:
        model_without_ddp: Model to evaluate
        data_loader: Validation data loader
        device: Device to use
        epoch: Current epoch
        log_writer: TensorBoard writer
    """
    model_without_ddp.eval()

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []
    hd95_scores = []

    with torch.no_grad():
        for data_iter_step, (images, masks) in enumerate(data_loader):
            # Normalize inputs to [-1, 1] (already [0, 1] from ToTensor)
            masks_gt = masks.to(device, non_blocking=True).to(torch.float32)
            images = images.to(device, non_blocking=True).to(torch.float32)
            images = images * 2.0 - 1.0

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Predict masks using the diffusion model (forward pass at inference time)
                # For now, sample masks conditioned on images
                masks_pred, _ = model_without_ddp.generate(images)

            # Compute metrics
            metrics = calculate_metrics(masks_pred, masks_gt, threshold=threshold)
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            sensitivity_scores.append(metrics['sensitivity'])
            specificity_scores.append(metrics['specificity'])
            hd95_scores.append(metrics['hd95'])
            if (data_iter_step + 1) % 20 == 0:
                print(f"  Eval step {data_iter_step + 1}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                      f"Sensitivity={metrics['sensitivity']:.4f}, Specificity={metrics['specificity']:.4f}, HD95={metrics['hd95']:.4f}")

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_hd95 = np.mean(hd95_scores)

    print(f"Epoch {epoch} - Validation Metrics:")
    print(f"  Dice: {avg_dice:.4f}")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  Sensitivity: {avg_sensitivity:.4f}")
    print(f"  Specificity: {avg_specificity:.4f}")
    print(f"  HD95: {avg_hd95:.4f}")

    if log_writer is not None:
        log_writer.add_scalar('val_dice', avg_dice, epoch)
        log_writer.add_scalar('val_iou', avg_iou, epoch)
        log_writer.add_scalar('val_sensitivity', avg_sensitivity, epoch)
        log_writer.add_scalar('val_specificity', avg_specificity, epoch)
        log_writer.add_scalar('val_hd95', avg_hd95, epoch)
        log_writer.add_images('val_images', (images + 1.0) / 2.0, epoch)
        log_writer.add_images('val_masks_gt', masks_gt, epoch)
        log_writer.add_images('val_masks_pred', masks_pred, epoch)

    model_without_ddp.train()


def save_metrics_to_csv(output_dir, epoch, dice_scores, iou_scores, sensitivity_scores, specificity_scores, hd95_scores, soft_vote):
    """
    Save individual and average metrics to CSV files.

    Args:
        output_dir: Directory to save CSV files
        epoch: Current epoch or checkpoint name
        dice_scores: List of Dice scores for each sample
        iou_scores: List of IoU scores for each sample
        sensitivity_scores: List of Sensitivity scores for each sample
        specificity_scores: List of Specificity scores for each sample
        hd95_scores: List of HD95 scores for each sample
    """
    # Calculate averages
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_hd95 = np.mean(hd95_scores)

    # Save individual results to CSV
    results_csv_path = os.path.join(output_dir, f"results-{epoch}{'-soft_vote' if soft_vote else ''}.csv")
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample', 'Dice', 'IoU', 'Sensitivity', 'Specificity', 'HD95'])
        for i in range(len(dice_scores)):
            writer.writerow([
                f"sample_{i:03d}",
                f"{dice_scores[i]:.4f}",
                f"{iou_scores[i]:.4f}",
                f"{sensitivity_scores[i]:.4f}",
                f"{specificity_scores[i]:.4f}",
                f"{hd95_scores[i]:.4f}"
            ])
    print(f"Individual results saved to {results_csv_path}")

    # Save average results to CSV
    avg_csv_path = os.path.join(output_dir, f"average_results-{epoch}{'-soft_vote' if soft_vote else ''}.csv")
    with open(avg_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Dice', f"{avg_dice:.4f}"])
        writer.writerow(['IoU', f"{avg_iou:.4f}"])
        writer.writerow(['Sensitivity', f"{avg_sensitivity:.4f}"])
        writer.writerow(['Specificity', f"{avg_specificity:.4f}"])
        writer.writerow(['HD95', f"{avg_hd95:.4f}"])
    print(f"Average results saved to {avg_csv_path}")
