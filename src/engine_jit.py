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


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
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


def compute_iou_score(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
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


def compute_sensitivity(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
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


def compute_specificity(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
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


def compute_hausdorff_distance_95(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    """
    Compute 95th percentile Hausdorff Distance (HD95) for binary segmentation.
    pred: (B, C, H, W) - predictions
    target: (B, C, H, W) - ground truth
    """
    pred_binary = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_binary = (target > threshold).cpu().numpy().astype(np.uint8)

    if pred_binary.ndim == 2:
        pred_binary = pred_binary[None, None, ...]
    elif pred_binary.ndim == 3:
        pred_binary = pred_binary[None, ...]

    if target_binary.ndim == 2:
        target_binary = target_binary[None, None, ...]
    elif target_binary.ndim == 3:
        target_binary = target_binary[None, ...]

    hd95_scores = []

    for b in range(pred_binary.shape[0]):
        pred_mask = pred_binary[b, 0]
        target_mask = target_binary[b, 0]

        # Get contours
        pred_dist = distance_transform_edt(1 - pred_mask)
        target_dist = distance_transform_edt(1 - target_mask)

        # Compute Hausdorff distance (95th percentile)
        hd_pred_to_target = np.percentile(
            pred_dist[target_mask > 0], 95) if target_mask.sum() > 0 else 0
        hd_target_to_pred = np.percentile(
            target_dist[pred_mask > 0], 95) if pred_mask.sum() > 0 else 0

        hd95 = max(hd_pred_to_target, hd_target_to_pred)
        hd95_scores.append(hd95)

    return np.mean(hd95_scores)


def compute_boundary_iou(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, boundary_width=1):
    """
    Compute Boundary IoU for binary segmentation.
    pred: (B, C, H, W) - predictions in [-1, 1]
    target: (B, C, H, W) - ground truth in [-1, 1]
    threshold: threshold for binarization
    boundary_width: dilation width (in pixels) for boundary extraction
    """
    pred_binary = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_binary = (target > threshold).cpu().numpy().astype(np.uint8)

    if pred_binary.ndim == 2:
        pred_binary = pred_binary[None, None, ...]
    if target_binary.ndim == 2:
        target_binary = target_binary[None, None, ...]

    kernel_size = max(1, int(boundary_width))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))

    boundary_ious = []
    for b in range(pred_binary.shape[0]):
        pred_mask = pred_binary[b, 0]
        target_mask = target_binary[b, 0]

        pred_eroded = cv2.erode(pred_mask, kernel, iterations=1)
        target_eroded = cv2.erode(target_mask, kernel, iterations=1)

        pred_boundary = pred_mask - pred_eroded
        target_boundary = target_mask - target_eroded

        intersection = np.logical_and(
            pred_boundary > 0, target_boundary > 0).sum()
        union = np.logical_or(pred_boundary > 0, target_boundary > 0).sum()

        boundary_iou = float(intersection) / float(union + 1e-8)
        boundary_ious.append(boundary_iou)

    return float(np.mean(boundary_ious))


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    if type(pred) is list or type(target) is list:
        raise NotImplementedError(
            "List of predictions not supported in this function.")
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


def compute_cldice(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    """
    Compute clDice for binary segmentation (tubular structures).
    pred: (B, C, H, W) - predictions in [-1, 1]
    target: (B, C, H, W) - ground truth in [-1, 1]
    threshold: threshold for binarization
    """
    pred_binary = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_binary = (target > threshold).cpu().numpy().astype(np.uint8)

    if pred_binary.ndim == 2:
        pred_binary = pred_binary[None, None, ...]
    if target_binary.ndim == 2:
        target_binary = target_binary[None, None, ...]

    def _skeletonize(mask):
        skel = np.zeros_like(mask, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = mask.copy()
        while True:
            eroded = cv2.erode(img, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel

    cldice_scores = []
    for b in range(pred_binary.shape[0]):
        pred_mask = pred_binary[b, 0]
        target_mask = target_binary[b, 0]

        pred_skel = _skeletonize(pred_mask)
        target_skel = _skeletonize(target_mask)

        tprec = (pred_skel & target_mask).sum() / (pred_skel.sum() + 1e-8)
        tsens = (target_skel & pred_mask).sum() / (target_skel.sum() + 1e-8)
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
        cldice_scores.append(float(cldice))

    return float(np.mean(cldice_scores))


def compute_aji(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    """
    Compute Aggregated Jaccard Index (AJI) for instance segmentation.
    pred: (B, C, H, W) - predictions
    target: (B, C, H, W) - ground truth
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.ndim == 4:
        scores = []
        for b in range(pred.shape[0]):
            scores.append(compute_aji(pred[b], target[b], threshold=threshold))
        return float(np.mean(scores)) if scores else 0.0
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]

    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "AJI expects 2D label maps or batched (B, 1, H, W) inputs.")

    pred = pred.astype(np.int64, copy=False)
    target = target.astype(np.int64, copy=False)

    gt_list = np.unique(target)
    gt_list = gt_list[gt_list != 0]

    pr_list = np.unique(pred)
    pr_list = pr_list[pr_list != 0]
    pr_used = {int(pid): 0 for pid in pr_list}

    overall_correct_count = 0
    union_pixel_count = 0

    gt_ids = list(gt_list)
    while gt_ids:
        gt_id = gt_ids.pop()
        gt_mask = target == gt_id

        predicted_match = pred[gt_mask]
        if predicted_match.size == 0:
            union_pixel_count += np.count_nonzero(gt_mask)
            continue

        predicted_nuc_index = np.unique(predicted_match)
        predicted_nuc_index = predicted_nuc_index[predicted_nuc_index != 0]

        if predicted_nuc_index.size == 0:
            union_pixel_count += np.count_nonzero(gt_mask)
            continue

        best_match = None
        best_ji = 0.0
        for pred_id in predicted_nuc_index:
            pred_mask = pred == pred_id
            inter = np.count_nonzero(gt_mask & pred_mask)
            uni = np.count_nonzero(gt_mask | pred_mask)
            ji = inter / (uni + 1e-8)
            if ji > best_ji:
                best_match = int(pred_id)
                best_ji = ji

        if best_match is None:
            union_pixel_count += np.count_nonzero(gt_mask)
            continue

        best_pred_mask = pred == best_match
        overall_correct_count += np.count_nonzero(gt_mask & best_pred_mask)
        union_pixel_count += np.count_nonzero(gt_mask | best_pred_mask)

        pr_used[best_match] = pr_used.get(best_match, 0) + 1

    for pred_id, used in pr_used.items():
        if used == 0:
            union_pixel_count += np.count_nonzero(pred == pred_id)

    if union_pixel_count == 0:
        return 0.0

    return float(overall_correct_count / union_pixel_count)


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None, additional_loss_fn=None,):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, (images, masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize to [-1, 1] (already [0, 1] from ToTensor)
        masks = masks.to(device, non_blocking=True).to(torch.float32)
        images = images.to(device, non_blocking=True).to(torch.float32)
        images = images * 2.0 - 1.0

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            diff_loss, mask_pred = model(masks, images)
            loss = diff_loss
            if args.add_loss:
                add_loss = additional_loss_fn(mask_pred, masks)
                loss = diff_loss + args.add_loss_weight * add_loss

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
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                if args.add_loss:
                    log_writer.add_scalar('train_diff_loss', diff_loss.item(), epoch_1000x)
                    log_writer.add_scalar('train_add_loss', add_loss.item(), epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_images('train_images', (images + 1.0) / 2.0, epoch_1000x)
                log_writer.add_images('train_masks_gt', masks, epoch_1000x)
                log_writer.add_images('train_masks_pred', mask_pred, epoch_1000x)


def validation(model_without_ddp, data_loader, device, epoch, log_writer=None, threshold=0.5, dataset=None, add_loss=False, additional_loss_fn=None):
    """
    Evaluate model on validation set using Dice, IoU, Sensitivity, Specificity, and HD95 metrics.

    Args:
        model_without_ddp: Model to evaluate
        data_loader: Validation data loader
        device: Device to use
        epoch: Current epoch
        log_writer: TensorBoard writer
        dataset: Dataset name
        add_loss: Whether to compute additional loss
        additional_loss_fn: Additional loss function to use if add_loss is True
    """
    model_without_ddp.eval()

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []
    hd95_scores = []
    if dataset and 'ISIC' in dataset:
        aji_scores = []
    if dataset and 'OCTA500' in dataset:
        cldice_scores = []
    loss_scores = []
    add_loss_scores = []

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
            metrics = calculate_metrics(
                masks_pred, masks_gt, threshold=threshold)
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            sensitivity_scores.append(metrics['sensitivity'])
            specificity_scores.append(metrics['specificity'])
            hd95_scores.append(metrics['hd95'])
            if dataset and 'ISIC' in dataset:
                aji = compute_aji(masks_pred, masks_gt, threshold=threshold)
                aji_scores.append(aji)
            if dataset and 'OCTA500' in dataset:
                cldice = compute_cldice(
                    masks_pred, masks_gt, threshold=threshold)
                cldice_scores.append(cldice)
            val_loss = torch.nn.MSELoss()(masks_pred, masks_gt).item()
            if add_loss and additional_loss_fn is not None:
                val_add_loss = additional_loss_fn(masks_pred, masks_gt).item()
                add_loss_scores.append(val_add_loss)
            loss_scores.append(val_loss)
            if (data_iter_step + 1) % 20 == 0:
                step_msg = (
                    f"  Eval step {data_iter_step + 1}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
                    f"Sensitivity={metrics['sensitivity']:.4f}, Specificity={metrics['specificity']:.4f}, HD95={metrics['hd95']:.4f}, Loss={val_loss:.4f}"
                )
                if add_loss and additional_loss_fn is not None:
                    step_msg += f", Additional Loss={val_add_loss:.4f}"
                if dataset and 'OCTA500' in dataset:
                    step_msg += f", clDice={cldice:.4f}"
                print(step_msg)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_hd95 = np.mean(hd95_scores)
    avg_loss = np.mean(loss_scores)

    print(f"Epoch {epoch} - Validation Metrics:")
    print(f"  Dice: {avg_dice:.4f}")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  Sensitivity: {avg_sensitivity:.4f}")
    print(f"  Specificity: {avg_specificity:.4f}")
    print(f"  HD95: {avg_hd95:.4f}")
    print(f"  Loss: {avg_loss:.4f}")

    if add_loss:
        avg_add_loss = np.mean(add_loss_scores) if add_loss_scores else 0.0
        print(f"  Additional Loss: {avg_add_loss:.4f}")

    if dataset and 'ISIC' in dataset:
        avg_aji = np.mean(aji_scores)
        print(f"  AJI: {avg_aji:.4f}")
    if dataset and 'OCTA500' in dataset:
        avg_cldice = np.mean(cldice_scores)
        print(f"  clDice: {avg_cldice:.4f}")

    if log_writer is not None:
        log_writer.add_scalar('val_dice', avg_dice, epoch)
        log_writer.add_scalar('val_iou', avg_iou, epoch)
        log_writer.add_scalar('val_sensitivity', avg_sensitivity, epoch)
        log_writer.add_scalar('val_specificity', avg_specificity, epoch)
        log_writer.add_scalar('val_hd95', avg_hd95, epoch)
        log_writer.add_images('val_images', (images + 1.0) / 2.0, epoch)
        log_writer.add_images('val_masks_gt', masks_gt, epoch)
        log_writer.add_images('val_masks_pred', masks_pred, epoch)
        log_writer.add_scalar('val_loss', avg_loss, epoch)
        if add_loss:
            log_writer.add_scalar('val_add_loss', avg_add_loss, epoch)
        if dataset and 'ISIC' in dataset:
            log_writer.add_scalar('val_aji', avg_aji, epoch)
        if dataset and 'OCTA500' in dataset:
            log_writer.add_scalar('val_cldice', avg_cldice, epoch)
    model_without_ddp.train()


def save_metrics_to_csv(output_dir, epoch,
                        dice_scores, iou_scores, sensitivity_scores, specificity_scores,
                        hd95_scores, aji_scores, cldice_scores,
                        args):
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
        aji_scores: List of AJI scores for each sample
        cldice_scores: List of clDice scores for each sample
        args: Arguments containing soft_vote and dataset information
    """
    # Calculate averages
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_hd95 = np.mean(hd95_scores)
    if args.dataset and 'ISIC' in args.dataset and aji_scores is not None:
        avg_aji = np.mean(aji_scores)
    else:
        avg_aji = None
    if args.dataset and 'OCTA500' in args.dataset and cldice_scores is not None:
        avg_cldice = np.mean(cldice_scores)
    else:
        avg_cldice = None

    # Save individual results to CSV
    results_csv_path = os.path.join(
        output_dir, f"results-{epoch}{'-soft_vote' if args.soft_vote else ''}.csv")
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if args.dataset and 'ISIC' in args.dataset and aji_scores is not None:
            writer.writerow(
                ['Sample', 'Dice', 'IoU', 'Sensitivity', 'Specificity', 'HD95', 'AJI'])
        elif args.dataset and 'OCTA500' in args.dataset and cldice_scores is not None:
            writer.writerow(
                ['Sample', 'Dice', 'IoU', 'Sensitivity', 'Specificity', 'HD95', 'clDice'])
        else:
            writer.writerow(
                ['Sample', 'Dice', 'IoU', 'Sensitivity', 'Specificity', 'HD95'])
        for i in range(len(dice_scores)):
            row = [
                f"sample_{i:03d}",
                f"{dice_scores[i]:.4f}",
                f"{iou_scores[i]:.4f}",
                f"{sensitivity_scores[i]:.4f}",
                f"{specificity_scores[i]:.4f}",
                f"{hd95_scores[i]:.4f}"
            ]
            if args.dataset and 'ISIC' in args.dataset and aji_scores is not None:
                row.append(f"{aji_scores[i]:.4f}")
            if args.dataset and 'OCTA500' in args.dataset and cldice_scores is not None:
                row.append(f"{cldice_scores[i]:.4f}")
            writer.writerow(row)
    print(f"Individual results saved to {results_csv_path}")

    # Save average results to CSV
    avg_csv_path = os.path.join(
        output_dir, f"average_results-{epoch}{'-soft_vote' if args.soft_vote else ''}.csv")
    with open(avg_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Dice', f"{avg_dice:.4f}"])
        writer.writerow(['IoU', f"{avg_iou:.4f}"])
        writer.writerow(['Sensitivity', f"{avg_sensitivity:.4f}"])
        writer.writerow(['Specificity', f"{avg_specificity:.4f}"])
        writer.writerow(['HD95', f"{avg_hd95:.4f}"])
        if args.dataset and 'ISIC' in args.dataset and avg_aji is not None:
            writer.writerow(['AJI', f"{avg_aji:.4f}"])
        if args.dataset and 'OCTA500' in args.dataset and avg_cldice is not None:
            writer.writerow(['clDice', f"{avg_cldice:.4f}"])
    print(f"Average results saved to {avg_csv_path}")
