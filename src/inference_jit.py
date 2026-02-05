import argparse
import contextlib
import os
from pathlib import Path

import autorootcwd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoiser import Denoiser
from engine_jit import compute_dice_score, compute_hausdorff_distance_95, compute_iou_score, compute_sensitivity, compute_specificity, save_metrics_to_csv
from util.octadataset import OCTASegmentationDataset, get_octa_transform


def get_args_parser():
    parser = argparse.ArgumentParser("JiT inference", add_help=False)

    # architecture
    parser.add_argument("--model", default="JiT-B/16", type=str, metavar="MODEL")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--mask_channel", type=int, default=1)

    # sampling
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--interval_min", default=0.0, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)

    # noise params
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)

    # ema params (required by Denoiser)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)

    # io
    parser.add_argument("--dataset", default="OCTA500_6M", type=str, help="Dataset name")
    parser.add_argument("--data_path", default="data", type=str, help="Path to images or dataset split")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (.pth)")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)

    # patch sampling
    parser.add_argument("--samp_patch_size", default=256, type=int)
    parser.add_argument("--stride", default=128, type=int)

    # inference options
    parser.add_argument("--ema", default="ema1", choices=["none", "ema1", "ema2"], type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--save_prob", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="Compute metrics if labels exist")

    return parser


def _extract_patches(image: torch.Tensor, samp_patch_size: int, stride: int):
    """Extract overlapping patches from image."""
    c, h, w = image.shape
    patches = []
    positions = []

    for top in range(0, h - samp_patch_size + 1, stride):
        for left in range(0, w - samp_patch_size + 1, stride):
            patch = image[:, top:top + samp_patch_size, left:left + samp_patch_size]
            patches.append(patch)
            positions.append((top, left))

    # Handle right and bottom edges if needed
    if h > samp_patch_size and (h - samp_patch_size) % stride != 0:
        top = h - samp_patch_size
        for left in range(0, w - samp_patch_size + 1, stride):
            patch = image[:, top:top + samp_patch_size, left:left + samp_patch_size]
            patches.append(patch)
            positions.append((top, left))

    if w > samp_patch_size and (w - samp_patch_size) % stride != 0:
        left = w - samp_patch_size
        for top in range(0, h - samp_patch_size + 1, stride):
            patch = image[:, top:top + samp_patch_size, left:left + samp_patch_size]
            patches.append(patch)
            positions.append((top, left))

    # Handle corner
    if h > samp_patch_size and w > samp_patch_size:
        if (h - samp_patch_size) % stride != 0 and (w - samp_patch_size) % stride != 0:
            top = h - samp_patch_size
            left = w - samp_patch_size
            patch = image[:, top:top + samp_patch_size, left:left + samp_patch_size]
            patches.append(patch)
            positions.append((top, left))

    return torch.stack(patches), positions


def _reconstruct_from_patches(patches: torch.Tensor, positions: list, orig_shape: tuple, samp_patch_size: int):
    """Reconstruct full image from overlapping patches using weighted averaging."""
    c, h, w = orig_shape
    output = torch.zeros((c, h, w), device=patches.device, dtype=patches.dtype)
    weight = torch.zeros((h, w), device=patches.device, dtype=patches.dtype)

    for patch, (top, left) in zip(patches, positions):
        output[:, top:top + samp_patch_size, left:left + samp_patch_size] += patch
        weight[top:top + samp_patch_size, left:left + samp_patch_size] += 1

    output = output / weight.clamp(min=1.0)
    return output


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, ema: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ema != "none" and f"model_{ema}" in checkpoint:
        ema_state = checkpoint[f"model_{ema}"]
        model_state = model.state_dict()
        for name in model_state:
            if name in ema_state:
                model_state[name] = ema_state[name]
        model.load_state_dict(model_state, strict=False)
        return

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)


def _save_mask(output_dir: str, rel_path: str, pred: torch.Tensor, threshold: float, save_prob: bool, postfix: str = "_mask", epoch: int = None, is_binary: bool = True):
    image_dir = Path(output_dir, f"images-{epoch}" if epoch is not None else "images")
    # pred = ((pred + 1.0) / 2.0).to(torch.float32)  # Rescale to [0, 1]

    # Remove all singleton dimensions and get 2D array
    while pred.ndim > 2:
        pred = pred.squeeze(0)

    rel_stem = os.path.splitext(rel_path)[0]
    out_path = os.path.join(image_dir, f"{rel_stem}{postfix}.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if save_prob:
        prob = pred.cpu().numpy()
        np.save(out_path.replace(f"{postfix}.npy", "_prob.npy"), prob)

    if not is_binary:
        data = pred.cpu().numpy()
        np.save(out_path, data)
    else:
        binary = (pred > threshold).cpu().numpy()
        np.save(out_path, binary)


def _save_intermediate_masks(output_dir: str, rel_path: str, intermediates: list, epoch: int = None):
    image_dir = Path(output_dir, f"images-{epoch}" if epoch is not None else "images", "intermediates")

    rel_stem = os.path.splitext(rel_path)[0]
    os.makedirs(image_dir, exist_ok=True)
    for idx, intermediate in enumerate(intermediates):
        while intermediate.ndim > 2:
            intermediate = intermediate.squeeze(0)
        intermediate = torch.clamp((intermediate + 1.0) / 2.0, 0.0, 1.0)
        data = intermediate.cpu().numpy()
        out_path = os.path.join(image_dir, f"{rel_stem}_{idx:03d}.npy")
        np.save(out_path, data)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    _, _, transform_test = get_octa_transform(args.img_size)
    dataset = OCTASegmentationDataset(
        args.data_path,
        img_size=args.img_size,
        transform=transform_test,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    output_dir = os.path.join(args.output_dir, f"{args.model.replace('/', '-')}-{args.dataset}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = Denoiser(args)
    _load_checkpoint(model, Path(output_dir, 'checkpoints', args.checkpoint), args.ema)
    epoch = args.checkpoint.replace('checkpoint-', '').replace('.pth', '')
    print(f"Loaded checkpoint '{args.checkpoint}' for epoch {epoch}")
    model.to(device)
    model.eval()

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []
    hd95_scores = []
    sample_idx = 0

    with torch.no_grad():
        for images, gts in tqdm(data_loader, desc="Processing images"):
            batch_preds = []
            batch_gts = []

            for idx in range(images.size(0)):
                image = images[idx]
                gt = gts[idx]
                orig_shape = image.shape

                # Extract patches
                patches, positions = _extract_patches(image, args.samp_patch_size, args.stride)

                # Process patches in mini-batches
                all_patch_preds = []
                for i in range(0, len(patches), args.batch_size):
                    batch_patches = patches[i:i + args.batch_size].to(device, non_blocking=True)
                    batch_patches = batch_patches * 2.0 - 1.0

                    with autocast_ctx:
                        patch_preds, intermediates = model.generate(batch_patches)

                    all_patch_preds.append(patch_preds.cpu())

                all_patch_preds = torch.cat(all_patch_preds, dim=0)

                # Reconstruct full mask
                full_pred = _reconstruct_from_patches(
                    all_patch_preds.squeeze(1), positions, orig_shape, args.samp_patch_size
                )

                batch_preds.append(full_pred)
                batch_gts.append(gt)

                # Save individual mask
                rel_name = f"sample_{sample_idx:03d}"
                _save_mask(output_dir, rel_name, full_pred, args.threshold, True, postfix="_pred", epoch=epoch)
                _save_mask(output_dir, rel_name, image, args.threshold, args.save_prob, postfix="_image", epoch=epoch, is_binary=False)
                _save_mask(output_dir, rel_name, gt, args.threshold, args.save_prob, postfix="_gt", epoch=epoch)
                # Save intermediate masks for the few samples
                if sample_idx % 20 == 0:
                    _save_intermediate_masks(output_dir, f"sample_{sample_idx:03d}_intermediate", intermediates, epoch=epoch)
                    _save_intermediate_masks(output_dir, f"sample_{sample_idx:03d}_patch", patches, epoch=epoch)
                sample_idx += 1

            if args.metrics:
                # Stack for batch metrics
                pred_batch = torch.stack(batch_preds).to(device)
                gts_batch = torch.stack(batch_gts).to(device)

                dice_scores.append(compute_dice_score(pred_batch, gts_batch, threshold=args.threshold))
                iou_scores.append(compute_iou_score(pred_batch, gts_batch, threshold=args.threshold))
                sensitivity_scores.append(compute_sensitivity(pred_batch, gts_batch, threshold=args.threshold))
                specificity_scores.append(compute_specificity(pred_batch, gts_batch, threshold=args.threshold))
                hd95_scores.append(compute_hausdorff_distance_95(pred_batch, gts_batch, threshold=args.threshold))

    if args.metrics and dice_scores:
        save_metrics_to_csv(output_dir, epoch, dice_scores, iou_scores,
                            sensitivity_scores, specificity_scores, hd95_scores)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
