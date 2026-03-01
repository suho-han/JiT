import argparse
import ast
import contextlib
import datetime
import os
from functools import partial
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
from engine_jit import compute_aji, compute_cldice, compute_dice_score, compute_hausdorff_distance_95, compute_iou_score, compute_sensitivity, compute_specificity, save_metrics_to_csv
from util.isicdataset import ISICSegmentationDataset, SameSizeBatchSampler, get_isic_transform
from util.monudataset import MoNuSegmentationDataset, get_monu_transform
from util.octadataset import OCTASegmentationDataset, get_octa_transform


def get_args_parser():
    parser = argparse.ArgumentParser("JiT inference", add_help=False)

    # architecture
    parser.add_argument("--model", default="JiT-B/16",
                        type=str, metavar="MODEL")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--mask_channel", type=int, default=1)
    parser.add_argument('--cond_weight', type=str, default=None,
                        help='Weight configs for cond, low_cond, high_cond')

    # sampling
    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--interval_min", default=0.0, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)
    parser.add_argument("--soft_vote", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_samples", default=1, type=int, help="Number of samples for ensemble averaging")

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
    parser.add_argument("--dataset", default="OCTA500_6M",
                        type=str, help="Dataset name")
    parser.add_argument("--data_path", default="data",
                        type=str, help="Path to images or dataset split")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="Path to checkpoint (.pth)")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--same_size_batch", action="store_true")
    parser.add_argument("--add_loss", action="store_true")
    parser.add_argument("--add_loss_name", default="aux_loss", type=str)
    parser.add_argument("--add_loss_weight", default=1.0, type=float)

    # patch sampling
    parser.add_argument("--samp_patch_size", default=256, type=int)
    parser.add_argument("--stride", default=128, type=int)

    # inference options
    parser.add_argument("--ema", default="ema1",
                        choices=["none", "ema1", "ema2"], type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--metrics", action="store_true",
                        help="Compute metrics if labels exist")
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


def _reconstruct_from_patches(
    patches: torch.Tensor,
    positions: list,
    orig_shape: tuple,
    samp_patch_size: int,
    soft_vote: bool = False,
    threshold: float = 0.5,
):
    """Reconstruct full image from overlapping patches using soft or hard voting."""
    if len(orig_shape) == 2:
        orig_shape = (1, orig_shape[0], orig_shape[1])
    c, h, w = orig_shape
    vote_sum = torch.zeros(
        (c, h, w), device=patches.device, dtype=torch.float32)
    vote_count = torch.zeros(
        (h, w), device=patches.device, dtype=torch.float32)

    # Hard voting: threshold patches at 0.5 first, then accumulate votes
    for patch, (top, left) in zip(patches, positions):
        if soft_vote:
            _patch = patch
        else:
            _patch = (patch > threshold).float()
        vote_sum[:, top:top + samp_patch_size, left:left + samp_patch_size] += _patch
        vote_count[top:top + samp_patch_size, left:left + samp_patch_size] += 1

    # Calculate voting confidence (ratio of votes that were 1)
    output = vote_sum / vote_count.clamp(min=1.0).unsqueeze(0)
    return (output >= threshold).float(), output


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


def _save_mask(output_dir: str, rel_name: str, data: torch.Tensor, threshold: float, postfix: str = "_mask", epoch: int = None, is_binary: bool = False):
    # Remove all singleton dimensions and get 2D array
    while data.ndim > 2 and data.shape[0] == 1:
        data = data.squeeze(0)

    rel_stem = os.path.splitext(rel_name)[0]
    out_path = os.path.join(output_dir, f"{rel_stem}{postfix}.npy")

    if not is_binary:
        data = data.cpu().numpy()
        np.save(out_path, data)
    else:
        binary = (data > threshold).cpu().numpy()
        np.save(out_path, binary)


def _save_intermediate_masks(output_dir: str, rel_name: str, intermediates: list, epoch: int = None):
    rel_stem = os.path.splitext(rel_name)[0]
    for idx, intermediate in enumerate(intermediates):
        while intermediate.ndim > 2 and intermediate.shape[0] == 1:
            intermediate = intermediate.squeeze(0)
        if intermediate.min() < 0.0 or intermediate.max() > 1.0:
            intermediate = (intermediate + 1.0) / 2.0
        data = intermediate.cpu().numpy()
        out_path = os.path.join(output_dir, f"{rel_stem}_{idx:03d}.npy")
        np.save(out_path, data)


def _log(message: str, log_path: Path | None = None):
    print(message, flush=True)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def main(args):
    if isinstance(args.cond_weight, str):
        try:
            args.cond_weight = ast.literal_eval(args.cond_weight)
        except Exception:
            pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    cond_weight_str = ""
    if args.cond_weight is not None:
        try:
            c = args.cond_weight.get('cond', 'fixed')[0]
            l = args.cond_weight.get('low_cond', 'fixed')[0]
            h = args.cond_weight.get('high_cond', 'fixed')[0]
            cond_weight_str = f"-{c}{l}{h}"
        except Exception:
            pass
    else:
        cond_weight_str = ""

    output_dir = os.path.join(args.output_dir, f"{args.model.replace('/', '-')}{cond_weight_str}-{args.dataset}{f'-{args.add_loss_name}' if args.add_loss else ''}")
    if not os.path.exists(output_dir):
        raise ValueError(f"output_dir does not exist: {output_dir}. Please run training first to generate checkpoints and create the output directory.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir, "logs")
    log_path = log_dir / f"inference_{timestamp}.log"

    _log("[1/4] Build dataset and dataloader", log_path)

    if 'OCTA' in args.dataset:
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
    elif 'MoNuSeg' in args.dataset:
        _,  transform_test = get_monu_transform(image_size=args.img_size)
        dataset = MoNuSegmentationDataset(
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
    elif 'ISIC' in args.dataset:
        _, _, transform_test = get_isic_transform(args.img_size)
        dataset = ISICSegmentationDataset(
            args.data_path,
            img_size=args.img_size,
            transform=transform_test,
        )
        use_same_size = args.same_size_batch or "test" in args.data_path
        if use_same_size:
            batch_sampler = SameSizeBatchSampler(
                dataset.get_image_sizes(),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                shuffle=False,
            )

    _log(f"[2/4] Initialize model and load checkpoint from {Path(output_dir, 'checkpoints', args.checkpoint)}", log_path)
    model = Denoiser(args)
    _load_checkpoint(model, Path(output_dir, 'checkpoints', args.checkpoint), args.ema)
    epoch = args.checkpoint.replace('checkpoint-', '').replace('.pth', '')
    _log(f"Loaded checkpoint '{args.checkpoint}' for epoch {epoch}", log_path)
    model.to(device)
    model.eval()

    # Print the cond_w configurations
    net = model.net if hasattr(model, 'net') else getattr(model, 'module', model).net
    _log("\n[Condition Weights]", log_path)
    if hasattr(net, 'cond_weight'):
        _log(f"Config: {net.cond_weight}", log_path)
    if hasattr(net, 'shared_cond_w'):
        _log(f"shared_cond_w: {net.shared_cond_w.mean().item():.4f}", log_path)
    if hasattr(net, 'shared_low_cond_w'):
        _log(f"shared_low_cond_w: {net.shared_low_cond_w.mean().item():.4f}", log_path)
    if hasattr(net, 'shared_high_cond_w'):
        _log(f"shared_high_cond_w: {net.shared_high_cond_w.mean().item():.4f}", log_path)

    if hasattr(net, 'blocks') and len(net.blocks) > 0:
        first_block = net.blocks[0]
        if hasattr(first_block, 'cond_mode') and first_block.cond_mode in ['learnable', 'zero_init']:
            _log(f"block[0].cond_w: {first_block.cond_w.mean().item():.4f} (mode: {first_block.cond_mode})", log_path)
        if hasattr(first_block, 'low_cond_mode') and first_block.low_cond_mode in ['learnable', 'zero_init']:
            _log(f"block[0].low_cond_w: {first_block.low_cond_w.mean().item():.4f} (mode: {first_block.low_cond_mode})", log_path)
        if hasattr(first_block, 'high_cond_mode') and first_block.high_cond_mode in ['learnable', 'zero_init']:
            _log(f"block[0].high_cond_w: {first_block.high_cond_w.mean().item():.4f} (mode: {first_block.high_cond_mode})", log_path)
    _log("", log_path)

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
    aji_scores = []
    cldice_scores = []
    sample_idx = 0

    image_dir = Path(output_dir, f"images-{epoch}{'-softvote' if args.soft_vote else ''}{f'-multi-{args.num_samples}' if args.num_samples > 1 else ''}" if epoch is not None else "images")
    _log(f"Saving results to {image_dir}", log_path)
    intermediates_dir = image_dir / "intermediates"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(intermediates_dir, exist_ok=True)

    _log("[3/4] Run inference", log_path)
    with torch.no_grad():
        for images, gts in tqdm(data_loader, desc="Processing images"):
            for idx in range(images.size(0)):
                image = images[idx]
                gt = gts[idx]
                orig_shape = gt.shape

                # Extract patches
                patches, positions = _extract_patches(image, args.samp_patch_size, args.stride)

                # Process patches in mini-batches
                all_patch_preds = []
                for i in range(0, len(patches), args.batch_size):
                    batch_patches = patches[i:i + args.batch_size].to(device, non_blocking=True)
                    batch_patches = batch_patches * 2.0 - 1.0

                    batch_sum_preds = None
                    for _ in range(args.num_samples):
                        with autocast_ctx:
                            patch_preds, _ = model.generate(batch_patches)
                        
                        if batch_sum_preds is None:
                            batch_sum_preds = patch_preds.cpu().float()
                        else:
                            batch_sum_preds += patch_preds.cpu().float()
                    
                    batch_avg_preds = batch_sum_preds / args.num_samples
                    all_patch_preds.append(batch_avg_preds)

                all_patch_preds = torch.cat(all_patch_preds, dim=0)

                # Reconstruct full mask
                full_pred_binary, full_pred = _reconstruct_from_patches(
                    all_patch_preds.squeeze(1),
                    positions,
                    orig_shape,
                    args.samp_patch_size,
                    args.soft_vote,
                    args.threshold,
                )

                # Save individual mask
                rel_name = f"sample_{sample_idx:03d}"
                save_mask = partial(_save_mask, output_dir=image_dir, rel_name=rel_name, threshold=args.threshold, epoch=epoch)
                save_mask(data=full_pred, postfix="_prob")
                save_mask(data=full_pred_binary, postfix="_pred", is_binary=True)
                save_mask(data=image, postfix="_image")
                save_mask(data=gt, postfix="_gt", is_binary=True)
                # Save intermediate masks for the few samples
                if sample_idx % 20 == 0:
                    _save_intermediate_masks(intermediates_dir, f"sample_{sample_idx:03d}_intermediate", intermediates, epoch=epoch)
                    _save_intermediate_masks(intermediates_dir, f"sample_{sample_idx:03d}_patch", patches, epoch=epoch)
                if args.metrics:
                    pred_single = full_pred_binary.to(device).unsqueeze(0)
                    gt_single = gt.to(device).unsqueeze(0)
                    dice_scores.append(compute_dice_score(pred_single, gt_single, threshold=args.threshold))
                    iou_scores.append(compute_iou_score(pred_single, gt_single, threshold=args.threshold))
                    sensitivity_scores.append(compute_sensitivity(pred_single, gt_single, threshold=args.threshold))
                    specificity_scores.append(compute_specificity(pred_single, gt_single, threshold=args.threshold))
                    hd95_scores.append(compute_hausdorff_distance_95(pred_single, gt_single, threshold=args.threshold))
                    if 'ISIC' in args.dataset:
                        aji_scores.append(compute_aji(pred_single, gt_single, threshold=args.threshold))
                    if 'OCTA' in args.dataset:
                        cldice_scores.append(compute_cldice(pred_single, gt_single, threshold=args.threshold))

                sample_idx += 1

    if args.metrics and dice_scores:
        _log("[4/4] Save metrics", log_path)
        save_metrics_to_csv(output_dir, epoch, dice_scores, iou_scores, sensitivity_scores, specificity_scores, hd95_scores, aji_scores, cldice_scores, args.soft_vote, dataset=args.dataset)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
