import argparse
import os

import autorootcwd
import natsort
import numpy as np
import torch
from make_average_results_table import make_average_results_table

from src.engine_jit import compute_aji, compute_cldice, compute_dice_score, compute_hausdorff_distance_95, compute_iou_score, compute_sensitivity, compute_specificity, save_metrics_to_csv

parser = argparse.ArgumentParser(description="Evaluate segmentation model")
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name (e.g., ISIC, OCTA500_6M<)')
parser.add_argument('--model', type=str, default=None,
                    help='Model name to filter experiments')
parser.add_argument('--epoch', type=int, default=None,
                    help='Epoch number for evaluation')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold for binarizing predictions')
args = parser.parse_args()


def calculate(images_dir, output_dir, dataset, epoch=None, threshold=0.5, soft_vote=False):
    preds = [i for i in os.listdir(images_dir) if i.endswith('pred.npy')]
    gts = [i for i in os.listdir(images_dir) if i.endswith('gt.npy')]

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []
    hd95_scores = []
    aji_scores = []
    cldice_scores = []
    for pred_file, gt_file in zip(sorted(preds), sorted(gts)):
        pred_path = os.path.join(images_dir, pred_file)
        gt_path = os.path.join(images_dir, gt_file)

        pred = np.load(pred_path)
        pred = torch.from_numpy(pred)
        gt = np.load(gt_path)
        gt = torch.from_numpy(gt)

        dice_scores.append(compute_dice_score(pred, gt, threshold=threshold))
        iou_scores.append(compute_iou_score(pred, gt, threshold=threshold))
        sensitivity_scores.append(
            compute_sensitivity(pred, gt, threshold=threshold))
        specificity_scores.append(
            compute_specificity(pred, gt, threshold=threshold))
        hd95_scores.append(compute_hausdorff_distance_95(
            pred, gt, threshold=threshold))
        if 'ISIC' in dataset:
            aji_scores.append(compute_aji(pred, gt, threshold=threshold))
        if 'OCTA' in dataset:
            cldice_scores.append(compute_cldice(pred, gt, threshold=threshold))

    if dice_scores:
        save_metrics_to_csv(output_dir, epoch,
                            dice_scores, iou_scores, sensitivity_scores, specificity_scores, hd95_scores,
                            aji_scores if 'ISIC' in dataset else None,
                            cldice_scores if 'OCTA' in dataset else None,
                            args)


if __name__ == "__main__":
    exps = [d for d in os.listdir('outputs') if args.dataset in d]
    if args.model:
        exps = [d for d in exps if args.model in d]

    print_msg = f"Evaluating on dataset: {args.dataset}"
    if args.model:
        print_msg += f", model: {args.model}"
    print(print_msg)
    args.soft_vote = None

    for exp in exps:
        base_dir = os.path.join('outputs', exp)
        print(f"Processing experiment directory: {base_dir}")
        if args.epoch is None:
            epoch_dirs = [d for d in os.listdir(base_dir) if d.startswith(
                'images-') and not d.endswith('last')]
            if not epoch_dirs:
                print(f"No epoch directories found in {base_dir}. Skipping...")
                continue
            epochs = natsort.natsorted(
                [int(d.split('-')[1]) for d in epoch_dirs])
            soft_vote = base_dir.endswith('-soft_vote')
            epoch = max(epochs)
        else:
            epoch = args.epoch
            soft_vote = base_dir.endswith('-soft_vote')

        # Construct the images directory path
        images_dir_candidates = [d for d in os.listdir(
            base_dir) if d.startswith(f'images-{epoch}')]
        if not images_dir_candidates:
            print(
                f"No images directory found for epoch {epoch} in {base_dir}. Skipping...")
            continue

        images_dir = os.path.join(base_dir, images_dir_candidates[0])
        calculate(images_dir, base_dir, args.dataset, epoch,
                  threshold=args.threshold, soft_vote=soft_vote)
    make_average_results_table()
