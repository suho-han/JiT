"""
Test evaluation metrics for segmentation.
Tests:
- Dice coefficient
- IoU (Intersection over Union)
- Sensitivity
- Specificity
- Hausdorff Distance 95 (HD95)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/suhohan/JiT')

from engine_jit import compute_dice_score, compute_hausdorff_distance_95, compute_iou_score, compute_sensitivity, compute_specificity


def test_dice_score():
    """Test Dice coefficient computation."""
    print("Testing Dice coefficient...")
    
    # Perfect prediction: should be 1.0
    pred_perfect = torch.ones(1, 1, 32, 32)
    target_perfect = torch.ones(1, 1, 32, 32)
    dice_perfect = compute_dice_score(pred_perfect, target_perfect)
    assert abs(dice_perfect - 1.0) < 1e-5, f"Perfect prediction should have Dice=1.0, got {dice_perfect}"
    print(f"✓ Perfect prediction: Dice = {dice_perfect:.6f}")
    
    # Opposite prediction: should be 0.0
    pred_opposite = torch.ones(1, 1, 32, 32)
    target_opposite = -torch.ones(1, 1, 32, 32)
    dice_opposite = compute_dice_score(pred_opposite, target_opposite)
    assert abs(dice_opposite) < 1e-5, f"Opposite prediction should have Dice≈0.0, got {dice_opposite}"
    print(f"✓ Opposite prediction: Dice = {dice_opposite:.6f}")
    
    # Partial overlap
    pred_partial = torch.ones(1, 1, 32, 32)
    target_partial = torch.zeros(1, 1, 32, 32)
    target_partial[:, :, :16, :] = 1.0
    dice_partial = compute_dice_score(pred_partial, target_partial)
    print(f"✓ Partial overlap (50%): Dice = {dice_partial:.6f}")
    assert 0 < dice_partial < 1, "Partial overlap should have 0 < Dice < 1"


def test_iou_score():
    """Test IoU (Intersection over Union) computation."""
    print("\nTesting IoU score...")
    
    # Perfect prediction: should be 1.0
    pred_perfect = torch.ones(1, 1, 32, 32)
    target_perfect = torch.ones(1, 1, 32, 32)
    iou_perfect = compute_iou_score(pred_perfect, target_perfect)
    assert abs(iou_perfect - 1.0) < 1e-5, f"Perfect prediction should have IoU=1.0, got {iou_perfect}"
    print(f"✓ Perfect prediction: IoU = {iou_perfect:.6f}")
    
    # No overlap: should be 0.0
    pred_no_overlap = torch.ones(1, 1, 32, 32)
    target_no_overlap = -torch.ones(1, 1, 32, 32)
    iou_no_overlap = compute_iou_score(pred_no_overlap, target_no_overlap)
    assert abs(iou_no_overlap) < 1e-5, f"No overlap should have IoU≈0.0, got {iou_no_overlap}"
    print(f"✓ No overlap: IoU = {iou_no_overlap:.6f}")
    
    # 50% overlap
    pred_partial = torch.ones(1, 1, 32, 32)
    target_partial = torch.zeros(1, 1, 32, 32)
    target_partial[:, :, :16, :] = 1.0
    iou_partial = compute_iou_score(pred_partial, target_partial)
    print(f"✓ 50% overlap: IoU = {iou_partial:.6f}")
    assert 0 < iou_partial < 1, "Partial overlap should have 0 < IoU < 1"


def test_sensitivity():
    """Test Sensitivity (Recall) computation."""
    print("\nTesting Sensitivity...")
    
    # Perfect prediction: should be 1.0
    pred_perfect = torch.ones(1, 1, 32, 32)
    target_perfect = torch.ones(1, 1, 32, 32)
    sensitivity_perfect = compute_sensitivity(pred_perfect, target_perfect)
    assert abs(sensitivity_perfect - 1.0) < 1e-5, f"Perfect prediction should have Sensitivity=1.0, got {sensitivity_perfect}"
    print(f"✓ Perfect prediction: Sensitivity = {sensitivity_perfect:.6f}")
    
    # All false negatives: should be 0.0
    pred_fn = -torch.ones(1, 1, 32, 32)
    target_fn = torch.ones(1, 1, 32, 32)
    sensitivity_fn = compute_sensitivity(pred_fn, target_fn)
    assert abs(sensitivity_fn) < 1e-5, f"All FN should have Sensitivity≈0.0, got {sensitivity_fn}"
    print(f"✓ All false negatives: Sensitivity = {sensitivity_fn:.6f}")
    
    # 50% true positives
    pred_half = torch.ones(1, 1, 32, 32)
    pred_half[:, :, 16:, :] = -1.0
    target_half = torch.ones(1, 1, 32, 32)
    sensitivity_half = compute_sensitivity(pred_half, target_half)
    print(f"✓ 50% detection rate: Sensitivity = {sensitivity_half:.6f}")
    assert 0 < sensitivity_half <= 1, "Sensitivity should be in (0, 1]"


def test_specificity():
    """Test Specificity computation."""
    print("\nTesting Specificity...")
    
    # Perfect balanced prediction: should be 1.0 (no FP, all TN correct)
    pred_perfect = torch.zeros(1, 1, 32, 32)
    pred_perfect[:, :, :16, :] = 1.0
    target_perfect = torch.zeros(1, 1, 32, 32)
    target_perfect[:, :, :16, :] = 1.0
    specificity_perfect = compute_specificity(pred_perfect, target_perfect)
    assert abs(specificity_perfect - 1.0) < 1e-5, f"Perfect prediction should have Specificity=1.0, got {specificity_perfect}"
    print(f"✓ Perfect balanced prediction: Specificity = {specificity_perfect:.6f}")
    
    # All false positives: should be 0.0
    pred_fp = torch.ones(1, 1, 32, 32)
    target_fp = -torch.ones(1, 1, 32, 32)
    specificity_fp = compute_specificity(pred_fp, target_fp)
    assert abs(specificity_fp) < 1e-5, f"All FP should have Specificity≈0.0, got {specificity_fp}"
    print(f"✓ All false positives: Specificity = {specificity_fp:.6f}")
    
    # All negative ground truth (no negatives predicted): should be 1.0
    pred_no_neg = torch.zeros(1, 1, 32, 32)
    target_no_neg = -torch.ones(1, 1, 32, 32)
    specificity_no_neg = compute_specificity(pred_no_neg, target_no_neg)
    assert abs(specificity_no_neg - 1.0) < 1e-5, f"No FP should have Specificity=1.0, got {specificity_no_neg}"
    print(f"✓ No false positives: Specificity = {specificity_no_neg:.6f}")


def test_hausdorff_distance_95():
    """Test Hausdorff Distance 95 computation."""
    print("\nTesting Hausdorff Distance 95...")
    
    # Perfect prediction: should be 0
    pred_perfect = torch.ones(1, 1, 32, 32)
    target_perfect = torch.ones(1, 1, 32, 32)
    hd95_perfect = compute_hausdorff_distance_95(pred_perfect, target_perfect)
    assert hd95_perfect < 1e-5, f"Perfect prediction should have HD95≈0, got {hd95_perfect}"
    print(f"✓ Perfect prediction: HD95 = {hd95_perfect:.6f}")
    
    # Shifted prediction
    pred_shifted = torch.zeros(1, 1, 32, 32)
    pred_shifted[:, :, 2:30, 2:30] = 1.0
    target_shifted = torch.zeros(1, 1, 32, 32)
    target_shifted[:, :, :28, :28] = 1.0
    hd95_shifted = compute_hausdorff_distance_95(pred_shifted, target_shifted)
    print(f"✓ Shifted prediction (2px offset): HD95 = {hd95_shifted:.6f}")
    assert hd95_shifted > 0, "Different prediction should have HD95 > 0"
    
    # Completely different
    pred_different = torch.ones(1, 1, 32, 32)
    target_different = torch.zeros(1, 1, 32, 32)
    hd95_different = compute_hausdorff_distance_95(pred_different, target_different)
    print(f"✓ Completely different: HD95 = {hd95_different:.6f}")
    assert hd95_different > 0, "Different predictions should have HD95 > 0"


def test_metric_consistency():
    """Test that metrics are consistent across batch."""
    print("\nTesting metric consistency across batches...")
    
    # Create batch with same prediction repeated
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randn(4, 1, 32, 32)
    
    # Compute batch metrics
    dice_batch = compute_dice_score(pred, target)
    iou_batch = compute_iou_score(pred, target)
    sens_batch = compute_sensitivity(pred, target)
    spec_batch = compute_specificity(pred, target)
    hd95_batch = compute_hausdorff_distance_95(pred, target)
    
    # Compute individual metrics
    dice_individual = [compute_dice_score(pred[i:i+1], target[i:i+1]) for i in range(4)]
    
    print(f"✓ Batch Dice: {dice_batch:.6f}")
    print(f"✓ Individual Dice: {dice_individual}")
    print(f"✓ Batch IoU: {iou_batch:.6f}")
    print(f"✓ Batch Sensitivity: {sens_batch:.6f}")
    print(f"✓ Batch Specificity: {spec_batch:.6f}")
    print(f"✓ Batch HD95: {hd95_batch:.6f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Evaluation Metrics")
    print("=" * 60)
    
    test_dice_score()
    test_iou_score()
    test_sensitivity()
    test_specificity()
    test_hausdorff_distance_95()
    test_metric_consistency()
    
    print("\n" + "=" * 60)
    print("All metric tests passed! ✓")
    print("=" * 60)
