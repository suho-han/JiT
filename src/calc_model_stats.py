import argparse
import ast
import csv
import os
import sys

import autorootcwd
import torch
from thop import clever_format, profile

from src.denoiser import Denoiser
from src.main_jit import get_args_parser

MODELS_FOR_STATS = [
    "JiT-B/16",
    "JiT-L/16",
    "JiT-H/16",
    "JiT_CondImg-B/16",
    "JiT_CondImg-L/16",
    "JiT_CondImg-H/16",
    "JiT_ParaCond-B/16",
    "JiT_ParaCond-L/16",
    "JiT_ParaCond-H/16",
    "JiT_ParaCondWave-B/16",
    "JiT_ParaCondWave-L/16",
    "JiT_ParaCondWave-H/16",
]


def _calculate_stats(args):
    print(f"Instantiating model: {args.model}")
    try:
        model = Denoiser(args)
    except TypeError as e:
        if "cond_weight" in str(e):
            print(
                f"Warning: {args.model} does not support cond_weight. Passing None instead.", file=sys.stderr)
            args.cond_weight = None
            # If Denoiser is hardcoded to pass cond_weight, we might need a workaround for basic JiT models.
            # But normally we just run with models that do support it or we patch kwargs.
            raise
        else:
            raise

    model.eval()

    # 1. Parameter calculation
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params / 1e6:.6f} M")

    # 2. GFLOPs calculation
    # Denoiser forward: forward(self, x, cond)
    # x: (bs, mask_channel, img_size, img_size)
    # cond: (bs, img_channel, img_size, img_size)

    dummy_x = torch.randn(1, args.mask_channel, args.img_size, args.img_size)
    dummy_cond = torch.randn(1, args.img_channel, args.img_size, args.img_size)

    result = {
        "model": args.model,
        "trainable_params": n_params,
        "trainable_params_m": n_params / 1e6,
        "macs": None,
        "gflops": None,
        "thop_params": None,
    }

    try:
        macs, params = profile(model, inputs=(dummy_x, dummy_cond), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        print(f"MACs (approx. FLOPs): {macs_str}")
        print(f"GFLOPs: {macs / 1e9:.3f} G")
        print(f"Total Params (thop): {params_str}")
        result["macs"] = macs
        result["gflops"] = macs / 1e9
        result["thop_params"] = params
    except Exception as e:
        print(f"Failed to calculate GFLOPs using thop: {e}")

    return result


def main():
    parser = get_args_parser()
    parser.add_argument('--stats_csv', default='outputs/model_stats.csv', type=str,
                        help='CSV path to store model stats')
    args, _ = parser.parse_known_args()

    # Process cond_weight from string to dict, just like main_jit.py does
    if hasattr(args, 'cond_weight') and isinstance(args.cond_weight, str):
        try:
            args.cond_weight = ast.literal_eval(args.cond_weight)
        except Exception:
            pass

    # Set default values if not provided
    if not hasattr(args, 'img_channel'):
        args.img_channel = 3
    if not hasattr(args, 'mask_channel'):
        args.mask_channel = 1
    if not hasattr(args, 'img_size'):
        args.img_size = 256

    results = []
    for model_name in MODELS_FOR_STATS:
        args.model = model_name
        try:
            results.append(_calculate_stats(args))
        except Exception as e:
            print(f"Failed to calculate stats for {model_name}: {e}", file=sys.stderr)

    if results:
        csv_path = args.stats_csv
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        fieldnames = [
            "model",
            "trainable_params",
            "trainable_params_m",
            "macs",
            "gflops",
            "thop_params",
        ]
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved stats CSV to: {csv_path}")


if __name__ == '__main__':
    main()
