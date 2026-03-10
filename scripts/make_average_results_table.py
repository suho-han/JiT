#!/usr/bin/env python3
"""Generate a LaTeX table from outputs/**/average_results-*.csv."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table from average_results CSV files.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root outputs directory to scan (default: outputs).",
    )
    parser.add_argument(
        "--out-tex",
        default="outputs/figures/average_results_table.tex",
        help="Path to write the LaTeX table.",
    )
    parser.add_argument(
        "--caption",
        default="Average results",
        help="LaTeX table caption text.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated dataset names to include (e.g., MoNuSeg,OCTA500-6M,ISIC2016,ISIC2018) "
            "or 'all' to generate all supported datasets."
        ),
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated step numbers to include (e.g., 9000,10000).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help=(
            "Single epoch/step to include. If omitted and --steps is empty, "
            "the highest average_results-<step>.csv is selected."
        ),
    )
    parser.add_argument(
        "--name-split",
        choices=["first", "last"],
        default="first",
        help="Split folder name into model/dataset by first or last '-' (default: first).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ResultRow:
    model: str
    dataset: str
    group: str
    step: int
    metrics: Dict[str, str]
    patch_size: str = ""  # e.g., '16' or '32'
    add_loss: str = "--"
    cond_weight: str = "--"


METRIC_ORDER = ["Dice", "IoU", "Sensitivity", "Specificity", "HD95"]
DATASET_ALIASES = {
    "MoNuSeg": {"monuseg", "monu", "monu-seg", "monu_seg"},
    "OCTA500-6M": {"octa500-6m", "octa500_6m", "octa500", "octa500-600", "octa500_600"},
    "ISIC2016": {"isic2016"},
    "ISIC2018": {"isic2018"},
}
FIGURES_DIR = Path("outputs/figures")

logger = logging.getLogger(__name__)


def render_latex_table_content(rows: List[ResultRow], caption: str) -> str:
    """Generate table content without document wrapper."""
    def format_model_name(model: str) -> str:
        model = re.sub(r"-(\d+)", r"/\1", model)
        return model.replace("_", r"\_")

    def parse_metric_value(value: str) -> Optional[float]:
        if not value or value.strip() in {"--", "nan", "NaN"}:
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        return number if number == number else None

    available_metrics = set()
    for row in rows:
        available_metrics.update(row.metrics.keys())

    metric_order = ["Dice", "IoU", "Sensitivity", "Specificity", "HD95"]
    if "AJI" in available_metrics:
        metric_order.append("AJI")
    if "clDice" in available_metrics:
        metric_order.append("clDice")

    headers = ["Model", "Add Loss", "Cond Weight", "Step"] + metric_order
    column_spec = "l" + "c" * (len(headers) - 1)
    metric_ranks: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for metric in metric_order:
        values = [
            parse_metric_value(row.metrics.get(metric, ""))
            for row in rows
        ]
        if metric == "HD95":
            filtered = sorted({v for v in values if v is not None})
        else:
            filtered = sorted(
                {v for v in values if v is not None}, reverse=True)
        metric_ranks[metric] = (
            filtered[0],
            filtered[1] if len(filtered) > 1 else None,
        ) if filtered else (None, None)

    lines = [
        "    \\begin{table}[ht]",
        "        \\centering",
        "        {\\footnotesize",
        f"            \\begin{{tabular}}{{{column_spec}}}",
        "                \\toprule",
        "                " + " & ".join(headers) + " \\\\",
        "                \\midrule",
    ]

    prev_group: Optional[str] = None
    for row in rows:
        group = row.group
        if prev_group is not None and group != prev_group:
            lines.append("                \\midrule")
        values = [
            format_model_name(row.model),
            row.add_loss.replace("_", r"\_"),
            row.cond_weight.replace("_", r"\_"),
            str(row.step),
        ]
        for metric in metric_order:
            raw_value = row.metrics.get(metric, "--")
            number = parse_metric_value(raw_value)
            if number is None:
                cell = raw_value
            else:
                best, second = metric_ranks.get(metric, (None, None))
                if best is not None and number == best:
                    cell = f"\\textbf{{{raw_value}}}"
                elif second is not None and number == second:
                    cell = f"\\underline{{{raw_value}}}"
                else:
                    cell = raw_value
            values.append(cell)
        lines.append("                " + " & ".join(values) + " \\\\")
        prev_group = group

    lines.append("                \\bottomrule")
    lines.append("            \\end{tabular}")
    lines.append("        }")
    lines.append(f"        \\caption{{{caption}}}")
    lines.append("    \\end{table}")

    return "\n".join(lines) + "\n"


def render_latex(rows: List[ResultRow], caption: str) -> str:
    """Generate complete LaTeX document (deprecated, use render_latex_table_content)."""
    table_content = render_latex_table_content(rows, caption)
    lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage[a4paper,margin=1cm,landscape]{geometry}",
        "\\begin{document}",
    ]
    lines.append(table_content)
    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def _compile_latex(tex_path: Path, copy_to_figures: bool = True) -> None:
    """Compile LaTeX file to PDF."""
    try:
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-quiet",
                f"-output-directory={tex_path.parent}",
                str(tex_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        if copy_to_figures:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            target_path = FIGURES_DIR / pdf_path.name
            if pdf_path.resolve() != target_path.resolve():
                shutil.copy(pdf_path, target_path)
            logger.info("PDF generated: %s", target_path)
        else:
            logger.info("PDF generated: %s", pdf_path)
        try:
            subprocess.run(
                [
                    "latexmk",
                    "-c",
                    f"-output-directory={tex_path.parent}",
                    str(tex_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def make_average_results_table() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    step_pattern = re.compile(r"^average_results-(\d+)")

    def normalize_dataset_name(dataset: str) -> str:
        if not dataset:
            return dataset
        key = dataset.lower().replace(" ", "").replace("_", "-")
        for canonical, aliases in DATASET_ALIASES.items():
            if key in aliases or key == canonical.lower():
                return canonical
        return dataset

    def model_group(model: str) -> str:
        lowered = model.lower()
        if "condimgwave" in lowered:
            return "condimgwave"
        if "condimg" in lowered:
            return "condimg"
        if "paracondwave" in lowered:
            return "paracondwave"
        if "paracond" in lowered:
            return "paracond"
        if "jit" in lowered:
            return "jit"
        return "other"

    steps = (
        [args.epoch]
        if args.epoch is not None
        else [int(x) for x in args.steps.split(",") if x.strip()]
    )
    datasets_arg = [chunk.strip()
                    for chunk in args.datasets.split(",") if chunk.strip()]

    files: List[str] = []
    for root, _dirs, filenames in os.walk(args.outputs_dir):
        for name in filenames:
            if name.startswith("average_results-") and name.endswith(".csv"):
                files.append(os.path.join(root, name))

    failed_files: List[str] = []
    rows: List[ResultRow] = []
    for path in files:
        match = step_pattern.match(os.path.basename(path))
        if not match:
            continue
        step = int(match.group(1))
        metrics: Dict[str, str] = {}
        try:
            with open(path, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    metric = (row.get("Metric") or "").strip()
                    value = (row.get("Value") or "").strip()
                    if metric:
                        metrics[metric] = value
        except Exception as e:
            logger.warning("Failed to read CSV file: %s (Error: %s)", path, e)
            failed_files.append(path)
            continue
        folder_name = os.path.basename(os.path.dirname(path))

        loss_suffix_match = re.search(r"-([a-z0-9_]+)$", folder_name)
        add_loss = "--"
        if loss_suffix_match and loss_suffix_match.group(1) in {"dice_bce"}:
            add_loss = loss_suffix_match.group(1)
            folder_name = folder_name[:loss_suffix_match.start()]

        # Check if the folder contains a cond_weight string (e.g. -ffs, -lfs, etc)
        # Assuming cond_weight string is exactly 3 lowercase letters surrounded by hyphens
        cond_match = re.search(r"-([a-z]{3})-(?!.*-[a-z]{3}-)", folder_name)

        cond_weight = "--"
        if cond_match:
            # Reconstruct model and dataset around the matched cond_weight string
            model = folder_name[:cond_match.start()]
            dataset = folder_name[cond_match.end():]

            cond_weight = cond_match.group(1)
            model = model.replace("_", "-")
        else:
            # Fallback to old logic
            parts = folder_name.split("-")
            if len(parts) == 1:
                model = folder_name.replace("_", "-")
                dataset = ""
            elif args.name_split == "first":
                if len(parts) >= 3 and parts[0].startswith("JiT"):
                    model = "-".join(parts[:3])
                    dataset = "-".join(parts[3:])
                else:
                    model = parts[0]
                    dataset = "-".join(parts[1:])
                model = model.replace("_", "-")
            else:
                model = "-".join(parts[:-1]).replace("_", "-")
                dataset = parts[-1]

        # Extract patch size from model name (e.g., "JiT-B-16")
        patch_size = ""
        match_patch = re.search(r"-(\d+)$", model)
        if match_patch:
            patch_size = match_patch.group(1)

        rows.append(
            ResultRow(
                model=model,
                dataset=normalize_dataset_name(dataset),
                group=model_group(model),
                step=step,
                metrics=metrics,
                patch_size=patch_size,
                add_loss=add_loss,
                cond_weight=cond_weight,
            )
        )

    if failed_files:
        logger.error("\n=== Failed to load %d CSV file(s) ===",
                     len(failed_files))
        for failed_path in failed_files:
            logger.error("  - %s", failed_path)
        logger.error("==================================\n")

    if not steps:
        max_steps: Dict[Tuple[str, str, str, str], int] = {}
        for row in rows:
            key = (row.model, row.dataset, row.add_loss, row.cond_weight)
            max_steps[key] = max(row.step, max_steps.get(key, row.step))
        rows = [
            row
            for row in rows
            if row.step == max_steps.get((row.model, row.dataset, row.add_loss, row.cond_weight))
        ]
    elif steps:
        step_set = set(steps)
        rows = [row for row in rows if row.step in step_set]

    if datasets_arg and datasets_arg[0].lower() == "all":
        datasets = list(DATASET_ALIASES.keys())
    else:
        datasets = [normalize_dataset_name(ds) for ds in datasets_arg]

    multiple = bool(datasets)
    if not datasets:
        datasets = sorted({row.dataset for row in rows})

    step_tag = "-".join(str(step) for step in steps) if steps else ""
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.dataset == dataset]

        # Separate by patch size (16 and 32)
        patch_sizes = sorted(
            {row.patch_size for row in dataset_rows if row.patch_size})
        patch_sizes = patch_sizes if patch_sizes else [""]

        group_order = {
            "jit": 1,
            "condimg": 2,
            "paracond": 3,
            "paracondwave": 4,
            "other": 5,
        }

        def model_size_rank(model: str) -> int:
            lowered = model.lower()
            if "-b-" in lowered or lowered.endswith("-b"):
                return 1
            if "-l-" in lowered or lowered.endswith("-l"):
                return 2
            if "-h-" in lowered or lowered.endswith("-h"):
                return 3
            return 99

        # Collect all table contents
        table_contents = []
        for patch_size in patch_sizes:
            if patch_size:
                patch_rows = [
                    row for row in dataset_rows if row.patch_size == patch_size]
            else:
                patch_rows = [
                    row for row in dataset_rows if not row.patch_size]

            if not patch_rows:
                continue

            patch_rows.sort(
                key=lambda r: (
                    group_order.get(r.group, 99),
                    model_size_rank(r.model),
                    r.model,
                    0 if r.add_loss == "--" else 1,
                    r.add_loss,
                    r.cond_weight,
                    r.step,
                )
            )

            # Build caption for this patch size
            patch_suffix = f" (\\mbox{{patch{patch_size}}})" if patch_size else ""
            dataset_formatted = f"\\mbox{{{dataset}}}"
            caption = (
                args.caption.format(dataset=dataset_formatted)
                if "{dataset}" in args.caption
                else f"{args.caption} ({dataset_formatted}){patch_suffix}" if multiple else args.caption + patch_suffix
            )

            # Generate table content (without document wrapper)
            table_content = render_latex_table_content(patch_rows, caption)
            table_contents.append(table_content)

        # Write all tables to a single file
        if table_contents:
            out_tex = (
                args.out_tex.format(dataset=dataset)
                if "{dataset}" in args.out_tex
                else f"{os.path.splitext(args.out_tex)[0]}_{dataset}{os.path.splitext(args.out_tex)[1] or '.tex'}"
                if multiple else args.out_tex
            )

            if step_tag:
                out_tex_path = Path(out_tex)
                suffix = out_tex_path.suffix or ".tex"
                out_tex = str(
                    Path("outputs")
                    / "steps"
                    / step_tag
                    / f"{out_tex_path.stem}_{step_tag}{suffix}"
                )

            os.makedirs(os.path.dirname(out_tex), exist_ok=True)

            # Build complete LaTeX document
            lines = [
                "\\documentclass{article}",
                "\\usepackage{booktabs}",
                "\\usepackage[a4paper,margin=1cm,landscape]{geometry}",
                "\\begin{document}",
            ]
            lines.extend(table_contents)
            lines.append("\\end{document}")

            full_document = "\n".join(lines) + "\n"
            print(f"Writing LaTeX table to {out_tex}...")
            with open(out_tex, "w", encoding="utf-8") as handle:
                handle.write(full_document)
            _compile_latex(Path(out_tex), copy_to_figures=not step_tag)


if __name__ == "__main__":
    make_average_results_table()
