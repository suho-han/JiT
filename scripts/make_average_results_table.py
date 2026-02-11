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
    step: int
    metrics: Dict[str, str]


METRIC_ORDER = ["Dice", "IoU", "Sensitivity", "Specificity", "HD95"]
DATASET_ALIASES = {
    "MoNuSeg": {"monuseg", "monu", "monu-seg", "monu_seg"},
    "OCTA500-6M": {"octa500-6m", "octa500_6m", "octa500", "octa500-600", "octa500_600"},
    "ISIC": {"isic"},
}
FIGURES_DIR = Path("outputs/figures")

logger = logging.getLogger(__name__)


def find_average_result_files(outputs_dir: str) -> List[str]:
    matches: List[str] = []
    for root, _dirs, files in os.walk(outputs_dir):
        for name in files:
            if name.startswith("average_results-") and name.endswith(".csv"):
                matches.append(os.path.join(root, name))
    return matches


def parse_model_dataset(folder_name: str, split_mode: str) -> Tuple[str, str]:
    parts = folder_name.split("-")
    if len(parts) == 1:
        return folder_name.replace("_", "-"), ""
    if split_mode == "first":
        if len(parts) >= 3 and parts[0].startswith("JiT"):
            model = "-".join(parts[:3])
            dataset = "-".join(parts[3:])
            return model.replace("_", "-"), dataset
        model = parts[0]
        dataset = "-".join(parts[1:])
        return model.replace("_", "-"), dataset
    model = "-".join(parts[:-1])
    dataset = parts[-1]
    return model.replace("_", "-"), dataset


def normalize_dataset_name(dataset: str) -> str:
    if not dataset:
        return dataset
    key = dataset.lower().replace(" ", "").replace("_", "-")
    for canonical, aliases in DATASET_ALIASES.items():
        if key in aliases or key == canonical.lower():
            return canonical
    return dataset


def read_metrics(csv_path: str) -> Optional[Dict[str, str]]:
    metrics: Dict[str, str] = {}
    try:
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                metric = (row.get("Metric") or "").strip()
                value = (row.get("Value") or "").strip()
                if metric:
                    metrics[metric] = value
        return metrics
    except Exception as e:
        logger.warning("Failed to read CSV file: %s (Error: %s)", csv_path, e)
        return None


def parse_step_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"average_results-(\d+)\.csv$", filename)
    if not match:
        return None
    return int(match.group(1))


def render_latex(rows: List[ResultRow], caption: str) -> str:
    headers = ["Model"] + METRIC_ORDER
    column_spec = "c" * len(headers)
    metric_ranks = build_metric_ranks(rows)

    lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage[a4paper,margin=1cm,landscape]{geometry}",
        "\\begin{document}",
        "\\begin{table}[ht]",
        "    \\centering",
        "    {\\footnotesize",
        f"        \\begin{{tabular}}{{{column_spec}}}",
        "            \\toprule",
        "            " + " & ".join(headers) + " \\\\",
        "            \\midrule",
    ]

    prev_group: Optional[str] = None
    for row in rows:
        group = model_group(row.model)
        if prev_group is not None and group != prev_group:
            lines.append("            \\midrule")
        values = [
            row.model,
        ]
        for metric in METRIC_ORDER:
            raw_value = row.metrics.get(metric, "--")
            values.append(format_metric_cell(raw_value, metric, metric_ranks))
        lines.append("            " + " & ".join(values) + " \\\\")
        prev_group = group

    lines.append("            \\bottomrule")
    lines.append("        \\end{tabular}")
    lines.append("    }")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append("\\end{table}")
    lines.append("\\end{document}")

    return "\n".join(lines) + "\n"


def parse_metric_value(value: str) -> Optional[float]:
    if not value or value.strip() in {"--", "nan", "NaN"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number:
        return None
    return number


def model_group(model: str) -> str:
    lowered = model.lower()
    if "condimgwave" in lowered:
        return "condimgwave"
    if "condimg" in lowered:
        return "condimg"
    if "jit-b" in lowered:
        return "jit-b"
    if "jit-l" in lowered:
        return "jit-l"
    if "jit-h" in lowered:
        return "jit-h"
    return "other"


def build_metric_ranks(rows: List[ResultRow]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    ranks: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for metric in METRIC_ORDER:
        values = (
            parse_metric_value(row.metrics.get(metric, ""))
            for row in rows
        )
        filtered = sorted({v for v in values if v is not None}, reverse=True)
        ranks[metric] = (filtered[0], filtered[1] if len(filtered) > 1 else None) if filtered else (None, None)
    return ranks


def format_metric_cell(
    value: str,
    metric: str,
    ranks: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> str:
    number = parse_metric_value(value)
    if number is None:
        return value
    best, second = ranks.get(metric, (None, None))
    if best is not None and number == best:
        return f"\\textbf{{{value}}}"
    if second is not None and number == second:
        return f"\\underline{{{value}}}"
    return value


def _compile_latex(tex_path: Path) -> None:
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
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        target_path = FIGURES_DIR / pdf_path.name
        if pdf_path.resolve() != target_path.resolve():
            shutil.copy(pdf_path, target_path)
        logger.info("PDF generated: %s", target_path)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    steps = (
        [args.epoch]
        if args.epoch is not None
        else [int(x) for x in args.steps.split(",") if x.strip()]
    )
    datasets_arg = [chunk.strip() for chunk in args.datasets.split(",") if chunk.strip()]

    files = find_average_result_files(args.outputs_dir)
    if not steps:
        latest_step = max(
            (parse_step_from_filename(os.path.basename(path)) for path in files),
            default=None,
        )
        steps = [latest_step] if latest_step is not None else []

    failed_files: List[str] = []
    rows: List[ResultRow] = []
    for path in files:
        step = parse_step_from_filename(os.path.basename(path))
        if step is None:
            continue
        metrics = read_metrics(path)
        if metrics is None:
            failed_files.append(path)
            continue
        rows.append(
            ResultRow(
                model=parse_model_dataset(os.path.basename(os.path.dirname(path)), args.name_split)[0],
                dataset=normalize_dataset_name(
                    parse_model_dataset(os.path.basename(os.path.dirname(path)), args.name_split)[1]
                ),
                step=step,
                metrics=metrics,
            )
        )

    if failed_files:
        logger.error("\n=== Failed to load %d CSV file(s) ===", len(failed_files))
        for failed_path in failed_files:
            logger.error("  - %s", failed_path)
        logger.error("==================================\n")

    if steps:
        step_set = set(steps)
        rows = [row for row in rows if row.step in step_set]
    if datasets_arg and datasets_arg[0].lower() == "all":
        datasets = list(DATASET_ALIASES.keys())
    else:
        datasets = [normalize_dataset_name(ds) for ds in datasets_arg]

    multiple = bool(datasets)
    if not datasets:
        datasets = sorted({row.dataset for row in rows})

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.dataset == dataset]
        group_order = {
            "jit-b": 0,
            "jit-l": 1,
            "jit-h": 2,
            "condimg": 3,
            "condimgwave": 4,
            "other": 5,
        }
        dataset_rows.sort(
            key=lambda r: (
                group_order.get(model_group(r.model), 99),
                r.model,
                r.step,
            )
        )
        caption = (
            args.caption.format(dataset=dataset)
            if "{dataset}" in args.caption
            else f"{args.caption} ({dataset})" if multiple else args.caption
        )
        out_tex = (
            args.out_tex.format(dataset=dataset)
            if "{dataset}" in args.out_tex
            else f"{os.path.splitext(args.out_tex)[0]}_{dataset}{os.path.splitext(args.out_tex)[1] or '.tex'}"
            if multiple else args.out_tex
        )
        table = render_latex(dataset_rows, caption)
        print(f"Writing LaTeX table to {out_tex}...")
        with open(out_tex, "w", encoding="utf-8") as handle:
            handle.write(table)
        _compile_latex(Path(out_tex))


if __name__ == "__main__":
    main()
