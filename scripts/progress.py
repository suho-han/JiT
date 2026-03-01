#!/usr/bin/env python3
"""
Convert bash progress.sh to Python for running process monitoring and checkpoint tracking.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NO_STEP_VALUE = "N/A"


def append_blank_line(lines: List[str]) -> None:
    """Append a blank line only when the previous line is not blank."""
    if lines and lines[-1] != "":
        lines.append("")


def check_nvidia_smi() -> None:
    """Check if nvidia-smi is available."""
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found in PATH.", file=sys.stderr)
        sys.exit(1)


def run_command(cmd: str) -> str:
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return ""


def extract_flag_value(flag: str, tokens: List[str]) -> str:
    """Extract flag value from command tokens."""
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return ""


def trim(value: str) -> str:
    """Trim whitespace from string."""
    return value.strip()


def gpu_mem_to_gb(mem: str) -> str:
    """Convert GPU memory to GB format."""
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)$", mem.strip())
    if not match:
        return mem

    value = float(match.group(1))
    unit = match.group(2).lower()

    gb_value = None
    if unit in ("mib", "mib"):
        gb_value = value / 1024
    elif unit in ("gib", "gib"):
        gb_value = value
    elif unit in ("mb", "mb"):
        gb_value = value / 1000
    elif unit in ("gb", "gb"):
        gb_value = value
    else:
        return mem

    return f"{gb_value:.2f} GB"


def format_model_name(value: str) -> str:
    """Format model name by replacing dash-number to slash-number."""
    return re.sub(r"-([0-9]+)", r"/\1", value)


def extract_cond_weight_suffix(cmdline: str) -> str:
    """Extract conditional weight from command line and convert to suffix (e.g. lll)."""
    match = re.search(r"--cond_weight\s+(\"?\{.*?\}\"?|\'?\{.*?\}\'?|\S+)", cmdline)
    if not match:
        return ""

    val = match.group(1)

    def get_val(key):
        m = re.search(rf"['\"]{key}['\"]\s*:\s*['\"]([^'\"]+)['\"]", val)
        if m:
            v = m.group(1)
            if v == "fixed":
                return "f"
            elif v == "shared":
                return "s"
            elif v == "learnable":
                return "l"
            elif v in ("learnable_0", "zero_init"):
                return "z"
        return ""

    cw1 = get_val('cond')
    cw2 = get_val('low_cond')
    cw3 = get_val('high_cond')

    if cw1 and cw2 and cw3:
        return f"{cw1}{cw2}{cw3}"

    # Also support shorthand if they passed it correctly as -lll
    # But usually it's passed as full dict through python main_jit.py
    # If the user passed directly e.g. --cond_weight lll
    if re.match(r"^[fslz]{3}$", val):
        return val

    return ""


def build_details(cmd_tokens: List[str]) -> str:
    """Build details string from command tokens."""
    detail_tokens = []
    skip_next = False

    for i, token in enumerate(cmd_tokens):
        if skip_next:
            skip_next = False
            continue

        if token in ("--dataset", "--model", "--epochs", "--epoch", "--add_loss_name"):
            skip_next = True
            continue

        if any(token.startswith(f"{flag}=") for flag in ("--dataset", "--model", "--epochs", "--epoch", "--add_loss_name")):
            continue

        if i == 0 and "python" in token:
            continue

        if "main_jit.py" in token or "inference_jit.py" in token:
            continue

        detail_tokens.append(token)

    if not detail_tokens:
        return "-"

    pairs = []
    i = 0
    while i < len(detail_tokens):
        item = detail_tokens[i]

        if "=" in item and item.startswith("--"):
            key, val = item.split("=", 1)
            key = key.lstrip("-")
            pairs.append(f"{trim(key)}={trim(val)}")
            i += 1
            continue

        if item.startswith("--") and i + 1 < len(detail_tokens) and not detail_tokens[i + 1].startswith("--"):
            item_key = item.lstrip("-")
            pairs.append(f"{trim(item_key)}={trim(detail_tokens[i + 1])}")
            i += 2
            continue

        item_key = item.lstrip("-")
        pairs.append(f"{trim(item_key)}=-")
        i += 1

    pairs.sort()

    formatted_parts = []
    for count, pair in enumerate(pairs):
        key, val = pair.split("=", 1)
        formatted_parts.append(f"{key}: {val}")
        if (count + 1) % 5 == 0 and count + 1 < len(pairs):
            formatted_parts.append("<br>")

    formatted = ", ".join([p for p in formatted_parts if p != "<br>"])
    formatted = formatted.replace(", <br>", "<br>")
    return formatted


def max_checkpoint_step(run_dir: str) -> str:
    """Find maximum checkpoint step in a run directory."""
    checkpoint_dir = Path(run_dir.rstrip("/")) / "checkpoints"

    if not checkpoint_dir.exists():
        return NO_STEP_VALUE

    max_step = None
    for checkpoint in checkpoint_dir.glob("checkpoint-*.pth"):
        filename = checkpoint.name
        step_str = filename.replace("checkpoint-", "").replace(".pth", "")

        if step_str.isdigit():
            step = int(step_str)
            if max_step is None or step > max_step:
                max_step = step

    return str(max_step) if max_step is not None else NO_STEP_VALUE


def get_gpu_uuid_to_index() -> Dict[str, str]:
    """Get mapping of GPU UUID to GPU index."""
    gpu_uuid_to_index = {}
    output = run_command(
        "nvidia-smi --query-gpu=index,uuid --format=csv,noheader")

    for line in output.split("\n"):
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            idx = trim(parts[0])
            uuid = trim(parts[1])
            if uuid:
                gpu_uuid_to_index[uuid] = idx

    return gpu_uuid_to_index


def get_running_gpu_processes() -> List[Tuple[str, str, str, str]]:
    """Get list of GPU processes."""
    output = run_command(
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader")
    processes = []

    for line in output.split("\n"):
        if not line:
            continue
        parts = [trim(p) for p in line.split(",")]
        if len(parts) >= 4:
            processes.append((parts[0], parts[1], parts[2], parts[3]))

    return processes


def get_process_cmdline(pid: str) -> str:
    """Get command line for a process."""
    try:
        return run_command(f"ps -p {pid} -o args=")
    except:
        return ""


def sort_run_dirs(dirs: List[str]) -> List[str]:
    """Sort run directories by prefix, size, and model."""
    sort_keys = []

    for dir_path in dirs:
        name = Path(dir_path).name

        # Determine prefix key
        if name.startswith("JiT_ParaCondWave"):
            prefix_key = "3"
        elif name.startswith("JiT_ParaCond"):
            prefix_key = "2"
        elif name.startswith("JiT_CondImg"):
            prefix_key = "1"
        else:
            prefix_key = "0"

        # Determine size key
        if "-16-" in name:
            size_key = "1"
        elif "-32-" in name:
            size_key = "2"
        else:
            size_key = "9"

        # Determine model key
        model_key = "9"
        if "JiT-B-" in name or "JiT_CondImg-B-" in name or "JiT_ParaCond-B-" in name or "JiT_ParaCondWave-B-" in name:
            model_key = "1"
        elif "JiT-L-" in name or "JiT_CondImg-L-" in name or "JiT_ParaCond-L-" in name or "JiT_ParaCondWave-L-" in name:
            model_key = "2"
        elif "JiT-H-" in name or "JiT_CondImg-H-" in name or "JiT_ParaCond-H-" in name or "JiT_ParaCondWave-H-" in name:
            model_key = "3"

        sort_key = f"{prefix_key}|{size_key}|{model_key}|{name}|{dir_path}"
        sort_keys.append(sort_key)

    sort_keys.sort()
    return [key.split("|")[4] for key in sort_keys]


def extract_model_dataset_and_variant(run_name: str, dataset_candidates: List[str]) -> Tuple[str, str, str]:
    """Extract model, dataset, and add-loss variant from a run directory name."""
    best_match: Optional[Tuple[int, int, str]] = None

    for dataset_name in dataset_candidates:
        marker = f"-{dataset_name}"
        position = run_name.rfind(marker)
        if position == -1:
            continue

        end_position = position + len(marker)
        if end_position < len(run_name) and run_name[end_position] != "-":
            continue

        candidate = (position, end_position, dataset_name)
        if best_match is None or candidate > best_match:
            best_match = candidate

    if best_match is not None:
        position, end_position, dataset_name = best_match
        model_name = run_name[:position]
        suffix = run_name[end_position:]
        variant = suffix.lstrip("-") if suffix.startswith("-") else "default"
        return model_name, dataset_name, variant

    if "-" in run_name:
        model_name, dataset_name = run_name.rsplit("-", 1)
        return model_name, dataset_name, "default"

    return run_name, "-", "default"


def main():
    output_file = sys.argv[1] if len(
        sys.argv) > 1 else "outputs/running_processes.md"

    check_nvidia_smi()

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get GPU info
    gpu_uuid_to_index = get_gpu_uuid_to_index()
    gpu_processes = get_running_gpu_processes()

    training_info = {}

    # Start building markdown
    lines = []
    lines.append("# Running JiT processes")
    lines.append("")
    lines.append("|GPU|PID|Dataset|Model|Process|GPU Memory|Epochs|Loss Name|Details|")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    row_count = 0

    for gpu_uuid, pid, proc_name, gpu_mem_raw in gpu_processes:
        gpu_index = gpu_uuid_to_index.get(gpu_uuid, "?")
        gpu_mem = gpu_mem_to_gb(gpu_mem_raw)

        cmdline = get_process_cmdline(pid)
        if not cmdline:
            continue

        if "main_jit.py" not in cmdline and "inference_jit.py" not in cmdline:
            continue

        if "main_jit.py" in cmdline:
            script_name = "main_jit.py"
        else:
            script_name = "inference_jit.py"

        proc_base = Path(proc_name).name
        process_label = f"{script_name}({proc_base})"

        cmd_tokens = cmdline.split()

        dataset = extract_flag_value("--dataset", cmd_tokens)
        if not dataset:
            data_path = extract_flag_value("--data_path", cmd_tokens)
            if data_path:
                dataset = Path(data_path).name

        model = extract_flag_value("--model", cmd_tokens)
        epochs = extract_flag_value("--epochs", cmd_tokens)
        if not epochs:
            epochs = extract_flag_value("--epoch", cmd_tokens)
        if not epochs:
            epochs = "-"

        add_loss_name = extract_flag_value("--add_loss_name", cmd_tokens)
        if not add_loss_name:
            add_loss_name = "-"

        details = build_details(cmd_tokens)

        dataset = dataset or "-"
        model = model or "-"
        display_model = format_model_name(model)
        model_key = model.replace("/", "-")

        cond_suffix = extract_cond_weight_suffix(cmdline)
        if cond_suffix:
            display_model = f"{display_model}-{cond_suffix}"
            model_key = f"{model_key}-{cond_suffix}"

        variant_key = "default" if add_loss_name == "-" else add_loss_name
        training_info[f"{dataset}|{model_key}|{variant_key}"] = f"GPU {gpu_index}|{pid}"

        lines.append(
            f"|{gpu_index}|{pid}|{dataset}|{display_model}|{process_label}|{gpu_mem}|{epochs}|{add_loss_name}|{details}|")
        row_count += 1

    if row_count == 0:
        lines.append("|-|-|-|-|-|-|-|-|No matching processes|")

    append_blank_line(lines)
    lines.append("## JiT checkpoint steps")
    append_blank_line(lines)

    # Find JiT directories
    outputs_dir = Path("outputs")
    jit_dirs = sorted(outputs_dir.glob("JiT*/")
                      ) if outputs_dir.exists() else []
    data_dir = Path("data")
    dataset_candidates = [d.name for d in data_dir.iterdir()
                          if d.is_dir()] if data_dir.exists() else []

    if not jit_dirs:
        lines.append("|-|-|N/A|")
    else:
        jit_sorted = sort_run_dirs([str(d) for d in jit_dirs])

        dataset_rows = {}
        dataset_seen = {}
        datasets = []

        for run_dir in jit_sorted:
            run_name = Path(run_dir).name

            model_name, dataset_name, variant_name = extract_model_dataset_and_variant(
                run_name, dataset_candidates)

            max_step = max_checkpoint_step(run_dir)
            model_key = model_name.replace("/", "-")
            training_info_val = training_info.get(
                f"{dataset_name}|{model_key}|{variant_name}", "-")

            training_gpu = "-"
            training_pid = "-"
            if training_info_val != "-":
                parts = training_info_val.split("|")
                training_gpu = parts[0]
                training_pid = parts[1] if len(parts) > 1 else "-"

            if dataset_name not in dataset_seen:
                dataset_seen[dataset_name] = True
                datasets.append(dataset_name)

            if dataset_name not in dataset_rows:
                dataset_rows[dataset_name] = []
            dataset_rows[dataset_name].append(
                (variant_name, model_name, max_step, training_gpu, training_pid))

        if datasets:
            for dataset_name in sorted(datasets):
                append_blank_line(lines)
                lines.append(f"### {dataset_name}")
                append_blank_line(lines)
                variant_groups = {}
                for variant_name, model_name, max_step, training_gpu, training_pid in dataset_rows[dataset_name]:
                    if variant_name not in variant_groups:
                        variant_groups[variant_name] = []
                    variant_groups[variant_name].append(
                        (model_name, max_step, training_gpu, training_pid))

                ordered_variants = []
                if "default" in variant_groups:
                    ordered_variants.append("default")
                if "dice_bce" in variant_groups:
                    ordered_variants.append("dice_bce")
                for variant_name in sorted(variant_groups.keys()):
                    if variant_name not in ordered_variants:
                        ordered_variants.append(variant_name)

                for variant_index, variant_name in enumerate(ordered_variants):
                    heading = "default" if variant_name == "default" else variant_name
                    lines.append(f"#### {heading} - {dataset_name}")
                    append_blank_line(lines)
                    lines.append("|Model|Max Step|Training|PID|")
                    lines.append("|---|---|---|---|")

                    for model_name, max_step, training_gpu, training_pid in variant_groups[variant_name]:
                        display_model = format_model_name(model_name)
                        lines.append(
                            f"|{display_model}|{max_step}|{training_gpu}|{training_pid}|")

                    if variant_index < len(ordered_variants) - 1:
                        append_blank_line(lines)

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Progress written to {output_file}")


if __name__ == "__main__":
    main()
