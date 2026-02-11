#!/usr/bin/env bash
set -euo pipefail

output_file="${1:-outputs/running_processes.md}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
	echo "nvidia-smi not found in PATH." >&2
	exit 1
fi

extract_flag_value() {
	local flag="$1"
	shift
	local tokens=("$@")

	for i in "${!tokens[@]}"; do
		local token="${tokens[$i]}"
		if [[ "$token" == "$flag" ]]; then
			echo "${tokens[$((i + 1))]:-}"
			return
		fi
		if [[ "$token" == "$flag="* ]]; then
			echo "${token#*=}"
			return
		fi
	done
}

trim() {
	local value="$1"
	value="${value#${value%%[![:space:]]*}}"
	value="${value%${value##*[![:space:]]}}"
	echo "$value"
}

build_details() {
	local tokens=("$@")
	local details=()
	local skip_next=0

	for i in "${!tokens[@]}"; do
		local token="${tokens[$i]}"

		if (( skip_next )); then
			skip_next=0
			continue
		fi

		case "$token" in
			--dataset|--model|--epochs|--epoch)
				skip_next=1
				continue
				;;
			--dataset=*|--model=*|--epochs=*|--epoch=*)
				continue
				;;
		esac

		if [[ $i -eq 0 && "$token" == *python* ]]; then
			continue
		fi

		if [[ "$token" == *main_jit.py* || "$token" == *inference_jit.py* ]]; then
			continue
		fi

		details+=("$token")
	done

	if [[ ${#details[@]} -eq 0 ]]; then
		echo "-"
		return
	fi

	local pairs=()
	local i=0
	while [[ $i -lt ${#details[@]} ]]; do
		local item="${details[$i]}"
		if [[ "$item" == --*=* ]]; then
			local key="${item%%=*}"
			local val="${item#*=}"
			pairs+=("$(trim "$key")=$(trim "$val")")
			i=$((i + 1))
			continue
		fi

		if [[ "$item" == --* && $((i + 1)) -lt ${#details[@]} && "${details[$((i + 1))]}" != --* ]]; then
			pairs+=("$(trim "$item")=$(trim "${details[$((i + 1))]}")")
			i=$((i + 2))
			continue
		fi

		pairs+=("$(trim "$item")=-")
		i=$((i + 1))
	done

	local sorted
	sorted="$(printf "%s\n" "${pairs[@]}" | sort)"
	local bullets=""
	while IFS= read -r pair; do
		local key="${pair%%=*}"
		local val="${pair#*=}"
		if [[ -z "$bullets" ]]; then
			bullets="- ${key}: ${val}"
		else
			bullets+="<br>- ${key}: ${val}"
		fi
	done <<< "$sorted"

	echo "$bullets"
}

mkdir -p "$(dirname "$output_file")"

declare -A GPU_UUID_TO_INDEX
while IFS=, read -r idx uuid; do
	idx="$(trim "$idx")"
	uuid="$(trim "$uuid")"
	if [[ -n "$uuid" ]]; then
		GPU_UUID_TO_INDEX["$uuid"]="$idx"
	fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

mapfile -t gpu_lines < <(
	nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
)

tmp_file="${output_file}.tmp"
{
	echo "# Running JiT processes"
	echo
	echo "| GPU | PID | Process | GPU Memory | Dataset | Model | Epochs | Details |"
	echo "| --- | --- | --- | --- | --- | --- | --- | --- |"

	row_count=0

	for line in "${gpu_lines[@]:-}"; do
		gpu_uuid="$(echo "$line" | cut -d, -f1 | xargs)"
		pid="$(echo "$line" | cut -d, -f2 | xargs)"
		proc_name="$(echo "$line" | cut -d, -f3 | xargs)"
		gpu_mem="$(echo "$line" | cut -d, -f4 | xargs)"
		gpu_index="${GPU_UUID_TO_INDEX[$gpu_uuid]:-?}"

		cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
		if [[ -z "$cmdline" ]]; then
			continue
		fi

		if [[ "$cmdline" != *main_jit.py* && "$cmdline" != *inference_jit.py* ]]; then
			continue
		fi

		if [[ "$cmdline" == *main_jit.py* ]]; then
			script_name="main_jit.py"
		else
			script_name="inference_jit.py"
		fi

		proc_base="$(basename "$proc_name")"
		process_label="${script_name}(${proc_base})"

		read -r -a cmd_tokens <<< "$cmdline"

		dataset="$(extract_flag_value --dataset "${cmd_tokens[@]}")"
		if [[ -z "$dataset" ]]; then
			data_path="$(extract_flag_value --data_path "${cmd_tokens[@]}")"
			if [[ -n "$data_path" ]]; then
				dataset="$(basename "$data_path")"
			fi
		fi

		model="$(extract_flag_value --model "${cmd_tokens[@]}")"
		epochs="$(extract_flag_value --epochs "${cmd_tokens[@]}")"
		if [[ -z "$epochs" ]]; then
			epochs="$(extract_flag_value --epoch "${cmd_tokens[@]}")"
		fi
		if [[ -z "$epochs" ]]; then
			epochs="-"
		fi

		details="$(build_details "${cmd_tokens[@]}")"

		dataset="${dataset:-"-"}"
		model="${model:-"-"}"

		echo "| $gpu_index | $pid | $process_label | $gpu_mem | $dataset | $model | $epochs | $details |"
		row_count=$((row_count + 1))
	done

	if [[ $row_count -eq 0 ]]; then
		echo "| - | - | - | - | - | - | - | No matching processes |"
	fi
} > "$tmp_file"

mv "$tmp_file" "$output_file"
echo "Progress written to $output_file"