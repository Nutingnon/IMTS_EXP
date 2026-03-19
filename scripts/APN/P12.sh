#!/bin/bash

set -u

echo "Starting APN hyperparameter search script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

EXTRA_ARGS="$*"

# If GPU_IDS is not pre-set by user, auto-detect currently available GPUs.
# Availability heuristic: low utilization and low memory usage.
detect_available_gpus() {
    local available=()

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return
    fi

    while IFS=',' read -r idx util mem; do
        idx="$(echo "$idx" | xargs)"
        util="$(echo "$util" | xargs)"
        mem="$(echo "$mem" | xargs)"

        if [[ "$util" =~ ^[0-9]+$ ]] && [[ "$mem" =~ ^[0-9]+$ ]]; then
            if (( util <= 20 && mem <= 1024 )); then
                available+=("$idx")
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)

    if (( ${#available[@]} == 0 )); then
        # Fallback: choose GPU with smallest memory usage.
        local best_gpu
        best_gpu="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -n | head -n1 | cut -d',' -f1 | xargs)"
        if [[ -n "$best_gpu" ]]; then
            echo "$best_gpu"
            return
        fi
    fi

    echo "${available[*]}"
}

if [[ -n "${GPU_IDS:-}" ]]; then
    # Allow override, e.g. GPU_IDS="0 1" ./scripts/APN/P12.sh
    read -r -a GPU_IDS_ARR <<< "${GPU_IDS}"
else
    read -r -a GPU_IDS_ARR <<< "$(detect_available_gpus)"
fi

if (( ${#GPU_IDS_ARR[@]} == 0 )); then
    echo "No GPU detected. Exiting."
    exit 1
fi

echo "Using GPU IDs: ${GPU_IDS_ARR[*]}"
if [[ -n "${EXTRA_ARGS}" ]]; then
    echo "Extra args: ${EXTRA_ARGS}"
fi

model_name="APN"
dataset_root_path="storage/datasets/P12"
dataset_name="P12"
features="M"
seq_len=36
pred_len=3
enc_in=36
dec_in=36
c_out=36

# Profiles:
# - medium: quick stability verification before full run.
# - full: original paper-level setting from the previous script.
RUN_PROFILE="${RUN_PROFILE:-medium}"

# Original full config (kept for reference):
# train_epochs=200
# patience=10
# --itr 5
# dms=(24)
# lrs=(0.03)
# bss=(32)
# dps=(0.1)
# ps=(20)
# te_dims=(8)

if [[ "${RUN_PROFILE}" == "medium" ]]; then
    train_epochs=20
    patience=5
    itr=1
    dms=(24)
    lrs=(0.03)
    bss=(32)
    dps=(0.1)
    ps=(20)
    te_dims=(8)
elif [[ "${RUN_PROFILE}" == "full" ]]; then
    train_epochs=200
    patience=10
    itr=5
    dms=(24)
    lrs=(0.03)
    bss=(32)
    dps=(0.1)
    ps=(20)
    te_dims=(8)
else
    echo "Unsupported RUN_PROFILE='${RUN_PROFILE}'. Use 'medium' or 'full'."
    exit 1
fi

echo "RUN_PROFILE=${RUN_PROFILE}, train_epochs=${train_epochs}, itr=${itr}"

tasks=()
for dm in "${dms[@]}"; do
for lr in "${lrs[@]}"; do
for bs in "${bss[@]}"; do
for dp in "${dps[@]}"; do
for p in "${ps[@]}"; do
for te_dim in "${te_dims[@]}"; do

    model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_lr${lr}_bs${bs}_dp${dp}_p${p}_tedim${te_dim}"

    tasks+=( "python main.py \
        --is_training 1 \
        --model_id \"$model_id\" \
        --model_name \"$model_name\" \
        --dataset_root_path \"$dataset_root_path\" \
        --dataset_name \"$dataset_name\" \
        --features \"$features\" \
        --seq_len \"$seq_len\" \
        --pred_len \"$pred_len\" \
        --enc_in \"$enc_in\" \
        --dec_in \"$dec_in\" \
        --c_out \"$c_out\" \
        --loss \"MSE\" \
        --train_epochs \"$train_epochs\" \
        --patience \"$patience\" \
        --val_interval 1 \
        --itr "$itr" \
        --batch_size \"$bs\" \
        --learning_rate \"$lr\" \
        --d_model \"$dm\" \
        --dropout \"$dp\" \
        --apn_npatch \"$p\" \
        --apn_te_dim \"$te_dim\" \
        ${EXTRA_ARGS}" )

done; done; done; done; done; done

NUM_GPUS=${#GPU_IDS_ARR[@]}
total_tasks=${#tasks[@]}
task_idx=0

pids=()
for (( i=0; i<NUM_GPUS; i++ )); do
    pids+=([i]=0)
done

echo "Total GPUs available: ${NUM_GPUS}"
echo "Total tasks to run: ${total_tasks}"
echo "Starting task dispatcher..."

while [ $task_idx -lt $total_tasks ]; do
    free_gpu_idx=-1
    for i in "${!GPU_IDS_ARR[@]}"; do
        if [[ ${pids[$i]} -eq 0 ]] || ! ps -p ${pids[$i]} > /dev/null; then
            free_gpu_idx=$i
            break
        fi
    done

    if [ $free_gpu_idx -ne -1 ]; then
        gpu_id=${GPU_IDS_ARR[$free_gpu_idx]}
        command=${tasks[$task_idx]}

        echo "------------------------------------------------------------------------"
        echo "Assigning Task #${task_idx} to GPU #${gpu_id}..."
        echo "COMMAND: CUDA_VISIBLE_DEVICES=${gpu_id} ${command}"
        echo "------------------------------------------------------------------------"

        model_id_val=$(echo "$command" | grep -oP '(?<=--model_id ")[^"]*')
        (
            export CUDA_VISIBLE_DEVICES=${gpu_id}
            eval ${command} 2>&1 | tee "./logs/${model_id_val}.log"
        ) &

        pids[$free_gpu_idx]=$!
        task_idx=$((task_idx + 1))
    else
        wait -n
    fi
done

echo "########################################################################"
echo "All tasks have been launched. Waiting for the final jobs to complete..."
echo "########################################################################"
wait

echo "All experiments finished."