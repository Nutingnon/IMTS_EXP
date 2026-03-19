#!/bin/bash

echo "Starting APN hyperparameter search script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

EXTRA_ARGS="$*"

GPU_IDS=(0)

if [[ -n "${EXTRA_ARGS}" ]]; then
    echo "Extra args: ${EXTRA_ARGS}"
fi

model_name="APN"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
features="M"
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5
RUN_MODE="${RUN_MODE:-smoke}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
    train_epochs="${TRAIN_EPOCHS:-1}"
    patience="${PATIENCE:-1}"
    itr="${ITR:-1}"
elif [[ "${RUN_MODE}" == "full" ]]; then
    train_epochs="${TRAIN_EPOCHS:-200}"
    patience="${PATIENCE:-10}"
    itr="${ITR:-5}"
else
    echo "Unsupported RUN_MODE='${RUN_MODE}'. Use 'smoke' or 'full'."
    exit 1
fi

echo "RUN_MODE=${RUN_MODE}, train_epochs=${train_epochs}, itr=${itr}"

dms=(6)
lrs=(0.01)
bss=("${BATCH_SIZE:-32}")
dps=(0.1)
ps=(100)
te_dims=(32)

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
        --itr \"$itr\" \
        --batch_size \"$bs\" \
        --learning_rate \"$lr\" \
        --d_model \"$dm\" \
        --dropout \"$dp\" \
        --apn_npatch \"$p\" \
        --apn_te_dim \"$te_dim\" \
        ${EXTRA_ARGS}" )

done; done; done; done; done; done

NUM_GPUS=${#GPU_IDS[@]}
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
    for i in "${!GPU_IDS[@]}"; do
        if [[ ${pids[$i]} -eq 0 ]] || ! ps -p ${pids[$i]} > /dev/null; then
            free_gpu_idx=$i
            break
        fi
    done

    if [ $free_gpu_idx -ne -1 ]; then
        gpu_id=${GPU_IDS[$free_gpu_idx]}
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
