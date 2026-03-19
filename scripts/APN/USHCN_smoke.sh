#!/bin/bash

# Ensure the logs directory exists
mkdir -p ./logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

EXTRA_ARGS="$*"

echo "Starting APN smoke test script..."

# Use GPU 0 for smoke test
GPU_IDS=(0)

model_name="APN"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
features="M"
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5
# Reduce epochs for smoke test
train_epochs=1
patience=10

# Single configuration for smoke test
dms=(6)
lrs=(0.01)
bss=(32)
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

    model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_lr${lr}_bs${bs}_dp${dp}_p${p}_tedim${te_dim}_smoke"

    # Note: Reduced epoch to 1, itr to 1
    cmd="python main.py \
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
        --itr 1 \
        --batch_size \"$bs\" \
        --learning_rate \"$lr\" \
        --d_model \"$dm\" \
        --dropout \"$dp\" \
        --apn_npatch \"$p\" \
        --apn_te_dim \"$te_dim\" \
        ${EXTRA_ARGS}"
    
    tasks+=( "$cmd" )
done
done
done
done
done
done

# Run the task
echo "Running smoke test command..."
echo "${tasks[0]}"

export CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}
# Use eval to execute the command string properly
eval "${tasks[0]}"
