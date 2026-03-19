#!/bin/bash

set -u

echo "Starting APN HumanActivity script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs

# Optional proxy for dataset download.
# You can override by setting PROXY_URL, default is user-provided local proxy.
DEFAULT_PROXY_URL="http://127.0.0.1:7890"
PROXY_URL="${PROXY_URL:-}"

# GPU strategy:
# 1) GPU_IDS="0 1" takes highest priority
# 2) GPU_ID=0 fallback
# 3) default to GPU 0
GPU_IDS_RAW="${GPU_IDS:-}"
GPU_ID_SINGLE="${GPU_ID:-0}"
if [[ -n "${GPU_IDS_RAW}" ]]; then
    GPU_IDS_CLEAN="${GPU_IDS_RAW//,/ }"
    read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CLEAN}"
else
    GPU_IDS_ARR=("${GPU_ID_SINGLE}")
fi

if (( ${#GPU_IDS_ARR[@]} == 0 )); then
    echo "No GPU specified. Please set GPU_ID or GPU_IDS."
    exit 1
fi

echo "Using GPU IDs: ${GPU_IDS_ARR[*]}"

# 模型和数据集的固定参数
model_name="APN"
dataset_root_path="${DATASET_ROOT_PATH:-storage/datasets/HumanActivity}"
dataset_name="HumanActivity"
features="M"
seq_len="${SEQ_LEN:-3000}"
pred_len="${PRED_LEN:-300}"
enc_in=12
dec_in=12
c_out=12

# Modes:
# RUN_MODE=smoke  -> fast train+test check
# RUN_MODE=full   -> original settings
RUN_MODE="${RUN_MODE:-smoke}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
    train_epochs="${TRAIN_EPOCHS:-1}"
    patience="${PATIENCE:-1}"
    itr="${ITR:-1}"
    num_workers="${NUM_WORKERS:-0}"
    dms=(56)
    lrs=(0.01)
    bss=("${BATCH_SIZE:-2}")
    dps=(0)
    ps=(300)
    te_dims=(8)
elif [[ "${RUN_MODE}" == "full" ]]; then
    train_epochs="${TRAIN_EPOCHS:-200}"
    patience="${PATIENCE:-10}"
    itr="${ITR:-5}"
    num_workers="${NUM_WORKERS:-10}"
    dms=(56)
    lrs=(0.01)
    bss=("${BATCH_SIZE:-16}")
    dps=(0)
    ps=(300)
    te_dims=(8)
else
    echo "Unsupported RUN_MODE='${RUN_MODE}'. Use 'smoke' or 'full'."
    exit 1
fi

echo "RUN_MODE=${RUN_MODE}, train_epochs=${train_epochs}, itr=${itr}, num_workers=${num_workers}"

DATA_PT="${dataset_root_path}/processed/data.pt"
if [[ ! -f "${DATA_PT}" ]]; then
    echo "HumanActivity processed data not found. Triggering auto download/preprocess..."
    python - <<PY
from data.dependencies.HumanActivity.HumanActivity import HumanActivity

root = "${dataset_root_path}"
ds = HumanActivity(root=root, download=True)
print(f"HumanActivity ready: root={root}, n_records={len(ds)}")
PY
    prep_status=$?
    if [[ ${prep_status} -ne 0 ]]; then
        retry_proxy="${PROXY_URL:-${DEFAULT_PROXY_URL}}"
        echo "Dataset preparation failed on direct network. Retrying with proxy ${retry_proxy} ..."
        export http_proxy="${retry_proxy}"
        export https_proxy="${retry_proxy}"
        export HTTP_PROXY="${retry_proxy}"
        export HTTPS_PROXY="${retry_proxy}"
        python - <<PY
from data.dependencies.HumanActivity.HumanActivity import HumanActivity

root = "${dataset_root_path}"
ds = HumanActivity(root=root, download=True)
print(f"HumanActivity ready with proxy: root={root}, n_records={len(ds)}")
PY
        prep_status=$?
        if [[ ${prep_status} -ne 0 ]]; then
            echo "Dataset preparation failed even with proxy."
            exit 1
        fi
    fi
fi

if [[ ! -f "${DATA_PT}" ]]; then
    echo "Dataset is still not ready: ${DATA_PT}"
    exit 1
fi

tasks=()
for dm in "${dms[@]}"; do
for lr in "${lrs[@]}"; do
for bs in "${bss[@]}"; do
for dp in "${dps[@]}"; do
for p in "${ps[@]}"; do
for te_dim in "${te_dims[@]}"; do

    model_id="${MODEL_ID_PREFIX:-${model_name}_${dataset_name}}_sl${seq_len}_pl${pred_len}_dm${dm}_lr${lr}_bs${bs}_dp${dp}_p${p}_tedim${te_dim}"

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
        --num_workers \"$num_workers\" \
        --use_multi_gpu 0 \
        --gpu_id 0" )

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
echo "All tasks have been launched. Waiting for final jobs to complete..."
echo "########################################################################"
wait

echo "All experiments finished."