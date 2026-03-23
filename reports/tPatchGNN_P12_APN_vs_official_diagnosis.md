# APN vs Official t-PatchGNN (PhysioNet/P12) Gap Diagnosis

This document analyzes why APN reproduction can be much worse than the official t-PatchGNN implementation on PhysioNet/P12.

Scope:
- Data parity
- Preprocessing and split parity
- Model parameter/config parity
- Training recipe parity
- Evaluation parity

Primary metric focus: MAE.

## 1) Data Side

### 1.1 Feature space is different (high impact)
- APN PhysioNet task keeps canonical 36 columns in PhysioNet task class.
  - Evidence: [APN/data/dependencies/tsdm/tasks/P12.py](APN/data/dependencies/tsdm/tasks/P12.py#L206)
- Official PhysioNet loader uses 41 parameters including static/semi-static variables such as Age, Gender, Height, ICUType.
  - Evidence: [t-PatchGNN/lib/physionet.py](t-PatchGNN/lib/physionet.py#L18)

Impact:
- Different input information content means this is not the same task instance even if both are named P12.
- If official training uses static variables while APN excludes them, MAE can degrade significantly.

### 1.2 Sample pool may differ (medium-high impact)
- APN dataset wrapper comments indicate 11981 samples.
  - Evidence: [APN/data/data_provider/datasets/P12.py](APN/data/data_provider/datasets/P12.py#L78)
- Official loader concatenates set-a/set-b/set-c to 12000 records.
  - Evidence: [t-PatchGNN/lib/physionet.py](t-PatchGNN/lib/physionet.py#L57)

Impact:
- Missing/dropped records change split composition and target distribution.

## 2) Preprocessing and Split

### 2.1 Split strategy is different (high impact)
- APN split logic in task layer uses fixed random state and no shuffle in split calls.
  - Evidence: [APN/data/dependencies/tsdm/tasks/P12.py](APN/data/dependencies/tsdm/tasks/P12.py#L347)
  - Evidence: [APN/data/dependencies/tsdm/tasks/P12.py](APN/data/dependencies/tsdm/tasks/P12.py#L354)
- Official split uses:
  - 80/20 seen-test with shuffle=True and random_state=42
  - 75/25 train-val on seen data with shuffle=False and random_state=42
  - Evidence: [t-PatchGNN/lib/parse_datasets.py](t-PatchGNN/lib/parse_datasets.py#L32)
  - Evidence: [t-PatchGNN/lib/parse_datasets.py](t-PatchGNN/lib/parse_datasets.py#L33)

Impact:
- Different test sets and distribution shifts directly change MAE.

### 2.2 Normalization strategy differs (high impact)
- APN config default uses legacy_global value normalization.
  - Evidence: [APN/configs/tPatchGNN/tPatchGNN/P12.yaml](APN/configs/tPatchGNN/tPatchGNN/P12.yaml#L45)
- APN legacy_global path applies Standardizer and clipping around (-5, 5) before returning dataset.
  - Evidence: [APN/data/dependencies/tsdm/tasks/P12.py](APN/data/dependencies/tsdm/tasks/P12.py#L322)
  - Evidence: [APN/data/dependencies/tsdm/tasks/P12.py](APN/data/dependencies/tsdm/tasks/P12.py#L325)
- Official computes min/max from seen_data and uses masked min-max normalization for data and time.
  - Evidence: [t-PatchGNN/lib/parse_datasets.py](t-PatchGNN/lib/parse_datasets.py#L45)
  - Evidence: [t-PatchGNN/lib/utils.py](t-PatchGNN/lib/utils.py#L314)
  - Evidence: [t-PatchGNN/lib/utils.py](t-PatchGNN/lib/utils.py#L331)

Impact:
- Different scaling and clipping alter the optimization landscape and MAE comparability.

### 2.3 Time window semantics differ (high impact)
- APN entry uses seq_len=36 and pred_len=3 with patch_len=6.
  - Evidence: [APN/scripts/tPatchGNN/P12.sh](APN/scripts/tPatchGNN/P12.sh#L11)
  - Evidence: [APN/scripts/tPatchGNN/P12.sh](APN/scripts/tPatchGNN/P12.sh#L24)
  - Evidence: [APN/scripts/tPatchGNN/P12.sh](APN/scripts/tPatchGNN/P12.sh#L25)
- Official run uses history=24 with patch_size=8 and stride=8.
  - Evidence: [t-PatchGNN/tPatchGNN/scripts/run_all.sh](t-PatchGNN/tPatchGNN/scripts/run_all.sh#L7)
  - Evidence: [t-PatchGNN/tPatchGNN/scripts/run_all.sh](t-PatchGNN/tPatchGNN/scripts/run_all.sh#L9)

Impact:
- Forecast horizon and observed window are not matched, so MAE is not directly comparable.

## 3) Model Side

### 3.1 Core architecture is adapted, not identical (medium-high impact)
- APN model keeps tPatchGNN core blocks but adds adaptation around inputs and task pipeline integration.
  - Evidence: [APN/models/tPatchGNN.py](APN/models/tPatchGNN.py#L236)
- APN fixes some dimensions internally (for example hid_dim=64, n_layer=1, tf_layer=1) while reading other values from APN config.
  - Evidence: [APN/models/tPatchGNN.py](APN/models/tPatchGNN.py#L24)
  - Evidence: [APN/models/tPatchGNN.py](APN/models/tPatchGNN.py#L32)
  - Evidence: [APN/models/tPatchGNN.py](APN/models/tPatchGNN.py#L36)
- Official model takes these from argparse and uses them directly in constructor.
  - Evidence: [t-PatchGNN/tPatchGNN/model/tPatchGNN.py](t-PatchGNN/tPatchGNN/model/tPatchGNN.py#L82)

Impact:
- Even with same naming, effective computation graph and tensor path can diverge.

## 4) Training Side

### 4.1 Learning-rate schedule differs (high impact)
- APN default scheduler is DelayedStepDecayLR.
  - Evidence: [APN/configs/tPatchGNN/tPatchGNN/P12.yaml](APN/configs/tPatchGNN/tPatchGNN/P12.yaml#L61)
  - Evidence: [APN/exp/exp_main.py](APN/exp/exp_main.py#L66)
- Official run loop uses Adam with constant learning rate and no scheduler in loop.
  - Evidence: [t-PatchGNN/tPatchGNN/run_models.py](t-PatchGNN/tPatchGNN/run_models.py#L115)

Impact:
- Different convergence trajectories and best-epoch location.

### 4.2 Train/test execution rhythm differs (high impact)
- APN main does multiple train iterations, then test after loop.
  - Evidence: [APN/main.py](APN/main.py#L50)
  - Evidence: [APN/main.py](APN/main.py#L61)
- Official loop evaluates validation each epoch and updates test result when validation mse improves.
  - Evidence: [t-PatchGNN/tPatchGNN/run_models.py](t-PatchGNN/tPatchGNN/run_models.py#L136)
  - Evidence: [t-PatchGNN/tPatchGNN/run_models.py](t-PatchGNN/tPatchGNN/run_models.py#L141)

Impact:
- The selected checkpoint and reported MAE represent different selection protocols.

### 4.3 DataLoader behavior has hidden defaults in APN (medium impact)
- APN data factory uses expression that effectively defaults train/val to shuffle=True and drop_last=True unless explicitly set.
  - Evidence: [APN/data/data_provider/data_factory.py](APN/data/data_provider/data_factory.py#L29)
  - Evidence: [APN/data/data_provider/data_factory.py](APN/data/data_provider/data_factory.py#L30)

Impact:
- If official side does not match exact batching behavior, reproducibility can drift.

## 5) Evaluation Side

### 5.1 Metric family differs in final report outputs (medium impact)
- APN metric function returns MAE and MSE only.
  - Evidence: [APN/utils/metrics.py](APN/utils/metrics.py#L46)
- Official run logs and tracks loss, mse, rmse, mae, mape.
  - Evidence: [t-PatchGNN/tPatchGNN/run_models.py](t-PatchGNN/tPatchGNN/run_models.py#L148)

Impact:
- If report extraction scripts choose different fields, comparison can be misleading.

### 5.2 Best-model criterion mismatch for MAE comparison (high impact)
- Official chooses best epoch by validation mse and reports test metrics at that epoch.
  - Evidence: [t-PatchGNN/tPatchGNN/run_models.py](t-PatchGNN/tPatchGNN/run_models.py#L141)
- APN early stopping and test pass are integrated differently in a separate framework.
  - Evidence: [APN/exp/exp_main.py](APN/exp/exp_main.py#L356)
  - Evidence: [APN/exp/exp_main.py](APN/exp/exp_main.py#L659)

Impact:
- If MAE is the primary KPI, selecting by MSE can still produce MAE mismatch due to non-identical rank ordering.

## 6) Ranked Root-Cause Hypothesis (for APN much worse)

1. Task definition mismatch (window and patch semantics): 36/3/6 vs 24/8/8.
2. Feature space mismatch: 36D APN vs 41D official.
3. Split mismatch: non-equivalent train/val/test partitions.
4. Normalization mismatch and potential leakage/distortion from legacy_global + clipping.
5. Training selection protocol mismatch (offline test vs online best-val checkpointing).
6. Secondary factors: scheduler differences, batch/drop_last defaults.

## 7) What to Verify Next (execution checklist)

1. Export both test record ID lists and compute overlap ratio.
2. Align to one common feature subset and rerun one seed.
3. Force identical split logic and seeds.
4. Force identical normalization pipeline.
5. Force identical checkpoint selection rule for MAE.

Only after 1-5 match should MAE be considered directly comparable.
