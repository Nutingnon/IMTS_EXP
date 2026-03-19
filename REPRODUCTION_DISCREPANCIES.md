
# APN Reproduction Status and Discrepancy Analysis

This document is the single source of truth for the current reproduction status of the APN project in this workspace.

It consolidates three earlier root-level notes:

- `PROGRESS_2026-03-11.md`: the record that the training and testing pipeline was successfully run end-to-end.
- `数据处理声明.md`: the summary of how P12, MIMIC-III, HumanActivity, and USHCN are loaded and collated in the current codebase.
- the previous version of this file: the gap analysis between the current APN codebase and the official `t-PatchGNN` baseline.

The goal here is not to restate the paper claims. The goal is to record what the current code actually does, why the observed metrics are far from the paper, and what should be changed next.

## 1. Current Bottom Line

### 1.1 What has been confirmed

- The APN training and testing pipeline is runnable in this workspace.
- A medium-profile P12 run completed successfully on 2026-03-11 with final metrics:
    - `MAE: 0.37021493911743164`
    - `MSE: 0.3146660625934601`
- Those metrics are much worse than the paper-level numbers one would expect if the task setup and normalization protocol exactly matched the official `t-PatchGNN` baseline.
- The main confirmed reproduction problem is not that the model fails to run. The problem is that the current code path is not aligned with the original evaluation protocol.

### 1.2 Why the gap is not surprising

The largest source of mismatch is data preprocessing and evaluation scale:

- `t-PatchGNN` reports results for several irregular datasets under a train-set-fitted Min-Max normalization protocol.
- The current APN code path does not follow that protocol consistently.
- For P12, the current code uses full-dataset standardization, which leaks validation and test statistics into preprocessing.
- For MIMIC-III, the current code consumes a preprocessed file whose normalized values are already based on full-dataset statistics.
- For HumanActivity, the current APN loader uses raw values rather than the same Min-Max target scale used by the official `t-PatchGNN` irregular-data path.

As a result, current APN numbers should not be interpreted as a faithful reproduction of the paper tables.

## 2. Investigation Timeline

The following timeline is reconstructed from the APN root markdown files using filesystem modification time (`mtime`). This workspace is on Linux, so creation time is not consistently available; `mtime` is the most reliable sequencing signal here.

1. `README.md`
     - Initial project description and official usage-facing documentation.
     - High-level and paper-facing in tone.
2. `PROGRESS_2026-03-11.md`
     - First successful record that the APN pipeline runs end-to-end locally.
     - Includes environment notes, missing dependencies that were installed, and the successful P12 medium-profile run result.
3. `数据处理声明.md`
     - Follow-up note focused on the actual data loading and collate logic used by the current code.
     - Useful for tracing where preprocessing behavior comes from.
4. `REPRODUCTION_DISCREPANCIES.md`
     - Latest analysis note focused on why the observed metrics differ so much from paper-level expectations.

This merged document keeps the useful parts of all three notes and removes the overlap.

## 3. What Has Already Been Verified Locally

### 3.1 Environment and execution state

- The project was run successfully in the current environment without forcibly downgrading existing packages.
- Environment snapshot recorded during the successful run:
    - Python `3.12.9`
    - Torch `2.9.1`
    - CUDA available
- Only missing dependencies were added at that time:
    - `ptflops==0.7.4`
    - `wandb==0.21.1`

### 3.2 Entrypoint and launcher behavior

- The actual training and testing entrypoint is `main.py`.
- `--is_training 1` executes training and then testing.
- `scripts/APN/P12.sh` was updated previously to support:
    - automatic GPU detection
    - optional manual GPU override
    - a `RUN_PROFILE=medium` path for shorter validation runs
    - the original `RUN_PROFILE=full` path for longer runs

### 3.3 Confirmed successful run

- Launcher used: `scripts/APN/P12.sh`
- Verified command path:
    - `RUN_PROFILE=medium ./scripts/APN/P12.sh`
- Observed behavior:
    - training completed
    - early stopping triggered
    - checkpoint load and testing completed
- Final reported metrics:
    - `MAE: 0.37021493911743164`
    - `MSE: 0.3146660625934601`

These numbers confirm the code path works. They do not confirm paper-level reproducibility.

## 4. Current Data Processing Behavior in Code

This section records what the code actually does today.

### 4.1 Common loader structure

- Dataset modules are loaded dynamically by `dataset_name` through `data/data_provider/data_factory.py`.
- Custom `collate_fn` functions are also dynamically bound from dataset-specific modules.
- Runtime behavior is controlled by CLI arguments in `utils/configs.py`; YAML files are maintained as references and are not the authoritative source at runtime.

### 4.2 P12

Current code path:

- Dataset wrapper: `data/data_provider/datasets/P12.py`
- TSDM task: `data/dependencies/tsdm/tasks/P12.py`

Important current behavior:

- The task uses:
    - `Standardizer()` for feature columns
    - `MinMaxScaler()` for the time index
- The encoder is fitted on the concatenation of all three PhysioNet partitions (`A`, `B`, `C`) before the train/val/test split is applied inside the task.
- The file itself documents the issue in code comments: standardization is performed over the full data slice, including test.

Practical implication:

- P12 currently uses full-dataset Z-score normalization for values and Min-Max normalization for time.
- This is a data leakage issue relative to a train-only preprocessing protocol.

### 4.3 MIMIC-III

Current code path:

- Dataset wrapper: `data/data_provider/datasets/MIMIC_III.py`
- TSDM dataset loader: `data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py`
- TSDM task: `data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py`

Important current behavior:

- The project expects a preprocessed `complete_tensor.csv` generated by the `gru_ode_bayes` MIMIC preprocessing pipeline.
- The APN dataset loader reads `VALUENORM` from that file rather than reconstructing train-only normalization itself.
- The accompanying metadata (`MEANS`, `STDVS`) is computed by grouping the full CSV by `LABEL_CODE`.
- The task optionally normalizes time by dividing by the maximum timestamp of the full dataset.

Practical implication:

- The current MIMIC path already bakes in full-dataset normalization before APN training even starts.
- This means the leakage issue is upstream of the APN model code and tied to the preprocessed source file.

### 4.4 HumanActivity

Current code path:

- Dataset wrapper: `data/data_provider/datasets/HumanActivity.py`
- Internal dependency: `data/dependencies/HumanActivity/HumanActivity.py`

Important current behavior:

- Data is split with `shuffle=False` into `seen/test = 90/10`, then `train/val = 90/10` on the seen subset.
- The resulting subsets are passed into `Activity_time_chunk`.
- The current APN loader path does not perform the same train-fitted Min-Max normalization protocol described in the official `t-PatchGNN` irregular-data implementation.

Practical implication:

- HumanActivity in the current APN path is not aligned with the official `t-PatchGNN` normalization/evaluation scale.

### 4.5 USHCN

Current code path:

- Dataset wrapper: `data/data_provider/datasets/USHCN.py`
- TSDM task: `data/dependencies/tsdm/tasks/ushcn_debrouwer2019.py`

Important current behavior:

- USHCN is the least affected by the feature-normalization mismatch discussed above.
- The more relevant issue here is that the launch scripts and task window settings are not cleanly aligned with the time horizon described in comparison baselines.

Practical implication:

- USHCN should not be used as evidence that all datasets are equally impacted by normalization leakage.
- It still has configuration-alignment risk, but the issue is different.

## 5. Confirmed Reproduction Risks

This section includes only issues that are directly supported by the current code.

### 5.1 Full-dataset value normalization on P12

Confirmed in `data/dependencies/tsdm/tasks/P12.py`.

- `FrameEncoder(Standardizer(), index_encoders={"Time": MinMaxScaler()})` is fitted on the concatenated PhysioNet data before split-specific datasets are consumed.
- This means validation and test statistics influence preprocessing.

Impact:

- This is classic data leakage.
- It makes current P12 metrics incomparable to a strict train-only preprocessing protocol.

### 5.2 Full-dataset normalization baked into MIMIC input file

Confirmed in `data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py`.

- The APN path loads `VALUENORM` from `complete_tensor.csv`.
- The dataset metadata is derived by grouping the full preprocessed CSV.

Impact:

- Even if the APN training loop itself were perfectly clean, the current source file already carries global statistics.

### 5.3 Full-dataset metadata usage for irregular-length padding limits

Confirmed in:

- `data/data_provider/datasets/P12.py`
- `data/data_provider/datasets/MIMIC_III.py`
- `data/data_provider/datasets/HumanActivity.py`

Current behavior:

- For non-validation dataset initialization, the wrappers concatenate train, val, and test (or the equivalent merged set) to compute `seq_len_max_irr`, `pred_len_max_irr`, and sometimes `patch_len_max_irr`.

Impact:

- This is weaker than target leakage, but it still allows the model configuration path to depend on held-out-set metadata.
- For strict reproducibility and isolation, these limits should ideally be determined from the training split only, or from fixed constants documented per dataset.

### 5.4 Script-level task mismatch

Confirmed by the current shell scripts under `scripts/`.

Examples:

- Many APN and baseline scripts use `seq_len=36` for P12.
- Many MIMIC scripts use `seq_len=72`.
- Many USHCN scripts use `seq_len=150`.

Impact:

- These settings may be internally consistent for the current repository.
- They are not automatically the same thing as the physical history windows described in the external baseline implementation or paper text.

## 6. Comparison Against the Official `t-PatchGNN` Baseline

The most important comparison points are the following.

| Dataset | Official `t-PatchGNN` expectation | Current APN code path | Risk level |
| :--- | :--- | :--- | :--- |
| P12 | Train-fitted Min-Max normalization for inputs and targets | Full-dataset Z-score standardization for values, Min-Max on time | Critical |
| MIMIC-III | Train-fitted Min-Max normalization for inputs and targets | Preprocessed `VALUENORM` derived from full-dataset stats | Critical |
| HumanActivity | Train-fitted Min-Max normalization for inputs and targets | Current APN path does not mirror that protocol | High |
| USHCN | Raw feature handling is broadly acceptable | Normalization issue less central; script/task alignment still differs | Medium |

This is enough to explain why a runnable APN experiment can still be far from the paper numbers.

## 7. What Should Be Considered Canonical Right Now

Until the code is changed, the following statements should be treated as canonical for this repository:

- The APN project is runnable locally.
- The current root cause of poor reproducibility is primarily preprocessing and protocol mismatch, not an inability to train the model.
- P12 currently has confirmed data leakage from full-dataset standardization.
- MIMIC-III currently depends on a preprocessed source file with global normalization already baked in.
- HumanActivity and USHCN still require careful protocol alignment before any fair comparison to external baselines.

## 8. Planned Normalization Fixes

The following improvements are not fully implemented yet. They are the intended next step.

### 8.1 Add an explicit normalization mode option

The project should expose a runtime option with at least two safe modes for irregular datasets:

1. `none`
     - disable value normalization entirely
     - useful for ablation and for preserving raw physical units
2. `train_only`
     - fit normalization statistics on the training split only
     - reuse those statistics for validation and test
     - this is the minimum requirement for removing the current leakage path

An optional third mode can be considered later if needed:

3. `legacy_global`
     - preserve current behavior for backward comparison only
     - should be clearly marked as non-causal and leakage-prone

### 8.2 Dataset-specific implementation notes

- P12:
    - move value normalization fitting to training IDs only
    - apply the fitted transform to val/test without refitting
- MIMIC-III:
    - either support a raw-input path and normalize inside APN using train-only statistics
    - or regenerate the preprocessed source file with train-only statistics per split
- HumanActivity:
    - add a clear normalization switch and document the evaluation scale
- USHCN:
    - keep feature normalization optional, but align sequence-history interpretation with the comparison target

### 8.3 Reporting rules after the fix

Once the new option exists, every experiment report should state explicitly:

- whether normalization was `none`, `train_only`, or another mode
- whether targets were normalized together with inputs
- whether the reported loss is on normalized scale or original scale

Without that metadata, cross-run metric comparison remains ambiguous.

## 9. Practical Next Steps

The recommended order of work is:

1. Keep this document and `README.md` aligned with the current code.
2. Add a normalization-mode option for irregular datasets.
3. Implement a train-only path for P12 first, because the leakage is explicit and easy to verify.
4. Redesign the MIMIC preprocessing path so the project is not forced to consume globally normalized `VALUENORM` values.
5. Re-run a minimal reproduction matrix and compare the new scale against both the previous APN run and the official `t-PatchGNN` protocol.

## 10. Summary

APN has been run successfully in this workspace, but the current code path is not yet a faithful reproduction of the paper protocol.

The major confirmed reason is preprocessing mismatch, especially full-dataset normalization and related leakage. The next meaningful engineering step is not another blind rerun. It is to expose safe normalization modes and move the irregular-data path to train-only statistics before evaluating APN again.
