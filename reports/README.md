# APN Irregular Dataset Reports

## Reverse Lookup From Scripts

The reverse lookup chain in this workspace is:

`scripts/APN/<dataset>.sh`
-> `main.py`
-> `data/data_provider/data_factory.py`
-> `data/data_provider/datasets/<dataset>.py`
-> underlying dataset/task implementation
-> raw/preprocessed storage path

## Dataset Mapping

### 1. HumanActivity
- Script: `scripts/APN/HumanActivity.sh`
- APN dataset loader: `data/data_provider/datasets/HumanActivity.py`
- Raw/processing implementation: `data/dependencies/HumanActivity/HumanActivity.py`
- Storage used by APN: `storage/datasets/HumanActivity`
- Source: UCI wearable human activity dataset
- Meaning in APN:
  - 4 body locations: left ankle, right ankle, chest, belt
  - each location has 3 axes
  - total 12 channels
  - irregular sensor stream used for forecasting-style slicing

### 2. MIMIC_III
- Script: `scripts/APN/MIMIC_III.sh`
- APN dataset loader: `data/data_provider/datasets/MIMIC_III.py`
- TSDM task: `data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py`
- TSDM dataset: `data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py`
- Required raw file: `~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv`
- Meaning in APN:
  - ICU electronic health records
  - 96 clinical variables
  - DeBrouwer2019 preprocessing variant
  - APN uses first observation window to predict next measurements

### 3. P12
- Script: `scripts/APN/P12.sh`
- APN dataset loader: `data/data_provider/datasets/P12.py`
- TSDM task: `data/dependencies/tsdm/tasks/P12.py`
- TSDM dataset: `data/dependencies/tsdm/datasets/physionet2012.py`
- Raw cache path: `~/.tsdm/rawdata/Physionet2012/`
- Processed cache path: `~/.tsdm/datasets/Physionet2012/`
- Meaning in APN:
  - ICU 48-hour monitoring records
  - 36 temporal variables used in APN path
  - includes vitals, labs, blood gas, ventilation-related features

### 4. USHCN
- Script: `scripts/APN/USHCN.sh`
- APN dataset loader: `data/data_provider/datasets/USHCN.py`
- TSDM task: `data/dependencies/tsdm/tasks/ushcn_debrouwer2019.py`
- TSDM dataset: `data/dependencies/tsdm/datasets/ushcn_debrouwer2019.py`
- Raw cache path: `~/.tsdm/rawdata/USHCN_DeBrouwer2019/`
- Processed cache path: `~/.tsdm/datasets/USHCN_DeBrouwer2019/`
- Meaning in APN:
  - US climate station data
  - 5 climate variables
  - sparse/station-level time series over roughly 4 years
  - forecasting next observations from earlier sparse observations

## Generated Reports

- `reports/HumanActivity_eda_report.ipynb`
- `reports/MIMIC_III_eda_report.ipynb`
- `reports/P12_eda_report.ipynb`
- `reports/USHCN_eda_report.ipynb`

## What Each Notebook Contains

Each notebook includes:
- dataset background and APN-specific interpretation
- unified loading code
- sample-level size statistics
- missingness / sparsity heatmap
- inter-arrival time distribution
- multi-sample trajectory visualization
- channel correlation heatmap
- summary prompts for interpreting the dataset

## Important Caveat

The notebooks were generated successfully, but they have not been executed yet.
Execution depends on whether the corresponding dataset files are already available locally or can be downloaded/prepared from the paths above.
