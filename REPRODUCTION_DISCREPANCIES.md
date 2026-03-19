# t-PatchGNN Reproduction Analysis: Official vs. APN Implementation

This document outlines the critical discrepancies identified between the official `t-PatchGNN` implementation and the `APN` reproduction scripts. These differences, particularly **data normalization**, explain the significant performance gaps observed.

## 1. Executive Summary: The Normalization Gap

The most critical difference is **Data Normalization**.

*   **Official Implementation (`t-PatchGNN`)**: Explicitly calculates training set statistics (`data_min`, `data_max`) and applies Min-Max normalization to **both inputs and ground truth labels** in the `collate_fn` for PhysioNet, MIMIC-III, and Human Activity datasets. The reported MSE is calculated on **normalized values** (range [0, 1]).
*   **Reproduction (`APN`)**:
    *   **P12**: Uses **Standardization (Z-Score)** via `tsdm`, but critically, it computes statistics on the **entire dataset** (Train+Val+Test), leading to **Data Leakage**.
    *   **MIMIC-III & Human Activity**: Feeds **raw physical values** into the model without normalization.
    *   **USHCN**: Matches the official implementation (Raw Features, Normalized Time).

## 2. Dataset-Specific Discrepancies

### 2.1 PhysioNet 2012 (P12)

| Feature | Official Implementation | Reproduction (`P12.sh`) | Impact |
| :--- | :--- | :--- | :--- |
| **Normalization** | **MinMax [0,1]** (Train fit) | **Z-Score** (All-data fit + Leakage) | **Critical** (explains 0.005 vs 0.319) |
| **Input (Seq Len)** | `history=24` (Hours) | `seq_len=36` (Time Steps) | **High**. Input density differs. |
| **Patch Size** | `patch_size=8`, `stride=8` | `patch_len=6` | **Medium**. |

### 2.2 MIMIC-III

| Feature | Official Implementation | Reproduction (`MIMIC_III.sh`) | Impact |
| :--- | :--- | :--- | :--- |
| **Normalization** | **MinMax [0,1]** (Train fit) | **No** (Raw Data) | **Critical**. Huge scale difference. |
| **Input (Seq Len)** | `history=24` (Hours) | `seq_len=72` (Time Steps) | **High**. |
| **Patch Size** | `patch_size=8` | `patch_len=12` | **Medium**. |

### 2.3 Human Activity

| Feature | Official Implementation | Reproduction (`HumanActivity.sh`) | Impact |
| :--- | :--- | :--- | :--- |
| **Normalization** | **MinMax [0,1]** (Train fit) | **No** (Raw Data) | **Critical**. |
| **Input (Seq Len)** | `history=3000` (ms) | `seq_len=3000` | **Matched**. |
| **Patch Size** | `patch_size=300` | `patch_len=300` | **Matched**. |

### 2.4 USHCN

| Feature | Official Implementation | Reproduction (`USHCN.sh`) | Impact |
| :--- | :--- | :--- | :--- |
| **Normalization** | **No** (Raw Feat) | **No** (Raw Feat) | **Matched**. Comparison should be fair here. |
| **Input (Seq Len)** | `history=24` (Months) | `seq_len=150` | **Critical**. Huge mismatch in temporal field. |
| **Patch Size** | `patch_size=2` | `patch_len=10` | **High**. |

## 3. Recommended Actions for Alignment

To align the reproduction with the official results, the following modifications are required in `APN`:

### 3.1 Data Loading & Normalization
*   **P12, MIMIC-III, Human Activity**:
    *   Implement **Min-Max Normalization** (feature-wise).
    *   Compute Min/Max statistics **only on the Training set**.
    *   Apply this normalization to both **Inputs (x)** and **Targets (y)** in the dataloaders.
    *   Report Validation/Test Loss on these normalized values.
*   **USHCN**:
    *   Keep features unnormalized (current state is correct).

### 3.2 Task Alignment
*   Ensure `seq_len` and `patch_len` in bash scripts align with the physical time windows defined in the official paper (e.g., ensuring `seq_len` covers 24 hours for P12/MIMIC).
