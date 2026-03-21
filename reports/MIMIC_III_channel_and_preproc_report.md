# MIMIC-III 通道与预处理深度说明（基于本仓库实现）

## 1. 报告目标与证据范围

本报告回答两个核心问题：

1. `complete_tensor.csv` 的 96 个通道分别是什么、代表什么。
2. 该数据集在本仓库中是如何从原始 MIMIC-III 表处理得到的。

证据分层：

- 仓库事实（代码/文件直接可验证）：
  - [gru_ode_bayes/data_preproc/MIMIC/readme.md](gru_ode_bayes/data_preproc/MIMIC/readme.md)
  - [gru_ode_bayes/data_preproc/MIMIC/DataMerging.ipynb](gru_ode_bayes/data_preproc/MIMIC/DataMerging.ipynb)
  - [mimic_process_work/label_dict.csv](mimic_process_work/label_dict.csv)
  - [mimic_process_work/complete_tensor.csv](mimic_process_work/complete_tensor.csv#L1)
  - [APN/data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py](APN/data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py)
  - [APN/data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py](APN/data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py)
- 外部背景（医学语义补充）：
  - PhysioNet MIMIC-III v1.4 主页：https://physionet.org/content/mimiciii/1.4/
  - MIMIC 文档入口：https://mimic.mit.edu/docs/iii/
  - GRU-ODE-Bayes NeurIPS 2019 摘要页：https://proceedings.neurips.cc/paper_files/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html

---

## 2. 端到端预处理链路（Notebook 顺序）

`gru_ode_bayes` 的预处理说明写在 [gru_ode_bayes/data_preproc/MIMIC/readme.md](gru_ode_bayes/data_preproc/MIMIC/readme.md)：

1. Admissions.ipynb
   - 处理 `PATIENTS.csv`、`ADMISSIONS.csv`、`INPUTEVENTS_MV.csv`
   - 输出 `Admissions_processed.csv`、`INPUTS_processed.csv`
2. Outputs.ipynb
   - 处理 `OUTPUTEVENTS.csv`
   - 输出 `OUTPUTS_processed.csv`
3. LabEvents.ipynb
   - 处理 `LABEVENTS.csv`
   - 输出 `LAB_processed.csv`
4. Prescriptions.ipynb
   - 处理 `PRESCRIPTIONS.csv`
   - 输出 `PRESCRIPTIONS_processed.csv`
5. DataMerging.ipynb
   - 合并上述中间表
   - 输出核心张量与标签/协变量（包括 `complete_tensor.csv`）

---

## 3. DataMerging 的底层逻辑（关键机制）

### 3.1 时间离散化

在 [gru_ode_bayes/data_preproc/MIMIC/DataMerging.ipynb](gru_ode_bayes/data_preproc/MIMIC/DataMerging.ipynb) 中使用：

- `TIME = round(TIME_STAMP.total_seconds() * bin_k / (100*36))`
- notebook 同时尝试了 `bin_k=60` 和 `bin_k=10` 两个分支。

### 3.2 同一时间桶重复观测聚合（核心）

按 `Origin` 使用不同聚合方式：

- Lab：`mean`（均值）
- Inputs：`sum`（求和）
- Outputs：`sum`
- Prescriptions：`sum`

这在 DataMerging 代码中是显式写死的：Lab 用 `groupby(...).mean()`，其余来源用 `groupby(...).sum()`。

### 3.3 去重与低观测样本过滤

在合并后执行：

- 去重键：`[HADM_ID, LABEL_CODE, TIME]`
- 过滤规则：去掉每次住院（`HADM_ID`）中观测数 `< 50` 的样本

### 3.4 编码与导出

- `LABEL_CODE` 由 `LABEL` 映射生成，并导出为 `label_dict.csv`
- `HADM_ID` 映射为随机 `UNIQUE_ID`，导出 `UNIQUE_ID_dict.csv`
- 最终张量导出 `complete_tensor.csv`

### 3.5 归一化列生成

在导出前为每个 `LABEL_CODE` 计算并附加：

- `MEAN`
- `STD`
- `VALUENORM = (VALUENUM - MEAN) / STD`

---

## 4. 最终产物 complete_tensor.csv 的字段语义

根据 [mimic_process_work/complete_tensor.csv#L1](mimic_process_work/complete_tensor.csv#L1)，字段头为：

- 索引列（CSV 第一列，无业务语义）
- `UNIQUE_ID`：住院 ID（`HADM_ID`）映射后的整数 ID
- `LABEL_CODE`：通道编码（0-95）
- `TIME_STAMP`：离散时间戳
- `VALUENUM`：聚合后的原始数值
- `MEAN`：该通道全局均值
- `STD`：该通道全局标准差
- `VALUENORM`：标准化后的数值

---

## 5. 对当前 complete_tensor.csv 的实测核验（本地文件）

对 [mimic_process_work/complete_tensor.csv](mimic_process_work/complete_tensor.csv) 实测得到：

- 行数：`3,424,862`
- `UNIQUE_ID` 数：`22,413`
- `LABEL_CODE` 唯一值数：`96`
- `LABEL_CODE` 范围：`0..95`
- `TIME_STAMP` 最大值：`2879`

解释：`TIME_STAMP` 最大值接近 48 小时 * 60 分钟 = 2880，说明你当前这份产物更接近 `bin_k=60`（分钟级）分支，而不是 `bin_k=10` 分支。

这与 APN `tsdm` 数据集文件中的注释（`TIME_STAMP ≈ 10*total_hours`）存在版本差异，属于需要在实验记录中明确标注的点。

---

## 6. APN/tsdm 如何消费这个张量（与你当前 EDA相关）

### 6.1 数据集层

在 [APN/data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py](APN/data/dependencies/tsdm/datasets/mimic_iii_debrouwer2019.py)：

- 从 `complete_tensor.csv` 读取
- 默认使用 `VALUENORM`
- 通过 `TripletDecoder(value_name="VALUENORM", var_name="LABEL_CODE")` 把稀疏三元组转成宽表（96 通道）

### 6.2 任务层

在 [APN/data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py](APN/data/dependencies/tsdm/tasks/mimic_iii_debrouwer2019.py)：

- 任务目标：观测窗口后预测后续 3 步
- 默认参数中 `observation_time=72` 对应 36 小时（按 30 分钟步长设定）
- `value_norm` 支持：`legacy_global` / `none` / `train_only`
- 切分是 ID 级（`self.IDs`），`train_test_split(..., shuffle=False)`

---

## 7. 96 通道详细字典（LABEL_CODE -> 含义）

以下名称来自 [mimic_process_work/label_dict.csv](mimic_process_work/label_dict.csv)，中文释义为医学常见解释（外部语义层）；“来源类别”根据名称与处理分支推断（Lab / Input-Drug-Fluid / Output）。

| LABEL_CODE | 通道名 | 中文说明 | 可能来源类别 |
|---:|---|---|---|
| 0 | Alanine Aminotransferase (ALT) | 丙氨酸氨基转移酶 | Lab |
| 1 | Alkaline Phosphatase | 碱性磷酸酶 | Lab |
| 2 | Anion Gap | 阴离子间隙 | Lab |
| 3 | Asparate Aminotransferase (AST) | 天门冬氨酸氨基转移酶 | Lab |
| 4 | Bicarbonate | 碳酸氢根 | Lab |
| 5 | Bilirubin, Total | 总胆红素 | Lab |
| 6 | Chloride | 氯离子 | Lab |
| 7 | Creatinine | 肌酐 | Lab |
| 8 | Glucose | 葡萄糖 | Lab |
| 9 | Potassium | 钾离子 | Lab |
| 10 | Sodium | 钠离子 | Lab |
| 11 | Urea Nitrogen | 尿素氮 | Lab |
| 12 | Basophils | 嗜碱性粒细胞 | Lab |
| 13 | Eosinophils | 嗜酸性粒细胞 | Lab |
| 14 | Hematocrit | 红细胞压积 | Lab |
| 15 | Hemoglobin | 血红蛋白 | Lab |
| 16 | Lymphocytes | 淋巴细胞 | Lab |
| 17 | MCH | 平均红细胞血红蛋白量 | Lab |
| 18 | MCHC | 平均红细胞血红蛋白浓度 | Lab |
| 19 | MCV | 平均红细胞体积 | Lab |
| 20 | Monocytes | 单核细胞 | Lab |
| 21 | Neutrophils | 中性粒细胞 | Lab |
| 22 | Platelet Count | 血小板计数 | Lab |
| 23 | PT | 凝血酶原时间 | Lab |
| 24 | PTT | 部分凝血活酶时间 | Lab |
| 25 | RDW | 红细胞分布宽度 | Lab |
| 26 | Red Blood Cells | 红细胞计数 | Lab |
| 27 | White Blood Cells | 白细胞计数 | Lab |
| 28 | pH | 酸碱度 | Lab |
| 29 | Specific Gravity | 尿比重 | Lab |
| 30 | Lactate | 乳酸 | Lab |
| 31 | Calcium, Total | 总钙 | Lab |
| 32 | Magnesium | 镁离子 | Lab |
| 33 | Phosphate | 磷酸盐 | Lab |
| 34 | Base Excess | 碱剩余 | Lab/血气 |
| 35 | Calculated Total CO2 | 计算总二氧化碳 | Lab/血气 |
| 36 | pCO2 | 二氧化碳分压 | Lab/血气 |
| 37 | pO2 | 氧分压 | Lab/血气 |
| 38 | Sodium Chloride 0.9% Flush Drug | 0.9% 氯化钠冲管药物 | Input/Drug |
| 39 | D5W Drug | 5% 葡萄糖水（药/液体） | Input/Drug |
| 40 | Magnesium Sulfate Drug | 硫酸镁（药） | Input/Drug |
| 41 | Potassium Chloride Drug | 氯化钾（药） | Input/Drug |
| 42 | Potassium Chloride | 氯化钾 | Input/Drug |
| 43 | Magnesium Sulfate | 硫酸镁 | Input/Drug |
| 44 | Calcium Gluconate | 葡萄糖酸钙 | Input/Drug |
| 45 | Morphine Sulfate | 吗啡硫酸盐 | Input/Drug |
| 46 | PO Intake | 经口摄入量 | Input |
| 47 | Insulin - Regular | 普通胰岛素 | Input/Drug |
| 48 | Furosemide (Lasix) | 呋塞米（速尿） | Input/Drug |
| 49 | Insulin - Humalog | 赖脯胰岛素 | Input/Drug |
| 50 | OR Cell Saver Intake | 手术回收血输入 | Input |
| 51 | Insulin - Glargine | 甘精胰岛素 | Input/Drug |
| 52 | OR Crystalloid Intake | 手术晶体液输入 | Input |
| 53 | KCL (Bolus) | 氯化钾推注 | Input/Drug |
| 54 | Solution | 溶液输入（泛称） | Input |
| 55 | LR | 乳酸林格液 | Input |
| 56 | Magnesium Sulfate (Bolus) | 硫酸镁推注 | Input/Drug |
| 57 | Dextrose 5% | 5% 葡萄糖液 | Input |
| 58 | Sterile Water | 无菌水 | Input |
| 59 | Piggyback | 静滴加药（piggyback） | Input |
| 60 | Nitroglycerin | 硝酸甘油 | Input/Drug |
| 61 | Albumin | 白蛋白 | Input |
| 62 | Chest Tube #1 | 胸腔引流 1 | Output |
| 63 | Foley | 导尿管尿量 | Output |
| 64 | Metoprolol Tartrate Drug | 酒石酸美托洛尔（药） | Input/Drug |
| 65 | Bisacodyl Drug | 比沙可啶（药） | Input/Drug |
| 66 | Docusate Sodium Drug | 多库酯钠（药） | Input/Drug |
| 67 | Aspirin Drug | 阿司匹林（药） | Input/Drug |
| 68 | Phenylephrine | 去氧肾上腺素 | Input/Drug |
| 69 | Metoprolol | 美托洛尔 | Input/Drug |
| 70 | Gastric Meds | 胃管给药/胃内药物 | Input/Drug |
| 71 | Midazolam (Versed) | 咪达唑仑 | Input/Drug |
| 72 | GT Flush | 胃管冲洗量 | Input |
| 73 | Hydralazine | 肼屈嗪 | Input/Drug |
| 74 | Packed Red Blood Cells | 浓缩红细胞输入 | Input |
| 75 | D5 1/2NS | 5%葡萄糖+半盐水 | Input |
| 76 | TF Residual | 肠内喂养残余量 | Output |
| 77 | Void | 自主排尿量 | Output |
| 78 | OR EBL | 手术估计失血量 | Output |
| 79 | Albumin 5% | 5% 白蛋白 | Input |
| 80 | Lorazepam (Ativan) | 劳拉西泮 | Input/Drug |
| 81 | K Phos | 磷酸钾 | Input/Drug |
| 82 | Jackson Pratt #1 | JP 引流 1 | Output |
| 83 | Pre-Admission | 入院前来源量（上下文相关） | Input/Other |
| 84 | Pantoprazole Drug | 泮托拉唑（药） | Input/Drug |
| 85 | Humulin-R Insulin Drug | Humulin-R 胰岛素 | Input/Drug |
| 86 | Stool Out Stool | 粪便排出量 | Output |
| 87 | Ultrafiltrate Ultrafiltrate | 超滤液量 | Output |
| 88 | Chest Tube #2 | 胸腔引流 2 | Output |
| 89 | Heparin Sodium | 肝素钠 | Input/Drug |
| 90 | Norepinephrine | 去甲肾上腺素 | Input/Drug |
| 91 | Urine Out Incontinent | 失禁尿量 | Output |
| 92 | Ostomy (output) | 造口输出量 | Output |
| 93 | Fecal Bag | 粪袋输出量 | Output |
| 94 | Gastric Gastric Tube | 胃管引流/输出量 | Output |
| 95 | Condom Cath | 阴茎套导尿输出量 | Output |

说明：部分药物项在 MIMIC 原始体系中可能来自不同事件表或编码合并过程，因此“来源类别”在个别条目上可能存在轻微歧义，但不影响建模上“Lab 用均值聚合、其余来源多为求和聚合”的核心逻辑。

---

## 8. 预处理设计背后的统计与建模含义

1. 混合聚合策略（mean vs sum）是合理的
- Lab 多是“状态测量”，同桶均值更接近状态估计。
- 输入/输出/处方多是“量/剂量/流量”，同桶求和更符合守恒与剂量累计。

2. 低观测过滤（<50）提高训练稳定性，但会引入选择偏差
- 会偏向保留监测更密集患者，可能降低对稀疏病例泛化。

3. 全局标准化（每 LABEL_CODE 的 `MEAN/STD`）有泄漏风险
- 若均值方差在全量数据上计算，严格评估中可能产生轻微信息泄漏。
- APN task 中的 `value_norm='train_only'` 正是为避免这一点。

4. 时间尺度版本差异必须在实验记录里写明
- 你当前文件实测更像 `bin_k=60`。
- APN 注释里提到 `≈10*hours`，属于不同预处理版本痕迹。

---

## 9. 与当前 EDA 的直接衔接建议（本阶段仅建议，不实施）

基于 [APN/reports/MIMIC_III_eda_report.ipynb](APN/reports/MIMIC_III_eda_report.ipynb) 后续可优先增加：

1. 通道覆盖率与稀疏度排名（96 通道完整条形图）。
2. 按“Lab / Input-Drug / Output”分组的观测密度对比。
3. 每通道时间间隔分布（`Δt`）与尾部分析。
4. 观测窗口（36h）与目标窗口（后续 3 步）分离可视化。
5. `value_norm` 三模式（legacy_global / none / train_only）分布对比。

---

## 10. 结论

- 本仓库中 MIMIC-III 的建模输入来自稀疏三元组张量化后的 96 通道时间序列。
- 通道语义由 `label_dict.csv` 稳定定义，`LABEL_CODE` 为 0-95。
- 预处理核心是：时间离散、按来源聚合、去重、低观测过滤、按通道标准化。
- 你当前本地 `complete_tensor.csv` 的时间尺度实测与 APN 注释存在版本差异，后续实验应显式写入这一点，避免复现歧义。
