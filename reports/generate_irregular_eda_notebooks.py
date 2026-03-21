"""Generate four dataset-specific EDA notebooks for APN irregular time-series datasets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_notebook(dataset_name: str, intro: str, loader_code: str, channel_hint_code: str) -> dict:
    cells = [
        md_cell(
            f"# APN Dataset EDA Report: {dataset_name}\n\n"
            "本报告用于深入理解该数据集在 APN 项目中的形态、含义与时序特性。\n\n"
            "- 覆盖内容：数据来源、样本规模、不规则采样特性、缺失模式、变量相关性、典型时序轨迹\n"
            "- 注意：首次运行可能触发自动下载或预处理（取决于数据集）\n"
        ),
        md_cell(f"## 数据集背景\n\n{intro}\n"),
        code_cell(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if (PROJECT_ROOT / 'reports').exists() and str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "from reports.irregular_eda_utils import (\n"
            "    compute_dataset_summary,\n"
            "    get_sample_lengths,\n"
            "    plot_channel_correlation,\n"
            "    plot_channel_missingness,\n"
            "    plot_interarrival_distribution,\n"
            "    plot_multichannel_trajectories,\n"
            "    plot_sample_length_distributions,\n"
            "    to_long_frame,\n"
            ")\n"
            "\n"
            "sns.set_theme(style='whitegrid', context='talk')\n"
            "np.random.seed(0)\n"
        ),
        md_cell("## 1) 加载数据并统一结构\n\n将不同数据集统一为 records = [{t, x}] 形式：\n- t: 一维时间戳数组\n- x: 二维观测矩阵 [time, channel]\n"),
        code_cell(loader_code),
        code_cell(
            "summary = compute_dataset_summary(records, channel_names)\n"
            "lens = get_sample_lengths(records)\n"
            "summary\n"
        ),
        md_cell("## 2) 样本级统计\n\n观察每个样本时间点数量、总观测数量、时长分布。"),
        code_cell("plot_sample_length_distributions(lens, title_prefix=f'{dataset_name}: ')"),
        md_cell("## 3) 生成长表并查看总体观测规模"),
        code_cell(
            "n_jobs = min(8, os.cpu_count() or 1)\n"
            "long_df = to_long_frame(records, channel_names, n_jobs=n_jobs)\n"
            "print(f'Using n_jobs={n_jobs}')\n"
            "print('long_df shape:', long_df.shape)\n"
            "long_df.head()\n"
        ),
        md_cell("## 4) 缺失与稀疏模式（样本 x 通道）\n\n热图显示 log(1 + 观测次数)，越亮表示该通道在该样本中记录越密集。"),
        code_cell("plot_channel_missingness(long_df, channel_names, n_samples=200)"),
        md_cell("## 5) 采样间隔分布\n\n衡量不规则性：如果分布非常分散，说明采样时间并不均匀。"),
        code_cell("plot_interarrival_distribution(records, time_unit=time_unit)"),
        md_cell("## 6) 多变量时序轨迹（随机样本）\n\n显示若干关键通道的多样本轨迹叠加。"),
        code_cell(channel_hint_code),
        code_cell("plot_multichannel_trajectories(long_df, channels=channels_to_plot, n_samples=8)"),
        md_cell("## 7) 通道相关性\n\n基于“每个样本每个通道的均值”估计变量之间的相关结构。"),
        code_cell("plot_channel_correlation(long_df, channel_names, max_samples=300)"),
        md_cell(
            "## 8) 结论模板\n\n"
            "你可以根据上面的图与统计，从以下角度总结：\n"
            "1. 数据规模与稀疏程度：样本数、每样本观测数、时长分布\n"
            "2. 不规则时间特征：时间间隔是否长尾、是否存在明显采样节律\n"
            "3. 变量信息密度：哪些通道最稀疏、哪些最稳定\n"
            "4. 变量联动关系：相关热图是否出现明显块状结构\n"
            "5. 建模启发：是否适合 patch、插值、显式时间编码等策略\n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebooks = {}

    notebooks["HumanActivity_eda_report.ipynb"] = build_notebook(
        dataset_name="HumanActivity",
        intro=(
            "HumanActivity 来自 UCI 的可穿戴传感器活动识别数据。\n"
            "在 APN 中，数据被整理为 4 个身体位置传感器（每个 3 轴），共 12 通道。\n"
            "任务本质是对不规则采样的人体活动传感序列进行建模。"
        ),
        loader_code=(
            "from data.dependencies.HumanActivity.HumanActivity import HumanActivity\n"
            "from reports.irregular_eda_utils import records_from_human_activity\n"
            "\n"
            "dataset_name = 'HumanActivity'\n"
            "time_unit = 'quantized tick (dataset-specific)'\n"
            "root = PROJECT_ROOT / 'storage' / 'datasets' / 'HumanActivity'\n"
            "root.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "ha = HumanActivity(root=str(root), download=True)\n"
            "records = records_from_human_activity(ha)\n"
            "\n"
            "tag_map = ['ANKLE_LEFT', 'ANKLE_RIGHT', 'CHEST', 'BELT']\n"
            "axes = ['x', 'y', 'z']\n"
            "channel_names = [f'{tag}_{ax}' for tag in tag_map for ax in axes]\n"
            "print(f'Loaded {len(records)} records from HumanActivity.')\n"
        ),
        channel_hint_code=(
            "channels_to_plot = [\n"
            "    'ANKLE_LEFT_x',\n"
            "    'ANKLE_RIGHT_y',\n"
            "    'CHEST_z',\n"
            "    'BELT_x',\n"
            "]\n"
        ),
    )

    notebooks["MIMIC_III_eda_report.ipynb"] = build_notebook(
        dataset_name="MIMIC_III",
        intro=(
            "MIMIC-III 是 ICU 电子病历数据库。\n"
            "APN 使用的是 DeBrouwer2019 预处理版本（96 个连续变量，时间离散后再形成不规则观测）。\n"
            "注意：该版本需要你先具备受控访问并准备 complete_tensor.csv。"
        ),
        loader_code=(
            "from data.dependencies.tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019\n"
            "from reports.irregular_eda_utils import records_from_task_dataset\n"
            "\n"
            "dataset_name = 'MIMIC_III'\n"
            "time_unit = 'normalized time (0-1)'\n"
            "\n"
            "task = MIMIC_III_DeBrouwer2019(seq_len=72 - 0.5, pred_len=3, value_norm='none')\n"
            "all_ds = task.get_dataset((0, 'train'))\n"
            "records = records_from_task_dataset(all_ds)\n"
            "channel_names = [str(c) for c in task.dataset.columns.tolist()]\n"
            "print(f'Loaded {len(records)} samples from fold-0 train split.')\n"
            "print('Hint: MIMIC full report requires ~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv')\n"
        ),
        channel_hint_code=(
            "top_channels = long_df['channel'].value_counts().head(4).index.tolist()\n"
            "channels_to_plot = top_channels if len(top_channels) >= 2 else channel_names[:4]\n"
            "channels_to_plot\n"
        ),
    )

    notebooks["P12_eda_report.ipynb"] = build_notebook(
        dataset_name="P12",
        intro=(
            "P12 (PhysioNet Challenge 2012) 是 ICU 48 小时多变量监测数据。\n"
            "APN 使用 36 个时序变量（不含部分静态描述符），并进行不规则建模。\n"
            "其典型应用是短期临床状态预测与重建。"
        ),
        loader_code=(
            "from data.dependencies.tsdm.tasks.P12 import Physionet2012\n"
            "from reports.irregular_eda_utils import records_from_task_dataset\n"
            "\n"
            "dataset_name = 'P12'\n"
            "time_unit = 'normalized time (0-1)'\n"
            "\n"
            "task = Physionet2012(seq_len=36 - 0.5, pred_len=3, value_norm='none')\n"
            "all_ds = task.get_dataset((0, 'train'))\n"
            "records = records_from_task_dataset(all_ds)\n"
            "channel_names = [str(c) for c in task.dataset.columns.tolist()]\n"
            "print(f'Loaded {len(records)} samples from fold-0 train split.')\n"
        ),
        channel_hint_code=(
            "preferred = ['HR', 'MAP', 'RespRate', 'Temp']\n"
            "channels_to_plot = [c for c in preferred if c in channel_names]\n"
            "if len(channels_to_plot) < 2:\n"
            "    channels_to_plot = channel_names[:4]\n"
            "channels_to_plot\n"
        ),
    )

    notebooks["USHCN_eda_report.ipynb"] = build_notebook(
        dataset_name="USHCN",
        intro=(
            "USHCN 是美国历史气候网络的站点级时间序列。\n"
            "APN 当前路径采用 DeBrouwer2019 的稀疏化子集（5 通道，约 4 年窗口）。\n"
            "任务是基于前段稀疏观测预测后续观测。"
        ),
        loader_code=(
            "from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019\n"
            "from reports.irregular_eda_utils import records_from_task_dataset\n"
            "\n"
            "dataset_name = 'USHCN'\n"
            "time_unit = 'normalized time (0-1)'\n"
            "\n"
            "task = USHCN_DeBrouwer2019(seq_len=150 - 0.5, pred_len=3, value_norm='none')\n"
            "all_ds = task.get_dataset((0, 'train'))\n"
            "records = records_from_task_dataset(all_ds)\n"
            "channel_names = [str(c) for c in task.dataset.columns.tolist()]\n"
            "print(f'Loaded {len(records)} samples from fold-0 train split.')\n"
        ),
        channel_hint_code=(
            "channels_to_plot = channel_names[:4] if len(channel_names) >= 4 else channel_names\n"
            "channels_to_plot\n"
        ),
    )

    for fname, nb in notebooks.items():
        out_path = REPORT_DIR / fname
        out_path.write_text(json.dumps(nb, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
