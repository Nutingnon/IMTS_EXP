#!/usr/bin/env python3
"""Static parity audit for APN vs official t-PatchGNN PhysioNet/P12 pipelines.

The script compares:
1) Entry script runtime arguments.
2) Feature space definitions (APN canonical columns vs official PhysioNet params).
3) Split strategy hints from source code.
4) Normalization strategy hints from source code.
5) Training/evaluation loop strategy hints.

This is a static scanner (no model training). It is designed to be fast and
dependency-light for reproducibility triage.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


def parse_shell_args(script_path: Path, python_entry: str) -> dict[str, str]:
    text = script_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    shell_vars: dict[str, str] = {}
    assign_pat = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for_pat = re.compile(r"^for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^;]+);\s*do\s*$")
    env_default_pat = re.compile(r'^"?\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]+)\}"?$')

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = assign_pat.match(line)
        mf = for_pat.match(line)
        if mf:
            loop_var = mf.group(1)
            loop_vals = [v for v in mf.group(2).strip().split() if v]
            if loop_vals:
                shell_vars[loop_var] = loop_vals[0].strip("\"'")
            continue
        if not m:
            continue
        key, rhs = m.group(1), m.group(2).strip()
        # drop trailing comments
        if " #" in rhs:
            rhs = rhs.split(" #", 1)[0].strip()
        if "$(" in rhs:
            # skip command substitution, not safe to evaluate statically
            continue
        rhs = rhs.strip("\"'")
        m_default = env_default_pat.match(rhs)
        if m_default:
            rhs = m_default.group(1).strip("\"'")
        shell_vars[key] = rhs

    capture = False
    cmd_parts: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if not capture and python_entry in line and (
            "python" in line or "$launch_command" in line or "accelerate" in line
        ):
            capture = True

        if capture:
            if line.endswith("\\"):
                cmd_parts.append(line[:-1].strip())
                continue
            cmd_parts.append(line)
            # stop at the first non-continued command
            break

    joined = " ".join(cmd_parts)
    # parse --key value pairs conservatively
    tokens = re.findall(r"--[A-Za-z0-9_]+|\"[^\"]*\"|'[^']*'|[^\s]+", joined)
    result: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1].strip("\"'")
                if val.startswith("$") and len(val) > 1:
                    var_name = val[1:]
                    val = shell_vars.get(var_name, val)
                result[key] = val
                i += 2
            else:
                result[key] = "<flag>"
                i += 1
        else:
            i += 1
    return result


def extract_list_literal(py_path: Path, var_name: str) -> list[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    return []


def extract_class_attr_list(py_path: Path, class_name: str, attr_name: str) -> list[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            return ast.literal_eval(item.value)
    return []


def grep_lines(path: Path, patterns: list[str]) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: dict[str, list[str]] = {p: [] for p in patterns}
    for idx, line in enumerate(lines, start=1):
        for p in patterns:
            if re.search(p, line):
                result[p].append(f"L{idx}: {line.strip()}")
    return result


def build_report(repo_root: Path) -> dict[str, Any]:
    apn_root = repo_root / "APN"
    off_root = repo_root / "t-PatchGNN"

    apn_entry = apn_root / "scripts" / "tPatchGNN" / "P12.sh"
    off_entry = off_root / "tPatchGNN" / "scripts" / "run_all.sh"

    apn_p12_task = apn_root / "data" / "dependencies" / "tsdm" / "tasks" / "P12.py"
    off_physio = off_root / "lib" / "physionet.py"
    off_parse = off_root / "lib" / "parse_datasets.py"
    apn_main = apn_root / "main.py"
    apn_exp = apn_root / "exp" / "exp_main.py"
    off_run = off_root / "tPatchGNN" / "run_models.py"

    apn_args = parse_shell_args(apn_entry, "main.py")
    off_args = parse_shell_args(off_entry, "run_models.py")

    apn_cols = extract_class_attr_list(apn_p12_task, "Physionet2012", "CANONICAL_COLUMNS")
    off_cols = extract_class_attr_list(off_physio, "PhysioNet", "params")

    only_apn = sorted(set(apn_cols) - set(off_cols))
    only_off = sorted(set(off_cols) - set(apn_cols))

    apn_split_hits = grep_lines(
        apn_p12_task,
        [r"train_test_split", r"shuffle=False", r"test_size", r"valid_size", r"RANDOM_STATE"],
    )
    off_split_hits = grep_lines(
        off_parse,
        [r"train_test_split", r"random_state\s*=\s*42", r"train_size\s*=\s*0\.8", r"train_size\s*=\s*0\.75", r"shuffle\s*=\s*True", r"shuffle\s*=\s*False"],
    )

    apn_norm_hits = grep_lines(
        apn_p12_task,
        [r"legacy_global", r"Standardizer", r"MinMaxScaler", r"self\.encoder\.fit", r"\(-5 <", r"normalize_time"],
    )
    off_norm_hits = grep_lines(
        off_parse,
        [r"get_data_min_max", r"normalize_masked_data", r"normalize_masked_tp"],
    )

    apn_train_hits = grep_lines(
        apn_exp,
        [r"DelayedStepDecayLR", r"EarlyStopping", r"vali\(", r"metric\(", r"test\("],
    )
    apn_main_hits = grep_lines(apn_main, [r"for i in range\(configs\.itr\)", r"exp\.train\(\)", r"exp\.test\(\)"])
    off_train_hits = grep_lines(
        off_run,
        [r"best_val_mse", r"if\(val_res\[\"mse\"\] < best_val_mse\)", r"test_res = evaluation", r"patience", r"optimizer = optim\.Adam"],
    )

    key_param_pairs = {
        "task_window": {
            "apn_seq_len": apn_args.get("seq_len"),
            "apn_pred_len": apn_args.get("pred_len"),
            "apn_patch_len": apn_args.get("patch_len"),
            "off_history": off_args.get("history"),
            "off_patch_size": off_args.get("patch_size"),
            "off_stride": off_args.get("stride"),
        },
        "optimization": {
            "apn_lr": apn_args.get("learning_rate"),
            "apn_batch_size": apn_args.get("batch_size"),
            "apn_patience": apn_args.get("patience"),
            "off_lr": off_args.get("lr"),
            "off_batch_size": off_args.get("batch_size"),
            "off_patience": off_args.get("patience"),
        },
        "model_core": {
            "apn_n_heads": apn_args.get("n_heads"),
            "off_nhead": off_args.get("nhead"),
            "off_tf_layer": off_args.get("tf_layer"),
            "off_nlayer": off_args.get("nlayer"),
            "off_hid_dim": off_args.get("hid_dim"),
            "off_te_dim": off_args.get("te_dim"),
            "off_node_dim": off_args.get("node_dim"),
        },
    }

    return {
        "entry_args": {"apn": apn_args, "official": off_args},
        "key_param_pairs": key_param_pairs,
        "features": {
            "apn_feature_count": len(apn_cols),
            "official_feature_count": len(off_cols),
            "apn_only": only_apn,
            "official_only": only_off,
        },
        "split_evidence": {"apn": apn_split_hits, "official": off_split_hits},
        "normalization_evidence": {"apn": apn_norm_hits, "official": off_norm_hits},
        "training_eval_evidence": {
            "apn_exp": apn_train_hits,
            "apn_main": apn_main_hits,
            "official": off_train_hits,
        },
    }


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    report = build_report(repo_root)

    out_dir = repo_root / "APN" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "p12_apn_vs_official_parity_audit.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote parity audit report to: {out_json}")
    print("Top-level keys:", ", ".join(report.keys()))


if __name__ == "__main__":
    main()
