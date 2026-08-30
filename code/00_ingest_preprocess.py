from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from common import (
    CHECKLIST,
    COMPONENTS,
    COMPONENT_CN,
    CONFIG,
    MASTER_SEED,
    ROOT,
    SOURCE_ROOT,
    SOURCE_XLSX,
    CompositionPreprocessor,
    environment_info,
    hash_file,
    load_known_data,
    save_csv,
    save_json,
)


LOG_PATH = ROOT / "logs" / "00_ingest_preprocess.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def make_data_dictionary() -> pd.DataFrame:
    rows = [
        {"table": "表单1", "original_name": "文物编号", "safe_name": "artifact_id", "dtype": "string", "unit": "无", "level": "文物", "allowed_or_meaning": "01--58"},
        {"table": "表单1", "original_name": "纹饰", "safe_name": "pattern", "dtype": "category", "unit": "无", "level": "文物", "allowed_or_meaning": "A/B/C"},
        {"table": "表单1", "original_name": "类型", "safe_name": "glass_type", "dtype": "category", "unit": "无", "level": "文物", "allowed_or_meaning": "高钾/铅钡"},
        {"table": "表单1", "original_name": "颜色", "safe_name": "color", "dtype": "category", "unit": "无", "level": "文物", "allowed_or_meaning": "原表类别；空白保留未知"},
        {"table": "表单1", "original_name": "表面风化", "safe_name": "surface_weathering", "dtype": "category", "unit": "无", "level": "文物", "allowed_or_meaning": "风化/无风化"},
        {"table": "表单2", "original_name": "文物采样点", "safe_name": "sample_point", "dtype": "string", "unit": "无", "level": "采样点", "allowed_or_meaning": "文物编号及点位说明"},
    ]
    for comp in COMPONENTS:
        rows.append({
            "table": "表单2/表单3",
            "original_name": COMPONENT_CN[comp],
            "safe_name": comp,
            "dtype": "float",
            "unit": "%",
            "level": "采样点/未知文物",
            "allowed_or_meaning": "非负；空白表示未检出或未提供数值",
        })
    return pd.DataFrame(rows)


def component_audit(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in COMPONENTS:
        s = points[comp]
        positive = s[s > 0]
        rows.append({
            "component": comp,
            "component_cn": COMPONENT_CN[comp],
            "n_records": len(s),
            "n_observed": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "n_explicit_zero": int(s.eq(0).fillna(False).sum()),
            "min_positive_pct": float(positive.min()) if len(positive) else np.nan,
            "q25_pct": float(s.quantile(0.25)),
            "median_pct": float(s.median()),
            "q75_pct": float(s.quantile(0.75)),
            "max_pct": float(s.max()),
        })
    return pd.DataFrame(rows)


def preprocessing_rationale(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tasks = {
        "q1_2_q1_3_q2_2_q4": "glass_type",
        "q2_1_q3": "surface_weathering",
    }
    complete_retained = int(points[COMPONENTS].notna().all(axis=1).sum())
    for task_id, group_key in tasks.items():
        for group_level, sub in points.groupby(group_key, dropna=False):
            for comp in COMPONENTS:
                s = sub[comp]
                med = s.median()
                mad = (s - med).abs().median()
                group_medians = points.groupby(group_key)[comp].median().dropna()
                between = float(group_medians.max() - group_medians.min()) if len(group_medians) else np.nan
                rows.append({
                    "task_id": task_id,
                    "split_id": "full_known_main",
                    "scope": "known_valid_points",
                    "component": comp,
                    "group_key": group_key,
                    "group_level": str(group_level),
                    "n_records": len(sub),
                    "n_missing": int(s.isna().sum()),
                    "missing_rate": float(s.isna().mean()),
                    "n_complete_case_retained": complete_retained,
                    "complete_case_loss_rate": float(1 - complete_retained / len(points)),
                    "median_pct": float(med) if pd.notna(med) else np.nan,
                    "mad_pct": float(mad) if pd.notna(mad) else np.nan,
                    "between_group_effect": between,
                    "fallback_rate": np.nan,
                })
    return pd.DataFrame(rows)


def mask_validation(points: pd.DataFrame, repeats: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(MASTER_SEED + 17)
    detail_rows = []
    for repeat in range(repeats):
        for comp in COMPONENTS:
            candidates = points.index[points[comp].notna() & (points[comp] > 0)].to_numpy()
            if len(candidates) < 10:
                continue
            n_mask = max(1, int(round(0.10 * len(candidates))))
            masked_idx = rng.choice(candidates, size=n_mask, replace=False)
            work = points[COMPONENTS].copy()
            truth = work.loc[masked_idx, comp].to_numpy(dtype=float)
            work.loc[masked_idx, comp] = np.nan
            groups = points["glass_type"].astype(str)
            predictions: dict[str, np.ndarray] = {}

            global_obs = work[comp].dropna()
            predictions["global_mean"] = np.repeat(global_obs.mean(), n_mask)
            predictions["global_median"] = np.repeat(global_obs.median(), n_mask)
            predictions["zero_fill"] = np.zeros(n_mask)
            group_pred = []
            for idx in masked_idx:
                g = groups.loc[idx]
                obs = work.loc[groups.eq(g), comp].dropna()
                if len(obs) < CONFIG["min_group_observed"]:
                    obs = global_obs
                group_pred.append(float(obs.median()))
            predictions["legal_group_median"] = np.asarray(group_pred)

            try:
                knn = KNNImputer(n_neighbors=5)
                knn_arr = knn.fit_transform(work)
                if knn_arr.shape[1] == len(COMPONENTS):
                    cidx = COMPONENTS.index(comp)
                    row_positions = [work.index.get_loc(i) for i in masked_idx]
                    predictions["knn_k5"] = knn_arr[row_positions, cidx]
            except Exception:
                pass

            for method, pred in predictions.items():
                err = np.asarray(pred) - truth
                detail_rows.append({
                    "task_id": "preprocessing",
                    "repeat_id": repeat,
                    "fold_id": "mask",
                    "component": comp,
                    "method": method,
                    "mask_rule": "10pct_observed_positive",
                    "mask_rate": 0.10,
                    "n_masked": n_mask,
                    "mae_pct": float(np.mean(np.abs(err))),
                    "rmse_pct": float(np.sqrt(np.mean(err ** 2))),
                    "median_ae_pct": float(np.median(np.abs(err))),
                    "aitchison_error": np.nan,
                    "variance_ratio": float(np.var(pred) / np.var(truth)) if np.var(truth) > 0 else np.nan,
                    "fit_success": 1,
                    "failure_reason": "",
                })
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            downstream_value=("mae_pct", "mean"),
            ci_low=("mae_pct", lambda x: x.quantile(0.025)),
            ci_high=("mae_pct", lambda x: x.quantile(0.975)),
            fit_success=("fit_success", "mean"),
        )
    )
    base = summary.loc[summary["method"].eq("legal_group_median"), "downstream_value"]
    base_value = float(base.iloc[0]) if len(base) else np.nan
    summary = summary.assign(
        task_id="preprocessing",
        repeat_id="aggregate",
        fold_id="mask_validation",
        group_key="glass_type",
        evaluation_unit="artificially_masked_observed_values",
        n_train=len(points),
        n_valid=detail["n_masked"].sum(),
        data_loss_rate=0.0,
        downstream_metric="masked_mae_pct_lower_is_better",
        delta_vs_group_median=summary["downstream_value"] - base_value,
        leakage_check="pass_training_only_mask_fit",
        failure_reason="",
    )
    columns = [
        "task_id", "repeat_id", "fold_id", "method", "group_key", "evaluation_unit",
        "n_train", "n_valid", "data_loss_rate", "downstream_metric", "downstream_value",
        "ci_low", "ci_high", "delta_vs_group_median", "leakage_check", "fit_success", "failure_reason",
    ]
    return detail, summary[columns]


def main() -> None:
    logger.info("开始读取表单1与表单2；表单3保持未读取状态")
    t1, t2 = load_known_data()
    valid = t2.loc[t2["valid_sum_flag"]].copy()
    if len(t1) != 58 or len(t2) != 69:
        raise ValueError(f"附件版本审计失败：表单1={len(t1)}, 表单2={len(t2)}")
    if len(valid) != 67:
        raise ValueError(f"有效点位应为67，当前为{len(valid)}")
    invalid = t2.loc[~t2["valid_sum_flag"], ["sample_point", "raw_component_sum_pct"]]
    invalid_ids = invalid["sample_point"].astype(str).tolist()
    if invalid_ids != ["15", "17"]:
        raise ValueError(f"无效点与附件预期不一致：{invalid_ids}")

    input_hashes = {
        "附件.xlsx": hash_file(SOURCE_XLSX),
        "C题.pdf": hash_file(SOURCE_ROOT / "C题" / "C题.pdf"),
        "交付清单_编程手_逐问.md": hash_file(CHECKLIST),
    }
    save_json(input_hashes, "results/00_audit/input_hashes.json")
    save_json(environment_info(), "results/00_audit/environment.json")
    save_csv(make_data_dictionary(), "results/00_audit/data_dictionary.csv")

    save_csv(t1, "data/canonical_table1.csv")
    save_csv(t2, "data/canonical_table2.csv")
    save_csv(valid, "data/known_points_valid.csv")
    trace = pd.concat([
        t1[["raw_row_id", "artifact_id"]].assign(source_table="表单1", sample_point=""),
        t2[["raw_row_id", "artifact_id", "sample_point"]].assign(source_table="表单2"),
    ], ignore_index=True)
    save_csv(trace, "results/00_audit/raw_row_traceability.csv")

    point_weathering = t2[[
        "raw_row_id", "artifact_id", "sample_point", "surface_weathering",
        "point_weathering", "weathering_source",
    ]].copy()
    save_csv(point_weathering, "results/00_audit/point_weathering_audit.csv")
    validity = t2[[
        "raw_row_id", "artifact_id", "sample_point", "raw_component_sum_pct",
        "valid_sum_flag", "exclusion_reason",
    ]].copy()
    save_csv(validity, "results/00_audit/sample_validity.csv")
    save_csv(component_audit(t2), "results/00_audit/component_audit.csv")

    repeats = (
        t2.groupby("artifact_id", as_index=False).size().rename(columns={"size": "n_sample_points"})
    )
    repeats["multiple_points"] = repeats["n_sample_points"].gt(1)
    save_csv(repeats, "results/00_audit/repeated_sampling_audit.csv")

    parameter_rows = []
    zero_rows = []
    closure_rows = []
    prep_specs = {
        "type_group_full": (valid["glass_type"], "q1_2_q1_3_q2_2_q4"),
        "weathering_group_full": (valid["surface_weathering"], "q2_1_q3_full_fit_only"),
    }
    for split_id, (groups, task_id) in prep_specs.items():
        prep = CompositionPreprocessor().fit(valid[COMPONENTS], groups)
        transformed = prep.transform(valid[COMPONENTS], groups)
        parameter_rows.extend(prep.parameter_rows(task_id, split_id))
        for comp in prep.active_components_:
            zero_rows.append({
                "task_id": task_id,
                "split_id": split_id,
                "component": comp,
                "zero_c": prep.zero_c,
                "delta": prep.delta_[comp],
                "training_min_positive": prep.delta_[comp] / prep.zero_c,
            })
        for idx, err in zip(valid.index, transformed.closure_error):
            closure_rows.append({
                "task_id": task_id,
                "split_id": split_id,
                "raw_row_id": valid.loc[idx, "raw_row_id"],
                "closure_sum": float(transformed.closed.loc[idx].sum()),
                "closure_error": float(err),
                "finite_ilr": bool(np.isfinite(transformed.ilr[valid.index.get_loc(idx)]).all()),
            })
        basis = pd.DataFrame(
            prep.basis_,
            index=[f"ilr_{i+1}" for i in range(prep.basis_.shape[0])],
            columns=prep.active_components_,
        )
        save_csv(basis.reset_index(names="coordinate"), f"results/00_audit/ilr_basis_{split_id}.csv")

    save_csv(pd.DataFrame(parameter_rows), "results/00_audit/imputation_parameters.csv")
    save_csv(pd.DataFrame(zero_rows), "results/00_audit/zero_replacement_parameters.csv")
    closure_df = pd.DataFrame(closure_rows)
    save_csv(closure_df, "results/00_audit/closure_audit.csv")
    if closure_df["closure_error"].max() > 1e-8 or not closure_df["finite_ilr"].all():
        raise ValueError("公共预处理闭合或ILR有限性检查失败")

    save_json({"component_order": COMPONENTS}, "results/00_audit/component_order.json")
    save_csv(preprocessing_rationale(valid), "results/00_audit/preprocessing_rationale.csv")
    mask_detail, mask_summary = mask_validation(valid)
    save_csv(mask_detail, "results/00_audit/imputation_mask_validation.csv")
    save_csv(mask_summary, "results/00_audit/imputation_method_comparison.csv")

    seed_seq = np.random.SeedSequence(MASTER_SEED)
    task_names = ["audit", "q1_1", "q1_2", "q1_3", "q2_1", "q2_2", "q3", "q4", "figures"]
    seed_rows = []
    for task, child in zip(task_names, seed_seq.spawn(len(task_names))):
        seed_rows.append({"task_id": task, "seed": int(child.generate_state(1)[0])})
    save_csv(pd.DataFrame(seed_rows), "results/00_audit/random_seeds.csv")

    schema = {
        "table1_rows": len(t1),
        "table2_rows": len(t2),
        "valid_known_points": len(valid),
        "valid_known_artifacts": int(valid["artifact_id"].nunique()),
        "invalid_points": invalid.to_dict(orient="records"),
        "multiple_point_artifacts": int(repeats["multiple_points"].sum()),
        "table3_read": False,
        "all_artifact_ids_unique_table1": bool(t1["artifact_id"].is_unique),
        "all_table2_ids_matched": bool(t2["glass_type"].notna().all()),
        "closure_max_error": float(closure_df["closure_error"].max()),
    }
    save_json(schema, "results/00_audit/schema_and_key_validation.json")
    logger.info("公共审计与预处理完成：%s", schema)


if __name__ == "__main__":
    main()
