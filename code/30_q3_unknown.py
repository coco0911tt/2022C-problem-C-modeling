from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from common import (
    COMPONENTS,
    CONFIG,
    MASTER_SEED,
    ROOT,
    compute_balances,
    hash_file,
    load_joblib,
    load_unknown_data,
    save_csv,
    save_figure,
    save_json,
)


LOG_PATH = ROOT / "logs" / "30_q3_unknown.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

INV_LABEL = {0: "高钾", 1: "铅钡"}


def apply_frozen_calibrator(model_pack: dict, probability: np.ndarray) -> np.ndarray:
    calibrator = model_pack.get("calibrator")
    if calibrator is None:
        return np.asarray(probability, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def unknown_features(unknown: pd.DataFrame, prep, tr) -> pd.DataFrame:
    rows = []
    comps = []
    for i, (_, row) in enumerate(unknown.iterrows()):
        z = tr.ilr[i]
        comp = prep.inverse_ilr(z).iloc[0]
        rows.append({
            "artifact_id": row["unknown_id"], "unknown_id": row["unknown_id"],
            "surface_weathering": row["surface_weathering"], "n_points": 1,
            "missing_count_mean": float(row[[f"miss_{c}" for c in COMPONENTS]].sum()),
        })
        comps.append(comp)
    closed = pd.DataFrame(comps).reset_index(drop=True)
    balances = compute_balances(closed).reset_index(drop=True)
    return pd.concat([pd.DataFrame(rows), balances], axis=1)


def transform_unknown(pack: dict, unknown: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    prep = pack["preprocessor"]
    group_mode = pack.get("group_mode", "surface_weathering")
    groups = pd.Series("global", index=unknown.index) if group_mode == "global" else unknown["surface_weathering"]
    tr = prep.transform(unknown[COMPONENTS], groups)
    return unknown_features(unknown, prep, tr), tr


def model_probability(model_pack: dict, features: pd.DataFrame) -> np.ndarray:
    cfg = model_pack["config"]
    X = features[cfg["features"]].to_numpy(dtype=float)
    if model_pack.get("scaler") is not None:
        X = model_pack["scaler"].transform(X)
    raw_probability = model_pack["model"].predict_proba(X)[:, 1]
    return apply_frozen_calibrator(model_pack, raw_probability)


def bootstrap_probability(pack: dict, unknown: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    prep = pack["preprocessor"]
    tr = prep.transform(unknown[COMPONENTS], unknown["surface_weathering"])
    features = unknown_features(unknown, prep, tr)
    cfg = pack["config"]
    X = features[cfg["features"]].to_numpy(dtype=float)
    if pack.get("scaler") is not None:
        X = pack["scaler"].transform(X)
    raw_probability = pack["model"].predict_proba(X)[:, 1]
    p = apply_frozen_calibrator(pack, raw_probability)
    return p, tr


def main() -> None:
    model_dir = ROOT / "models" / "q2_1_frozen_pipeline"
    with (model_dir / "manifest.json").open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    file_checks = {}
    for name, expected in manifest["files"].items():
        path = model_dir / name
        actual = hash_file(path)
        file_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    if not all(v["match"] for v in file_checks.values()):
        raise RuntimeError("冻结管线哈希不匹配，停止问题3")

    full = load_joblib("models/q2_1_frozen_pipeline/final_pipeline.joblib")
    bootstrap_models = load_joblib("models/q2_1_frozen_pipeline/bootstrap_pipelines.joblib")
    sensitivity_pipelines = load_joblib("models/q2_1_frozen_pipeline/sensitivity_pipelines.joblib")
    first_read_utc = datetime.now(timezone.utc).isoformat()
    unknown = load_unknown_data()  # 冻结与哈希校验之后才首次读取表单3
    if unknown["unknown_id"].tolist() != [f"A{i}" for i in range(1, 9)]:
        raise ValueError(f"表单3编号不匹配：{unknown['unknown_id'].tolist()}")
    save_csv(unknown, "data/canonical_table3.csv")

    # 将未知样品有效性追加到统一审计
    validity_path = ROOT / "results" / "00_audit" / "sample_validity.csv"
    validity = pd.read_csv(validity_path, encoding="utf-8-sig")
    unknown_validity = pd.DataFrame({
        "raw_row_id": unknown["raw_row_id"], "artifact_id": unknown["unknown_id"],
        "sample_point": unknown["unknown_id"], "raw_component_sum_pct": unknown["raw_component_sum_pct"],
        "valid_sum_flag": unknown["valid_sum_flag"], "exclusion_reason": unknown["exclusion_reason"],
    })
    validity = pd.concat([validity[~validity["raw_row_id"].astype(str).str.startswith("表单3:")], unknown_validity], ignore_index=True)
    save_csv(validity, "results/00_audit/sample_validity.csv")

    features, main_transform = transform_unknown(full, unknown)
    model_probs = {}
    model_labels = {}
    for model_name, model_pack in full["models"].items():
        p = model_probability(model_pack, features)
        threshold = float(model_pack["config"]["threshold"])
        model_probs[model_name] = p
        model_labels[model_name] = (p >= threshold).astype(int)
    selected_model = full["selected_model"]
    main_p = model_probs[selected_model]
    threshold = float(full["models"][selected_model]["config"]["threshold"])

    boot_rows = []
    boot_matrix = []
    boot_failures = 0
    for iteration, pack in enumerate(bootstrap_models):
        try:
            p, _ = bootstrap_probability(pack, unknown)
            boot_matrix.append(p)
            for uid, prob in zip(unknown["unknown_id"], p):
                boot_rows.append({
                    "iteration": iteration, "unknown_id": uid, "model": selected_model,
                    "probability": float(prob), "label": INV_LABEL[int(prob >= threshold)],
                    "fit_success": 1, "failure_reason": "",
                })
        except Exception as exc:
            boot_failures += 1
            for uid in unknown["unknown_id"]:
                boot_rows.append({
                    "iteration": iteration, "unknown_id": uid, "model": selected_model,
                    "probability": np.nan, "label": "", "fit_success": 0, "failure_reason": str(exc),
                })
    boot_arr = np.asarray(boot_matrix)
    boot_df = pd.DataFrame(boot_rows)
    save_csv(boot_df, "results/03_q3/q3_bootstrap_predictions.csv")

    # 适用域：冻结选定特征空间
    train_features = full["artifact_features"].copy()
    selected_pack = full["models"][selected_model]
    selected_features = selected_pack["config"]["features"]
    X_train = train_features[selected_features].to_numpy(dtype=float)
    X_unknown = features[selected_features].to_numpy(dtype=float)
    if selected_pack.get("scaler") is not None:
        X_train = selected_pack["scaler"].transform(X_train)
        X_unknown = selected_pack["scaler"].transform(X_unknown)
    pair = pairwise_distances(X_train)
    np.fill_diagonal(pair, np.inf)
    loo_nearest = pair.min(axis=1)
    domain_threshold = float(np.quantile(loo_nearest, 0.95))
    dist_unknown = pairwise_distances(X_unknown, X_train)
    nearest_idx = dist_unknown.argmin(axis=1)
    nearest_dist = dist_unknown.min(axis=1)
    train_labels = train_features["true_label"].to_numpy(dtype=int)
    robust = MinCovDet(random_state=MASTER_SEED).fit(X_train)
    train_robust = np.sqrt(robust.mahalanobis(X_train))
    robust_threshold = float(np.quantile(train_robust, 0.95))
    unknown_robust = np.sqrt(robust.mahalanobis(X_unknown))

    applicability_rows = []
    final_rows = []
    for i, row in unknown.reset_index(drop=True).iterrows():
        main_label = int(main_p[i] >= threshold)
        same_class_mask = train_labels == main_label
        nearest_same = float(dist_unknown[i, same_class_mask].min()) if same_class_mask.any() else np.nan
        out_domain = bool(nearest_dist[i] > domain_threshold or unknown_robust[i] > robust_threshold)
        applicability_rows.append({
            "unknown_id": row["unknown_id"],
            "nearest_artifact_id": train_features.iloc[nearest_idx[i]]["artifact_id"],
            "nearest_distance": float(nearest_dist[i]),
            "nearest_same_predicted_class_distance": nearest_same,
            "robust_mahalanobis": float(unknown_robust[i]),
            "threshold": domain_threshold,
            "robust_threshold": robust_threshold,
            "flag": "out_of_domain" if out_domain else "in_domain",
        })
        if len(boot_arr) >= 800:
            values = boot_arr[:, i]
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            labels = values >= threshold
            majority_frequency = float(max(labels.mean(), 1 - labels.mean()))
        else:
            ci_low = ci_high = np.nan
            majority_frequency = np.nan
        reasons = []
        if not np.isfinite(majority_frequency) or majority_frequency < 0.80:
            reasons.append("bootstrap_majority_frequency_lt_0.80")
        if np.isfinite(ci_low) and ci_low <= threshold <= ci_high:
            reasons.append("probability_interval_crosses_threshold")
        labels_three = [int(model_labels[m][i]) for m in ["Logistic", "CART", "LDA"]]
        if len(set(labels_three)) > 1:
            reasons.append("three_models_disagree")
        if out_domain:
            reasons.append("out_of_applicability_domain")
        if bool(main_transform.fallback_used.iloc[i].any()):
            # 训练内小组回退是已冻结规则，只有未知组才视为拒识；本附件两个风化水平均已见
            group = str(row["surface_weathering"])
            if group not in full["preprocessor"].group_medians_:
                reasons.append("unseen_group_global_fallback")
        fuzzy_low = max(0.0, threshold - CONFIG["fuzzy_half_width"])
        fuzzy_high = min(1.0, threshold + CONFIG["fuzzy_half_width"])
        if fuzzy_low <= main_p[i] <= fuzzy_high:
            reasons.append("probability_in_fuzzy_zone")
        if not bool(row["valid_sum_flag"]):
            reasons.append("invalid_raw_component_sum")
        final_rows.append({
            "unknown_id": row["unknown_id"], "weathering": row["surface_weathering"],
            "main_model": selected_model, "tendency_label": INV_LABEL[main_label],
            "probability_lead_barium": float(main_p[i]), "prob_ci_low": float(ci_low), "prob_ci_high": float(ci_high),
            "cart_label": INV_LABEL[int(model_labels["CART"][i])],
            "lda_label": INV_LABEL[int(model_labels["LDA"][i])],
            "logistic_label": INV_LABEL[int(model_labels["Logistic"][i])],
            "bootstrap_majority_frequency": majority_frequency,
            "domain_distance": float(nearest_dist[i]), "domain_threshold": domain_threshold,
            "robust_mahalanobis": float(unknown_robust[i]), "robust_threshold": robust_threshold,
            "out_of_domain": int(out_domain), "reject_flag": int(len(reasons) > 0),
            "reject_reasons": ";".join(reasons), "raw_component_sum_pct": float(row["raw_component_sum_pct"]),
        })

    applicability = pd.DataFrame(applicability_rows)
    final = pd.DataFrame(final_rows)
    save_csv(applicability, "results/03_q3/q3_applicability.csv")
    save_csv(final, "results/03_q3/q3_unknown_predictions.csv")

    # 冻结替代管线与输入扰动灵敏度
    sensitivity_rows = []
    for scenario, pack in sensitivity_pipelines.items():
        feats, _ = transform_unknown(pack, unknown)
        model_pack = pack["models"][selected_model]
        p = model_probability(model_pack, feats)
        t = float(model_pack["config"]["threshold"])
        for uid, prob, base in zip(unknown["unknown_id"], p, main_p):
            sensitivity_rows.append({
                "scenario": scenario, "unknown_id": uid, "probability": float(prob),
                "label": INV_LABEL[int(prob >= t)], "delta_probability": float(prob - base),
                "label_changed": int((prob >= t) != (base >= threshold)),
            })
    for rate in [0.05, 0.10]:
        for phase in [0, 1]:
            perturbed = unknown.copy()
            for ridx, idx in enumerate(perturbed.index):
                for cidx, comp in enumerate(COMPONENTS):
                    value = perturbed.at[idx, comp]
                    if pd.notna(value) and value > 0:
                        perturbed.at[idx, comp] = float(value) * (1 + (1 if (cidx + phase) % 2 == 0 else -1) * rate)
            feats, _ = transform_unknown(full, perturbed)
            p = model_probability(selected_pack, feats)
            for uid, prob, base in zip(unknown["unknown_id"], p, main_p):
                sensitivity_rows.append({
                    "scenario": f"componentwise_pm_{int(rate*100)}pct_phase{phase}",
                    "unknown_id": uid, "probability": float(prob), "label": INV_LABEL[int(prob >= threshold)],
                    "delta_probability": float(prob - base), "label_changed": int((prob >= threshold) != (base >= threshold)),
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_csv(sensitivity, "results/03_q3/q3_sensitivity.csv")

    frozen_time = datetime.fromisoformat(manifest["frozen_at_utc"])
    read_time = datetime.fromisoformat(first_read_utc)
    check = {
        "manifest_loaded": True, "file_hash_checks": file_checks,
        "all_hashes_match": all(v["match"] for v in file_checks.values()),
        "frozen_at_utc": manifest["frozen_at_utc"], "table3_first_read_utc": first_read_utc,
        "freeze_precedes_table3_read": frozen_time < read_time,
        "unknown_rows": len(unknown), "all_unknown_sums_valid": bool(unknown["valid_sum_flag"].all()),
        "bootstrap_models_loaded": len(bootstrap_models), "bootstrap_prediction_failures": boot_failures,
        "no_unknown_accuracy_field": True,
    }
    save_json(check, "results/03_q3/q3_frozen_manifest_check.json")

    # 图1：概率及区间
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(final))
    p = final["probability_lead_barium"].to_numpy()
    ax.errorbar(x, p, yerr=np.vstack([p - final["prob_ci_low"], final["prob_ci_high"] - p]), fmt="o", capsize=3, color="#4C78A8")
    ax.axhline(threshold, color="#E45756", ls="--", label="冻结阈值")
    ax.axhspan(max(0, threshold - 0.1), min(1, threshold + 0.1), color="#F2CF5B", alpha=0.25, label="模糊区")
    for i, rejected in enumerate(final["reject_flag"]):
        if rejected:
            ax.scatter(i, p[i], facecolors="none", edgecolors="#B00020", s=90, linewidths=1.5)
    ax.set_xticks(x, final["unknown_id"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("铅钡概率（95% Bootstrap区间）")
    ax.legend()
    save_figure(fig, "q3_unknown_probabilities_intervals.pdf", final)

    # 图2：Bootstrap标签频率
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    colors = final["tendency_label"].map({"高钾": "#4C78A8", "铅钡": "#E45756"})
    ax.bar(final["unknown_id"], final["bootstrap_majority_frequency"], color=colors)
    ax.axhline(0.8, color="0.4", ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Bootstrap多数类别频率")
    save_figure(fig, "q3_bootstrap_label_frequency.pdf", final)

    # 图3：冻结特征空间投影
    combined = np.vstack([X_train, X_unknown])
    coords = PCA(n_components=2).fit_transform(combined)
    projection = pd.DataFrame({
        "id": train_features["artifact_id"].tolist() + unknown["unknown_id"].tolist(),
        "pc1": coords[:, 0], "pc2": coords[:, 1],
        "kind": ["known"] * len(X_train) + ["unknown"] * len(X_unknown),
        "label": [INV_LABEL[int(x)] for x in train_labels] + final["tendency_label"].tolist(),
    })
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for kind, marker, size in [("known", "o", 26), ("unknown", "*", 120)]:
        sub = projection[projection["kind"].eq(kind)]
        color = sub["label"].map({"高钾": "#4C78A8", "铅钡": "#E45756"})
        ax.scatter(sub["pc1"], sub["pc2"], c=color, marker=marker, s=size, alpha=0.8, label="已知文物" if kind == "known" else "A1--A8")
        if kind == "unknown":
            for _, r in sub.iterrows():
                ax.text(r["pc1"], r["pc2"], r["id"], fontsize=8)
    ax.set_xlabel("冻结特征PCA第一轴")
    ax.set_ylabel("冻结特征PCA第二轴")
    ax.legend()
    save_figure(fig, "q3_applicability_projection.pdf", projection)

    # 图4：三模型一致性
    consistency = final[["unknown_id", "logistic_label", "cart_label", "lda_label"]].set_index("unknown_id")
    matrix = consistency.replace({"高钾": 0, "铅钡": 1}).T.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(final)), final["unknown_id"])
    ax.set_yticks(range(3), ["Logistic", "CART", "LDA"])
    for i in range(3):
        for j in range(len(final)):
            ax.text(j, i, "铅钡" if matrix[i, j] == 1 else "高钾", ha="center", va="center", fontsize=7)
    save_figure(fig, "q3_three_model_consistency.pdf", consistency.reset_index())

    logger.info("问题3完成：%s", final[["unknown_id", "tendency_label", "probability_lead_barium", "reject_flag"]].to_dict(orient="records"))


if __name__ == "__main__":
    main()
