from __future__ import annotations

import itertools
import json
import logging
import math
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import (
    BALANCE_LIBRARY,
    COMPONENTS,
    CONFIG,
    MASTER_SEED,
    ROOT,
    CompositionPreprocessor,
    KNNCompositionPreprocessor,
    compute_balances,
    dump_joblib,
    hash_file,
    load_known_data,
    save_csv,
    save_figure,
    save_json,
)


LOG_PATH = ROOT / "logs" / "20_q2_1_classification.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

BALANCE_NAMES = [x["name"] for x in BALANCE_LIBRARY]
LABEL_MAP = {"高钾": 0, "铅钡": 1}
INV_LABEL_MAP = {0: "高钾", 1: "铅钡"}


def make_preprocessor(kind: str, zero_c: float, fixed_active: list[str] | None = None):
    if kind == "knn":
        return KNNCompositionPreprocessor(n_neighbors=5, zero_c=zero_c, fixed_active=fixed_active)
    return CompositionPreprocessor(zero_c=zero_c, fixed_active=fixed_active)


def artifact_features(points: pd.DataFrame, prep, tr) -> pd.DataFrame:
    ilr_df = pd.DataFrame(tr.ilr, index=points.index)
    rows = []
    compositions = []
    for artifact_id, idx in points.groupby("artifact_id").groups.items():
        idx_list = list(idx)
        z = ilr_df.loc[idx_list].mean(axis=0).to_numpy()
        comp = prep.inverse_ilr(z).iloc[0]
        first = points.loc[idx_list[0]]
        rows.append({
            "artifact_id": artifact_id,
            "glass_type": first["glass_type"],
            "true_label": LABEL_MAP[first["glass_type"]],
            "surface_weathering": first["surface_weathering"],
            "n_points": len(idx_list),
            "missing_count_mean": float(points.loc[idx_list, [f"miss_{c}" for c in COMPONENTS]].sum(axis=1).mean()),
        })
        compositions.append(comp)
    result = pd.DataFrame(rows)
    closed = pd.DataFrame(compositions).reset_index(drop=True)
    balances = compute_balances(closed).reset_index(drop=True)
    return pd.concat([result.reset_index(drop=True), balances], axis=1)


def prepare_split(
    train_points: pd.DataFrame,
    valid_points: pd.DataFrame,
    zero_c: float = 0.5,
    group_mode: str = "surface_weathering",
    prep_kind: str = "group_median",
    fixed_active: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    prep = make_preprocessor(prep_kind, zero_c, fixed_active)
    if group_mode == "global":
        train_groups = pd.Series("global", index=train_points.index)
        valid_groups = pd.Series("global", index=valid_points.index)
    else:
        train_groups = train_points[group_mode]
        valid_groups = valid_points[group_mode]
    prep.fit(train_points[COMPONENTS], train_groups)
    train_tr = prep.transform(train_points[COMPONENTS], train_groups)
    valid_tr = prep.transform(valid_points[COMPONENTS], valid_groups)
    return artifact_features(train_points, prep, train_tr), artifact_features(valid_points, prep, valid_tr), prep


def fit_classifier(model_name: str, config: dict, train: pd.DataFrame, valid: pd.DataFrame):
    features = list(config["features"])
    X_train = train[features].to_numpy(dtype=float)
    X_valid = valid[features].to_numpy(dtype=float)
    y_train = train["true_label"].to_numpy(dtype=int)
    scaler = None
    if model_name in {"Logistic", "LDA"}:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
    if model_name == "Logistic":
        cw = None if config["class_weight"] == "none" else "balanced"
        model = LogisticRegression(
            C=float(config["C"]), solver="liblinear", class_weight=cw,
            max_iter=5000, random_state=MASTER_SEED,
        )
    elif model_name == "CART":
        cw = None if config["class_weight"] == "none" else "balanced"
        model = DecisionTreeClassifier(
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            ccp_alpha=float(config["ccp_alpha"]),
            class_weight=cw,
            random_state=MASTER_SEED,
        )
    elif model_name == "LDA":
        shrinkage = config["shrinkage"]
        if shrinkage != "auto":
            shrinkage = float(shrinkage)
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_valid)[:, 1]
    return model, scaler, probability


def inner_fold_data(points: pd.DataFrame, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    artifacts = points[["artifact_id", "glass_type"]].drop_duplicates("artifact_id").reset_index(drop=True)
    y = artifacts["glass_type"].map(LABEL_MAP).to_numpy()
    cv = StratifiedGroupKFold(n_splits=CONFIG["inner_cv_folds"], shuffle=True, random_state=seed)
    folds = []
    for train_idx, valid_idx in cv.split(artifacts, y, groups=artifacts["artifact_id"]):
        train_ids = set(artifacts.iloc[train_idx]["artifact_id"])
        valid_ids = set(artifacts.iloc[valid_idx]["artifact_id"])
        train_points = points[points["artifact_id"].isin(train_ids)].copy()
        valid_points = points[points["artifact_id"].isin(valid_ids)].copy()
        train_f, valid_f, _ = prepare_split(train_points, valid_points)
        folds.append((train_f, valid_f))
    return folds


def evaluate_config(model_name: str, config: dict, folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    scores = []
    for train, valid in folds:
        try:
            _, _, prob = fit_classifier(model_name, config, train, valid)
            pred = (prob >= 0.5).astype(int)
            scores.append(balanced_accuracy_score(valid["true_label"], pred))
        except Exception:
            return -np.inf
    return float(np.mean(scores)) if scores else -np.inf


def tune_threshold(model_name: str, config: dict, folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    ys, probs = [], []
    for train, valid in folds:
        try:
            _, _, prob = fit_classifier(model_name, config, train, valid)
            ys.extend(valid["true_label"].tolist())
            probs.extend(prob.tolist())
        except Exception:
            continue
    if not ys:
        return 0.5
    candidates = np.linspace(0.2, 0.8, 61)
    scores = [balanced_accuracy_score(ys, np.asarray(probs) >= t) for t in candidates]
    best = max(scores)
    valid_t = [t for t, s in zip(candidates, scores) if abs(s - best) < 1e-12]
    return float(min(valid_t, key=lambda t: abs(t - 0.5)))


def tune_models(folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, dict]:
    # Logistic：1--3个可解释余额、完整C网格和类别权重
    log_candidates = []
    for size in [1, 2, 3]:
        for features in itertools.combinations(BALANCE_NAMES, size):
            for C in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
                for cw in ["none", "balanced"]:
                    cfg = {"features": list(features), "C": C, "class_weight": cw}
                    log_candidates.append((evaluate_config("Logistic", cfg, folds), cfg))
    log_candidates.sort(key=lambda x: (-x[0], len(x[1]["features"]), abs(math.log10(x[1]["C"]))))
    best_log = log_candidates[0][1]

    cart_candidates = []
    for depth in [1, 2, 3, 4]:
        for leaf in [3, 4, 5, 6, 8]:
            for alpha in [0.0, 0.005, 0.01]:
                for cw in ["none", "balanced"]:
                    cfg = {"features": BALANCE_NAMES, "max_depth": depth, "min_samples_leaf": leaf, "ccp_alpha": alpha, "class_weight": cw}
                    cart_candidates.append((evaluate_config("CART", cfg, folds), cfg))
    cart_candidates.sort(key=lambda x: (-x[0], x[1]["max_depth"], -x[1]["min_samples_leaf"], x[1]["ccp_alpha"]))
    best_cart = cart_candidates[0][1]

    lda_candidates = []
    for shrinkage in ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]:
        cfg = {"features": BALANCE_NAMES, "shrinkage": shrinkage}
        lda_candidates.append((evaluate_config("LDA", cfg, folds), cfg))
    lda_candidates.sort(key=lambda x: -x[0])
    best_lda = lda_candidates[0][1]

    configs = {"Logistic": best_log, "CART": best_cart, "LDA": best_lda}
    for name, cfg in configs.items():
        cfg["threshold"] = tune_threshold(name, cfg, folds)
    return configs


def probability_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
    }
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(logits, y)
        out["calibration_intercept"] = float(cal.intercept_[0])
        out["calibration_slope"] = float(cal.coef_[0, 0])
    except Exception:
        out["calibration_intercept"] = np.nan
        out["calibration_slope"] = np.nan
    return out


def probability_logit(p: np.ndarray | list[float]) -> np.ndarray:
    probability = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def fit_sigmoid_calibrator(p: np.ndarray, y: np.ndarray):
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(probability_logit(p), y)


def apply_sigmoid_calibrator(calibrator, p: np.ndarray | list[float]) -> np.ndarray:
    if calibrator is None:
        return np.asarray(p, dtype=float)
    return calibrator.predict_proba(probability_logit(p))[:, 1]


def calibration_decisions(oof: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    """Fit on repeat 0 and decide on held-out repeats 1--19 only.

    Calibration is enabled only when the held-out Brier score improves by at
    least 0.01 and balanced accuracy drops by no more than 0.01.  A final
    sigmoid is then refit on all OOF predictions for the frozen pipeline.
    """
    decisions: dict[str, dict] = {}
    rows = []
    for model_name, model_oof in oof.groupby("model"):
        fit_part = model_oof[model_oof["repeat_id"].eq(0)]
        eval_part = model_oof[model_oof["repeat_id"].gt(0)]
        raw_threshold = float(model_oof["threshold"].median())
        try:
            decision_calibrator = fit_sigmoid_calibrator(
                fit_part["artifact_probability"].to_numpy(), fit_part["true_label"].to_numpy()
            )
            raw_eval = eval_part["artifact_probability"].to_numpy()
            cal_eval = apply_sigmoid_calibrator(decision_calibrator, raw_eval)
            calibrated_threshold = float(apply_sigmoid_calibrator(decision_calibrator, [raw_threshold])[0])
            raw_metrics = probability_metrics(eval_part["true_label"].to_numpy(), raw_eval, raw_threshold)
            cal_metrics = probability_metrics(eval_part["true_label"].to_numpy(), cal_eval, calibrated_threshold)
            brier_improvement = float(raw_metrics["brier"] - cal_metrics["brier"])
            ba_change = float(cal_metrics["balanced_accuracy"] - raw_metrics["balanced_accuracy"])
            enabled = bool(brier_improvement >= 0.01 and ba_change >= -0.01)
            final_calibrator = fit_sigmoid_calibrator(
                model_oof["artifact_probability"].to_numpy(), model_oof["true_label"].to_numpy()
            ) if enabled else None
            final_threshold = float(apply_sigmoid_calibrator(final_calibrator, [raw_threshold])[0]) if enabled else raw_threshold
            reason = "enabled_brier_gain_ge_0.01_and_ba_loss_le_0.01" if enabled else "disabled_gate_not_met"
        except Exception as exc:
            enabled = False
            final_calibrator = None
            final_threshold = raw_threshold
            raw_metrics = {"brier": np.nan, "balanced_accuracy": np.nan}
            cal_metrics = {"brier": np.nan, "balanced_accuracy": np.nan}
            brier_improvement = np.nan
            ba_change = np.nan
            reason = f"disabled_calibration_error:{type(exc).__name__}"
        decisions[model_name] = {
            "enabled": enabled,
            "calibrator": final_calibrator,
            "raw_threshold": raw_threshold,
            "threshold": final_threshold,
            "reason": reason,
        }
        rows.append({
            "model": model_name,
            "fit_repeat": 0,
            "evaluation_repeats": "1-19",
            "raw_brier": raw_metrics["brier"],
            "calibrated_brier": cal_metrics["brier"],
            "brier_improvement": brier_improvement,
            "raw_balanced_accuracy": raw_metrics["balanced_accuracy"],
            "calibrated_balanced_accuracy": cal_metrics["balanced_accuracy"],
            "balanced_accuracy_change": ba_change,
            "raw_threshold": raw_threshold,
            "frozen_threshold": final_threshold,
            "calibration_enabled": int(enabled),
            "decision_rule": "Brier improvement >= 0.01 and BA decrease <= 0.01",
            "reason": reason,
        })
    return decisions, pd.DataFrame(rows)


def mode_config(configs: list[dict]) -> dict:
    strings = [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in configs]
    return json.loads(Counter(strings).most_common(1)[0][0])


def train_full_pipeline(points: pd.DataFrame, model_configs: dict[str, dict], group_mode: str = "surface_weathering", zero_c: float = 0.5, prep_kind: str = "group_median") -> dict:
    prep = make_preprocessor(prep_kind, zero_c)
    groups = pd.Series("global", index=points.index) if group_mode == "global" else points[group_mode]
    prep.fit(points[COMPONENTS], groups)
    tr = prep.transform(points[COMPONENTS], groups)
    features = artifact_features(points, prep, tr)
    fitted = {}
    for model_name, cfg in model_configs.items():
        model, scaler, _ = fit_classifier(model_name, cfg, features, features)
        fitted[model_name] = {"model": model, "scaler": scaler, "config": cfg}
    return {"preprocessor": prep, "artifact_features": features, "models": fitted, "group_mode": group_mode, "prep_kind": prep_kind, "zero_c": zero_c}


def attach_calibration(pipeline: dict, decisions: dict[str, dict]) -> dict:
    for model_name, model_pack in pipeline["models"].items():
        decision = decisions[model_name]
        model_pack["calibrator"] = decision["calibrator"]
        model_pack["raw_threshold"] = decision["raw_threshold"]
        model_pack["calibration_enabled"] = decision["enabled"]
        model_pack["calibration_reason"] = decision["reason"]
        model_pack["config"] = dict(model_pack["config"])
        model_pack["config"]["threshold"] = decision["threshold"]
    return pipeline


def bootstrap_pipelines(
    points: pd.DataFrame,
    selected_model: str,
    selected_config: dict,
    fixed_active: list[str],
    rng: np.random.Generator,
    calibration_decision: dict,
) -> tuple[list[dict], pd.DataFrame]:
    ids = points["artifact_id"].unique()
    models = []
    failures = []
    attempts = 0
    while len(models) < CONFIG["bootstrap_B"] and attempts < CONFIG["bootstrap_Bmax"]:
        attempts += 1
        sampled = rng.choice(ids, size=len(ids), replace=True)
        chunks = []
        for draw, artifact_id in enumerate(sampled):
            chunk = points[points["artifact_id"].eq(artifact_id)].copy()
            chunk["original_artifact_id"] = chunk["artifact_id"]
            chunk["artifact_id"] = f"b{draw:03d}_{artifact_id}"
            chunks.append(chunk)
        boot = pd.concat(chunks, ignore_index=True)
        if boot["glass_type"].nunique() < 2:
            failures.append({"attempt": attempts, "reason": "single_class_bootstrap"})
            continue
        try:
            prep = CompositionPreprocessor(zero_c=0.5, fixed_active=fixed_active).fit(boot[COMPONENTS], boot["surface_weathering"])
            tr = prep.transform(boot[COMPONENTS], boot["surface_weathering"])
            feats = artifact_features(boot, prep, tr)
            model, scaler, _ = fit_classifier(selected_model, selected_config, feats, feats)
            frozen_config = dict(selected_config)
            frozen_config["threshold"] = calibration_decision["threshold"]
            models.append({
                "iteration": len(models), "preprocessor": prep, "model": model, "scaler": scaler,
                "config": frozen_config, "selected_model": selected_model,
                "calibrator": calibration_decision["calibrator"],
                "raw_threshold": calibration_decision["raw_threshold"],
                "calibration_enabled": calibration_decision["enabled"],
            })
        except Exception as exc:
            failures.append({"attempt": attempts, "reason": str(exc)})
    return models, pd.DataFrame(failures, columns=["attempt", "reason"])


def quick_sensitivity_cv(points: pd.DataFrame, selected_model: str, selected_config: dict, scenario: str, group_mode: str, zero_c: float, prep_kind: str) -> dict:
    artifacts = points[["artifact_id", "glass_type"]].drop_duplicates("artifact_id").reset_index(drop=True)
    y = artifacts["glass_type"].map(LABEL_MAP).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=MASTER_SEED + 991)
    ys, probs = [], []
    failures = 0
    for tr_idx, va_idx in cv.split(artifacts, y, groups=artifacts["artifact_id"]):
        tr_ids = set(artifacts.iloc[tr_idx]["artifact_id"])
        va_ids = set(artifacts.iloc[va_idx]["artifact_id"])
        trp = points[points["artifact_id"].isin(tr_ids)].copy()
        vap = points[points["artifact_id"].isin(va_ids)].copy()
        try:
            trf, vaf, _ = prepare_split(trp, vap, zero_c=zero_c, group_mode=group_mode, prep_kind=prep_kind)
            _, _, p = fit_classifier(selected_model, selected_config, trf, vaf)
            ys.extend(vaf["true_label"].tolist())
            probs.extend(p.tolist())
        except Exception:
            failures += 1
    if not ys:
        return {"scenario": scenario, "model": selected_model, "metric": "balanced_accuracy", "value": np.nan, "fit_success": 0, "failure_folds": failures}
    metric = probability_metrics(np.asarray(ys), np.asarray(probs), float(selected_config["threshold"]))
    return {"scenario": scenario, "model": selected_model, "metric": "balanced_accuracy", "value": metric["balanced_accuracy"], "brier": metric["brier"], "fit_success": 1, "failure_folds": failures}


def main() -> None:
    _, t2 = load_known_data()  # 此脚本不读取表单3
    points = t2[t2["valid_sum_flag"]].copy()
    artifacts = points[["artifact_id", "glass_type"]].drop_duplicates("artifact_id").reset_index(drop=True)
    if len(artifacts) != 56:
        raise ValueError(f"问题2.1有效文物应为56，当前{len(artifacts)}")
    y_art = artifacts["glass_type"].map(LABEL_MAP).to_numpy()
    oof_rows = []
    split_rows = []
    coefficient_rows = []
    config_records = []
    model_configs_by_name: dict[str, list[dict]] = {"Logistic": [], "CART": [], "LDA": []}
    rng = np.random.default_rng(MASTER_SEED + 210)

    for repeat in range(CONFIG["outer_cv_repeats"]):
        seed = MASTER_SEED + 2100 + repeat
        outer = StratifiedGroupKFold(n_splits=CONFIG["outer_cv_folds"], shuffle=True, random_state=seed)
        for outer_fold, (tr_idx, va_idx) in enumerate(outer.split(artifacts, y_art, groups=artifacts["artifact_id"])):
            train_ids = set(artifacts.iloc[tr_idx]["artifact_id"])
            valid_ids = set(artifacts.iloc[va_idx]["artifact_id"])
            if train_ids & valid_ids:
                raise RuntimeError("同一文物跨外层折")
            for artifact_id in artifacts["artifact_id"]:
                split_rows.append({
                    "repeat_id": repeat, "outer_fold": outer_fold, "artifact_id": artifact_id,
                    "glass_type": artifacts.loc[artifacts["artifact_id"].eq(artifact_id), "glass_type"].iloc[0],
                    "split_role": "train" if artifact_id in train_ids else "validation", "seed": seed,
                })
            train_points = points[points["artifact_id"].isin(train_ids)].copy()
            valid_points = points[points["artifact_id"].isin(valid_ids)].copy()
            inner = inner_fold_data(train_points, seed + outer_fold)
            configs = tune_models(inner)
            train_f, valid_f, _ = prepare_split(train_points, valid_points)
            for model_name, cfg in configs.items():
                model_configs_by_name[model_name].append(dict(cfg))
                config_records.append({
                    "repeat_id": repeat, "outer_fold": outer_fold, "model": model_name,
                    "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                })
                model, scaler, probability = fit_classifier(model_name, cfg, train_f, valid_f)
                threshold = float(cfg["threshold"])
                labels = (probability >= threshold).astype(int)
                for i, row in valid_f.reset_index(drop=True).iterrows():
                    oof_rows.append({
                        "repeat_id": repeat, "outer_fold": outer_fold, "artifact_id": row["artifact_id"],
                        "true_label": int(row["true_label"]), "model": model_name,
                        "point_probability_mean": float(probability[i]), "artifact_probability": float(probability[i]),
                        "predicted_label": int(labels[i]), "threshold": threshold, "n_points": int(row["n_points"]),
                    })
                if model_name == "Logistic":
                    coef = model.coef_.ravel()
                    for feature, value in zip(cfg["features"], coef):
                        coefficient_rows.append({
                            "repeat_id/final": f"r{repeat}_f{outer_fold}", "feature_or_balance": feature,
                            "numerator_components": "+".join(next(s["numerator"] for s in BALANCE_LIBRARY if s["name"] == feature)),
                            "denominator_components": "+".join(next(s["denominator"] for s in BALANCE_LIBRARY if s["name"] == feature)),
                            "coefficient": float(value), "odds_ratio_per_sd": float(np.exp(np.clip(value, -30, 30))), "selected": 1,
                        })
        logger.info("问题2.1嵌套CV完成重复 %d/%d", repeat + 1, CONFIG["outer_cv_repeats"])

    oof = pd.DataFrame(oof_rows)
    splits = pd.DataFrame(split_rows)
    configs_df = pd.DataFrame(config_records)
    save_csv(splits, "results/02_q2/q2_1_cv_splits.csv")
    save_csv(oof, "results/02_q2/q2_1_oof_predictions.csv")
    save_csv(configs_df, "results/02_q2/q2_1_selected_params.csv")

    metric_rows = []
    confusion_rows = []
    for (repeat, model_name), sub in oof.groupby(["repeat_id", "model"]):
        if sub["artifact_id"].nunique() != 56 or len(sub) != 56:
            raise RuntimeError(f"OOF覆盖失败：repeat={repeat}, model={model_name}")
        threshold = float(sub["threshold"].median())
        m = probability_metrics(sub["true_label"].to_numpy(), sub["artifact_probability"].to_numpy(), threshold)
        metric_rows.append({"repeat_id": repeat, "model": model_name, **m})
        cm = confusion_matrix(sub["true_label"], sub["predicted_label"], labels=[0, 1])
        for true_label in [0, 1]:
            for pred_label in [0, 1]:
                confusion_rows.append({
                    "model": model_name, "repeat_id/aggregate": repeat, "true_label": true_label,
                    "predicted_label": pred_label, "artifact_count": int(cm[true_label, pred_label]),
                })
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    save_csv(metrics, "results/02_q2/q2_1_model_metrics.csv")
    save_csv(confusion, "results/02_q2/q2_1_confusion_matrix.csv")

    calibration_by_model, calibration_table = calibration_decisions(oof)
    save_csv(calibration_table, "results/02_q2/q2_1_calibration_decision.csv")

    comparison_rows = []
    for a, b in itertools.combinations(["Logistic", "CART", "LDA"], 2):
        ma = metrics[metrics["model"].eq(a)].set_index("repeat_id")
        mb = metrics[metrics["model"].eq(b)].set_index("repeat_id")
        for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "brier"]:
            diff = ma[metric] - mb[metric]
            comparison_rows.append({
                "model_a": a, "model_b": b, "metric": metric,
                "paired_difference": float(diff.mean()),
                "ci_low": float(diff.quantile(0.025)), "ci_high": float(diff.quantile(0.975)),
                "selection_decision": "higher_better" if metric != "brier" else "lower_better",
                "reason": "same_artifacts_same_outer_splits_paired_across_20_repeats",
            })
    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, "results/02_q2/q2_1_model_comparison.csv")

    coefficients = pd.DataFrame(coefficient_rows)
    save_csv(coefficients, "results/02_q2/q2_1_coefficients.csv")
    stability = coefficients.groupby("feature_or_balance", as_index=False).agg(
        selection_count=("selected", "sum"),
        coefficient_sign_frequency=("coefficient", lambda x: max((x >= 0).mean(), (x <= 0).mean())),
        median_abs_coefficient=("coefficient", lambda x: np.median(np.abs(x))),
    )
    stability["selection_frequency"] = stability["selection_count"] / (CONFIG["outer_cv_repeats"] * CONFIG["outer_cv_folds"])
    save_csv(stability, "results/02_q2/q2_1_feature_stability.csv")

    # 一标准误与简约性规则选择最终模型
    summary = metrics.groupby("model", as_index=False).agg(
        mean_ba=("balanced_accuracy", "mean"), sd_ba=("balanced_accuracy", "std"), mean_brier=("brier", "mean")
    )
    best_row = summary.loc[summary["mean_ba"].idxmax()]
    one_se = float(best_row["sd_ba"] / math.sqrt(CONFIG["outer_cv_repeats"]))
    eligible = summary[(summary["mean_ba"] >= best_row["mean_ba"] - one_se) & (summary["mean_brier"] <= summary["mean_brier"].min() + 0.02)].copy()
    simplicity = {"CART": 0, "Logistic": 1, "LDA": 2}
    eligible["simplicity"] = eligible["model"].map(simplicity)
    selected_model = str(eligible.sort_values(["simplicity", "mean_brier"]).iloc[0]["model"])
    final_configs = {name: mode_config(cfgs) for name, cfgs in model_configs_by_name.items()}
    for name in final_configs:
        thresholds = [float(c["threshold"]) for c in model_configs_by_name[name]]
        final_configs[name]["threshold"] = float(np.median(thresholds))

    full_pipeline = attach_calibration(train_full_pipeline(points, final_configs), calibration_by_model)
    full_pipeline["selected_model"] = selected_model
    full_pipeline["label_map"] = LABEL_MAP
    full_pipeline["balance_library"] = BALANCE_LIBRARY
    full_pipeline["frozen_at"] = datetime.now(timezone.utc).isoformat()
    full_pipeline["selection_summary"] = summary
    full_pipeline["one_se"] = one_se

    pipeline_dir = ROOT / "models" / "q2_1_frozen_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = dump_joblib(full_pipeline, "models/q2_1_frozen_pipeline/final_pipeline.joblib")

    # 预注册替代管线，供问题3仅调用
    sensitivity_pipelines = {
        "zero_c_0.25": attach_calibration(train_full_pipeline(points, final_configs, zero_c=0.25), calibration_by_model),
        "zero_c_0.75": attach_calibration(train_full_pipeline(points, final_configs, zero_c=0.75), calibration_by_model),
        "global_median": attach_calibration(train_full_pipeline(points, final_configs, group_mode="global", zero_c=0.5), calibration_by_model),
        "knn_k5": attach_calibration(train_full_pipeline(points, final_configs, prep_kind="knn", zero_c=0.5), calibration_by_model),
    }
    sensitivity_pipelines_path = dump_joblib(sensitivity_pipelines, "models/q2_1_frozen_pipeline/sensitivity_pipelines.joblib")

    logger.info("开始问题2.1最终模型的1000次文物Bootstrap冻结")
    boot_models, boot_failures = bootstrap_pipelines(
        points, selected_model, final_configs[selected_model], full_pipeline["preprocessor"].active_components_, rng,
        calibration_by_model[selected_model],
    )
    boot_path = dump_joblib(boot_models, "models/q2_1_frozen_pipeline/bootstrap_pipelines.joblib")
    save_csv(boot_failures, "results/02_q2/q2_1_bootstrap_failures.csv")

    marker = pipeline_dir / "FROZEN_BEFORE_TABLE3.txt"
    marker.write_text(
        f"frozen_at_utc={full_pipeline['frozen_at']}\n表单3未被code/20_q2_1_classification.py读取。\n",
        encoding="utf-8",
    )
    manifest = {
        "frozen_at_utc": full_pipeline["frozen_at"],
        "selected_model": selected_model,
        "selection_rule": "one_standard_error_then_brier_and_simplicity",
        "one_se": one_se,
        "input_xlsx_sha256": hash_file(ROOT.parent / "C题" / "附件.xlsx"),
        "table3_read_before_freeze": False,
        "active_components": full_pipeline["preprocessor"].active_components_,
        "group_key": "surface_weathering",
        "zero_c": 0.5,
        "balance_library": BALANCE_LIBRARY,
        "final_configs": final_configs,
        "calibration": {
            name: {
                "enabled": bool(decision["enabled"]),
                "raw_threshold": float(decision["raw_threshold"]),
                "frozen_threshold": float(decision["threshold"]),
                "reason": decision["reason"],
            }
            for name, decision in calibration_by_model.items()
        },
        "bootstrap_success": len(boot_models),
        "bootstrap_failure": len(boot_failures),
        "files": {
            "final_pipeline.joblib": hash_file(pipeline_path),
            "sensitivity_pipelines.joblib": hash_file(sensitivity_pipelines_path),
            "bootstrap_pipelines.joblib": hash_file(boot_path),
        },
    }
    save_json(manifest, "models/q2_1_frozen_pipeline/manifest.json")

    # 预注册敏感性，同一5折口径
    sensitivity_rows = []
    scenarios = [
        ("main_weather_group_c0.5", points.copy(), "surface_weathering", 0.5, "group_median"),
        ("zero_c_0.25", points.copy(), "surface_weathering", 0.25, "group_median"),
        ("zero_c_0.75", points.copy(), "surface_weathering", 0.75, "group_median"),
        ("global_median", points.copy(), "global", 0.5, "group_median"),
        ("knn_k5", points.copy(), "surface_weathering", 0.5, "knn"),
        ("include_invalid_15_17", t2.copy(), "surface_weathering", 0.5, "group_median"),
    ]
    for rate in [0.05, 0.10]:
        perturbed = points.copy()
        for ridx, idx in enumerate(perturbed.index):
            for cidx, comp in enumerate(COMPONENTS):
                value = perturbed.at[idx, comp]
                if pd.notna(value) and value > 0:
                    perturbed.at[idx, comp] = float(value) * (1 + (1 if (ridx + cidx) % 2 == 0 else -1) * rate)
        scenarios.append((f"componentwise_pm_{int(rate*100)}pct", perturbed, "surface_weathering", 0.5, "group_median"))
    for scenario, spoints, group_mode, c, kind in scenarios:
        sensitivity_rows.append(quick_sensitivity_cv(spoints, selected_model, final_configs[selected_model], scenario, group_mode, c, kind))
    sensitivity = pd.DataFrame(sensitivity_rows)
    main_value = float(sensitivity.loc[sensitivity["scenario"].eq("main_weather_group_c0.5"), "value"].iloc[0])
    sensitivity["delta_from_main"] = sensitivity["value"] - main_value
    sensitivity["label_agreement"] = np.nan
    save_csv(sensitivity, "results/02_q2/q2_1_sensitivity.csv")

    # 将同折下游比较追加到公共填补比较
    audit_path = ROOT / "results" / "00_audit" / "imputation_method_comparison.csv"
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    append_rows = []
    for _, row in sensitivity[sensitivity["scenario"].isin(["main_weather_group_c0.5", "global_median", "knn_k5"])].iterrows():
        method = {"main_weather_group_c0.5": "legal_group_median", "global_median": "global_median", "knn_k5": "knn_k5"}[row["scenario"]]
        append_rows.append({
            "task_id": "q2_1", "repeat_id": "single_5fold_sensitivity", "fold_id": "all",
            "method": method, "group_key": "surface_weathering", "evaluation_unit": "artifact",
            "n_train": 56, "n_valid": 56, "data_loss_rate": 0.0,
            "downstream_metric": "balanced_accuracy_higher_is_better", "downstream_value": row["value"],
            "ci_low": np.nan, "ci_high": np.nan, "delta_vs_group_median": row["value"] - main_value,
            "leakage_check": "training_fold_fit_only", "fit_success": row["fit_success"], "failure_reason": "",
        })
    audit = pd.concat([audit, pd.DataFrame(append_rows)], ignore_index=True)
    save_csv(audit, "results/00_audit/imputation_method_comparison.csv")

    # 图1：候选模型样本外指标分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, ylabel in [(axes[0], "balanced_accuracy", "文物级平衡准确率"), (axes[1], "brier", "Brier分数（越低越好）")]:
        groups = [metrics.loc[metrics["model"].eq(m), metric].to_numpy() for m in ["Logistic", "CART", "LDA"]]
        bp = ax.boxplot(groups, tick_labels=["Logistic", "CART", "LDA"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#4C78A8", "#F2CF5B", "#E45756"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_ylabel(ylabel)
    save_figure(fig, "q2_1_model_performance_distribution.pdf", metrics)

    # 图2：最终模型汇总混淆矩阵
    selected_oof = oof[oof["model"].eq(selected_model)]
    cm = confusion_matrix(selected_oof["true_label"], selected_oof["predicted_label"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["高钾", "铅钡"])
    ax.set_yticks([0, 1], ["高钾", "铅钡"])
    ax.set_xlabel("预测类别（20次重复汇总）")
    ax.set_ylabel("真实类别")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_figure(fig, "q2_1_selected_model_confusion_matrix.pdf", selected_oof)

    # 图3：Logistic特征稳定性
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    st = stability.sort_values("selection_frequency")
    ax.barh(st["feature_or_balance"], st["selection_frequency"], color="#4C78A8")
    ax.axvline(0.8, color="#E45756", ls="--", lw=1)
    ax.set_xlabel("外层折选择频率")
    save_figure(fig, "q2_1_feature_stability.pdf", stability)

    # 图4：冻结CART规则
    cart_pack = full_pipeline["models"]["CART"]
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_tree(
        cart_pack["model"], feature_names=cart_pack["config"]["features"], class_names=["高钾", "铅钡"],
        filled=True, rounded=True, proportion=True, ax=ax, fontsize=8,
    )
    save_figure(fig, "q2_1_frozen_cart_rules.pdf", full_pipeline["artifact_features"])

    # 图5：ROC与校准
    pooled = selected_oof.copy()
    fpr, tpr, _ = roc_curve(pooled["true_label"], pooled["artifact_probability"])
    prob_true, prob_pred = calibration_curve(pooled["true_label"], pooled["artifact_probability"], n_bins=8, strategy="quantile")
    roc_source = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    cal_source = pd.DataFrame({"predicted_probability": prob_pred, "observed_frequency": prob_true})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].plot(fpr, tpr, color="#4C78A8")
    axes[0].plot([0, 1], [0, 1], color="0.6", ls="--")
    axes[0].set_xlabel("假阳性率")
    axes[0].set_ylabel("真阳性率")
    axes[1].plot(prob_pred, prob_true, marker="o", color="#E45756")
    axes[1].plot([0, 1], [0, 1], color="0.6", ls="--")
    axes[1].set_xlabel("预测铅钡概率")
    axes[1].set_ylabel("观测铅钡频率")
    save_figure(fig, "q2_1_roc_and_calibration.pdf", pd.concat([roc_source.assign(panel="roc"), cal_source.assign(panel="calibration")], ignore_index=True))

    # 图6：一个完整重复的文物级OOF概率
    strip = selected_oof[selected_oof["repeat_id"].eq(0)].sort_values("artifact_probability")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = strip["true_label"].map({0: "#4C78A8", 1: "#E45756"})
    ax.scatter(np.arange(len(strip)), strip["artifact_probability"], c=colors)
    ax.axhline(strip["threshold"].median(), color="0.4", ls="--")
    ax.set_xticks(np.arange(len(strip)), strip["artifact_id"], rotation=90, fontsize=7)
    ax.set_xlabel("文物编号")
    ax.set_ylabel("OOF铅钡概率")
    save_figure(fig, "q2_1_oof_probability_strip.pdf", strip)

    logger.info("问题2.1完成：selected=%s, bootstrap_success=%d", selected_model, len(boot_models))


if __name__ == "__main__":
    main()
