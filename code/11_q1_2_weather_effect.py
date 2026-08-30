from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, shapiro, skew
from sklearn.decomposition import PCA

from common import (
    COMPONENTS,
    COMPONENT_CN,
    CONFIG,
    MASTER_SEED,
    ROOT,
    CompositionPreprocessor,
    aggregate_artifact_centers,
    artifact_weights,
    bh_fdr,
    dump_joblib,
    load_known_data,
    save_csv,
    save_figure,
    weighted_ols,
)


LOG_PATH = ROOT / "logs" / "11_q1_2_weather_effect.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def fit_type_model(
    points: pd.DataFrame,
    fixed_active: list[str] | None = None,
    zero_c: float = 0.5,
    artifact_key: str = "artifact_id",
) -> dict:
    work = points.copy()
    if artifact_key != "artifact_id":
        work["artifact_id_original"] = work["artifact_id"]
        work["artifact_id"] = work[artifact_key].astype(str)
    prep = CompositionPreprocessor(zero_c=zero_c, fixed_active=fixed_active).fit(
        work[COMPONENTS], work["glass_type"]
    )
    tr = prep.transform(work[COMPONENTS], work["glass_type"])
    weather = work["point_weathering"].map({"无风化": 0.0, "风化": 1.0}).to_numpy()
    if not np.isfinite(weather).all() or len(np.unique(weather)) < 2:
        raise ValueError("风化状态不足两个水平")
    X = np.column_stack([np.ones(len(work)), weather])
    weights = artifact_weights(work)
    coef, residual = weighted_ols(X, tr.ilr, weights)
    fitted = X @ coef
    return {
        "preprocessor": prep,
        "transform": tr,
        "intercept": coef[0],
        "beta": coef[1],
        "effect_norm": float(np.linalg.norm(coef[1])),
        "residual": residual,
        "fitted": fitted,
        "weights": weights,
        "work": work,
    }


def permutation_p(model: dict, rng: np.random.Generator) -> float:
    work = model["work"]
    ilr = model["transform"].ilr
    artifact_state = work.groupby("artifact_id")["point_weathering"].agg(
        lambda s: "风化" if (s == "风化").mean() >= 0.5 else "无风化"
    )
    observed = model["effect_norm"]
    exceed = 0
    B = int(CONFIG["permutation_B"])
    artifact_ids = artifact_state.index.to_numpy()
    states = artifact_state.to_numpy()
    for _ in range(B):
        perm_states = rng.permutation(states)
        mapping = dict(zip(artifact_ids, perm_states))
        w = work["artifact_id"].map(mapping).map({"无风化": 0.0, "风化": 1.0}).to_numpy()
        X = np.column_stack([np.ones(len(work)), w])
        coef, _ = weighted_ols(X, ilr, model["weights"])
        exceed += np.linalg.norm(coef[1]) >= observed - 1e-12
    return float((exceed + 1) / (B + 1))


def bootstrap_models(
    points: pd.DataFrame,
    main_active: list[str],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    artifact_ids = points["artifact_id"].unique()
    target = int(CONFIG["bootstrap_B"])
    max_attempts = int(CONFIG["bootstrap_Bmax"])
    models = []
    failures = []
    attempts = 0
    while len(models) < target and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(artifact_ids, size=len(artifact_ids), replace=True)
        chunks = []
        for draw, artifact_id in enumerate(sampled):
            chunk = points[points["artifact_id"].eq(artifact_id)].copy()
            chunk["boot_artifact_id"] = f"b{draw:03d}_{artifact_id}"
            chunks.append(chunk)
        boot = pd.concat(chunks, ignore_index=True)
        try:
            model = fit_type_model(boot, fixed_active=main_active, artifact_key="boot_artifact_id")
            models.append({
                "iteration": len(models),
                "preprocessor": model["preprocessor"],
                "intercept": model["intercept"],
                "beta": model["beta"],
                "effect_norm": model["effect_norm"],
            })
        except Exception as exc:
            failures.append({"attempt": attempts, "reason": str(exc)})
    return models, failures


def leave_one_out(points: pd.DataFrame, main_beta: np.ndarray) -> pd.DataFrame:
    rows = []
    for artifact_id in sorted(points["artifact_id"].unique()):
        sub = points[~points["artifact_id"].eq(artifact_id)]
        try:
            model = fit_type_model(sub, fixed_active=None)
            overlap = min(len(main_beta), len(model["beta"]))
            direction = float(np.mean(np.sign(main_beta[:overlap]) == np.sign(model["beta"][:overlap])))
            rows.append({
                "left_out_artifact": artifact_id,
                "effect_norm": model["effect_norm"],
                "direction_agreement": direction,
                "fit_success": 1,
                "failure_reason": "",
            })
        except Exception as exc:
            rows.append({
                "left_out_artifact": artifact_id,
                "effect_norm": np.nan,
                "direction_agreement": np.nan,
                "fit_success": 0,
                "failure_reason": str(exc),
            })
    return pd.DataFrame(rows)


def main() -> None:
    _, t2 = load_known_data()
    valid = t2[t2["valid_sum_flag"]].copy()
    rng = np.random.default_rng(MASTER_SEED + 120)
    effect_rows = []
    overall_rows = []
    contrast_rows = []
    baseline_rows = []
    residual_rows = []
    sensitivity_rows = []
    failure_rows = []
    loo_all = []
    saved_models = {}
    plot_point_rows = []

    for glass_type in ["高钾", "铅钡"]:
        points = valid[valid["glass_type"].eq(glass_type)].copy()
        n_by_weather = points.groupby("point_weathering")["artifact_id"].nunique()
        if len(n_by_weather) < 2 or n_by_weather.min() < 5:
            raise ValueError(f"{glass_type}风化/无风化文物不足5件，不能按清单主模型拟合：{n_by_weather.to_dict()}")
        main_model = fit_type_model(points)
        prep = main_model["preprocessor"]
        p_overall = permutation_p(main_model, rng)
        models, failures = bootstrap_models(points, prep.active_components_, rng)
        for f in failures:
            failure_rows.append({"glass_type": glass_type, **f})
        if len(models) < CONFIG["bootstrap_B"]:
            logger.warning("%s Bootstrap成功数不足1000：%d", glass_type, len(models))
        saved_models[glass_type] = {
            "main": {
                "preprocessor": prep,
                "intercept": main_model["intercept"],
                "beta": main_model["beta"],
                "effect_norm": main_model["effect_norm"],
            },
            "bootstrap": models,
        }
        beta_boot = np.asarray([m["beta"] for m in models])
        effect_boot = np.asarray([m["effect_norm"] for m in models])
        coord_p = []
        for j, beta in enumerate(main_model["beta"]):
            values = beta_boot[:, j] if len(beta_boot) else np.array([])
            lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
            sign_freq = float(max(np.mean(values >= 0), np.mean(values <= 0))) if len(values) else np.nan
            p_two = float(2 * min(np.mean(values >= 0), np.mean(values <= 0))) if len(values) else np.nan
            coord_p.append(min(1.0, p_two) if np.isfinite(p_two) else np.nan)
            effect_rows.append({
                "glass_type": glass_type,
                "ilr_coordinate": f"ilr_{j+1}",
                "balance_definition": "Helmert_ILR; basis见results/00_audit或q1_2_basis",
                "beta_weathering": float(beta),
                "boot_se": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_raw": min(1.0, p_two) if np.isfinite(p_two) else np.nan,
                "p_fdr": np.nan,
                "sign_frequency": sign_freq,
                "n_artifacts": int(points["artifact_id"].nunique()),
                "n_points": len(points),
            })
        start = len(effect_rows) - len(main_model["beta"])
        adjusted = bh_fdr(coord_p)
        for offset, value in enumerate(adjusted):
            effect_rows[start + offset]["p_fdr"] = float(value)

        overall_rows.append({
            "glass_type": glass_type,
            "effect_norm": main_model["effect_norm"],
            "effect_ci_low": float(np.quantile(effect_boot, 0.025)) if len(effect_boot) else np.nan,
            "effect_ci_high": float(np.quantile(effect_boot, 0.975)) if len(effect_boot) else np.nan,
            "permutation_p": p_overall,
            "model_formula": "ILR ~ point_weathering",
            "covariates_used": "none_due_small_sample_preregistered_rule",
            "bootstrap_success": len(models),
            "bootstrap_failure": len(failures),
        })

        center_unw = prep.inverse_ilr(main_model["intercept"]).iloc[0]
        center_w = prep.inverse_ilr(main_model["intercept"] + main_model["beta"]).iloc[0]
        boot_unw = np.asarray([m["preprocessor"].inverse_ilr(m["intercept"]).iloc[0].to_numpy() for m in models])
        boot_w = np.asarray([m["preprocessor"].inverse_ilr(m["intercept"] + m["beta"]).iloc[0].to_numpy() for m in models])
        boot_diff = boot_w - boot_unw
        for j, comp in enumerate(prep.active_components_):
            lo, hi = np.quantile(boot_diff[:, j], [0.025, 0.975]) if len(boot_diff) else (np.nan, np.nan)
            contrast_rows.append({
                "glass_type": glass_type,
                "component": comp,
                "predicted_unweathered_pct": float(center_unw[comp]),
                "predicted_weathered_pct": float(center_w[comp]),
                "difference_pct_point": float(center_w[comp] - center_unw[comp]),
                "ci_low": float(lo),
                "ci_high": float(hi),
            })

        tr = main_model["transform"]
        point_plot = points[["artifact_id", "sample_point", "point_weathering"]].copy()
        point_plot["glass_type"] = glass_type
        pca = PCA(n_components=2).fit_transform(tr.ilr)
        point_plot["pc1"] = pca[:, 0]
        point_plot["pc2"] = pca[:, 1]
        plot_point_rows.append(point_plot)

        for comp in prep.active_components_:
            values = tr.closed[comp]
            unw = values[points["point_weathering"].eq("无风化").to_numpy()]
            wea = values[points["point_weathering"].eq("风化").to_numpy()]
            if len(unw) and len(wea):
                stat, p = mannwhitneyu(unw, wea, alternative="two-sided")
                effect = float(np.median(wea) - np.median(unw))
            else:
                stat, p, effect = np.nan, np.nan, np.nan
            baseline_rows.append({
                "glass_type": glass_type,
                "component": comp,
                "test_used": "Mann_Whitney_descriptive_point_level_with_artifact_weighted_main_model",
                "statistic": float(stat),
                "p_raw": float(p),
                "p_fdr": np.nan,
                "effect_size": effect,
            })

        for j in range(main_model["residual"].shape[1]):
            resid = main_model["residual"][:, j]
            fitted = main_model["fitted"][:, j]
            shapiro_p = float(shapiro(resid).pvalue) if 3 <= len(resid) <= 5000 else np.nan
            residual_rows.append({
                "glass_type": glass_type,
                "coordinate": f"ilr_{j+1}",
                "diagnostic": "weighted_linear_residual_summary",
                "residual_mean": float(np.average(resid, weights=main_model["weights"])),
                "residual_sd": float(np.sqrt(np.average((resid - np.average(resid, weights=main_model["weights"])) ** 2, weights=main_model["weights"]))),
                "residual_skew": float(skew(resid, bias=False)),
                "residual_kurtosis": float(kurtosis(resid, bias=False)),
                "shapiro_p": shapiro_p,
                "abs_corr_fitted_residual": float(abs(np.corrcoef(fitted, resid)[0, 1])) if np.std(fitted) > 0 and np.std(resid) > 0 else np.nan,
                "flag": "review" if shapiro_p < 0.05 else "ok_or_low_power",
            })

        loo = leave_one_out(points, main_model["beta"])
        loo.insert(0, "glass_type", glass_type)
        loo_all.append(loo)

        # 预注册灵敏度：零替代、纳入15/17、点位风化全部继承文物状态
        scenarios = []
        for c in CONFIG["zero_c_sensitivity"]:
            scenarios.append((f"zero_c_{c}", points.copy(), float(c)))
        all_type = t2[t2["glass_type"].eq(glass_type)].copy()
        scenarios.append(("include_invalid_15_17", all_type, 0.5))
        inherited = points.copy()
        inherited["point_weathering"] = inherited["surface_weathering"]
        scenarios.append(("all_points_inherit_artifact_weathering", inherited, 0.5))
        for scenario, sdf, c in scenarios:
            try:
                smodel = fit_type_model(sdf, fixed_active=prep.active_components_, zero_c=c)
                agreement = float(np.mean(np.sign(smodel["beta"]) == np.sign(main_model["beta"])))
                sensitivity_rows.append({
                    "scenario": scenario,
                    "glass_type": glass_type,
                    "effect_norm": smodel["effect_norm"],
                    "direction_agreement": agreement,
                    "key_component_agreement": agreement,
                    "conclusion_grade": "stable" if agreement >= 0.8 else "sensitive",
                    "fit_success": 1,
                    "failure_reason": "",
                })
            except Exception as exc:
                sensitivity_rows.append({
                    "scenario": scenario,
                    "glass_type": glass_type,
                    "effect_norm": np.nan,
                    "direction_agreement": np.nan,
                    "key_component_agreement": np.nan,
                    "conclusion_grade": "not_computable",
                    "fit_success": 0,
                    "failure_reason": str(exc),
                })

        basis = pd.DataFrame(
            prep.basis_, index=[f"ilr_{i+1}" for i in range(prep.basis_.shape[0])], columns=prep.active_components_
        ).reset_index(names="coordinate")
        save_csv(basis, f"results/01_q1/q1_2_ilr_basis_{glass_type}.csv")

    effects = pd.DataFrame(effect_rows)
    overall = pd.DataFrame(overall_rows)
    contrast = pd.DataFrame(contrast_rows)
    baseline = pd.DataFrame(baseline_rows)
    for glass_type, idx in baseline.groupby("glass_type").groups.items():
        baseline.loc[idx, "p_fdr"] = bh_fdr(baseline.loc[idx, "p_raw"])
    residuals = pd.DataFrame(residual_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    loo_df = pd.concat(loo_all, ignore_index=True)
    failures_df = pd.DataFrame(failure_rows, columns=["glass_type", "attempt", "reason"])
    save_csv(effects, "results/01_q1/q1_2_ilr_effects.csv")
    save_csv(overall, "results/01_q1/q1_2_overall_effect.csv")
    save_csv(contrast, "results/01_q1/q1_2_composition_contrast.csv")
    save_csv(baseline, "results/01_q1/q1_2_component_baseline.csv")
    save_csv(residuals, "results/01_q1/q1_2_residual_diagnostics.csv")
    save_csv(sensitivity, "results/01_q1/q1_2_sensitivity.csv")
    save_csv(loo_df, "results/01_q1/q1_2_leave_one_out.csv")
    save_csv(failures_df, "results/01_q1/q1_2_bootstrap_failures.csv")
    dump_joblib(saved_models, "models/q1_2_weathering_models.joblib")

    # 图1：两类各自在ILR空间的PCA投影
    point_plot = pd.concat(plot_point_rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = point_plot[point_plot["glass_type"].eq(glass_type)]
        for state, color, marker in [("无风化", "#4C78A8", "o"), ("风化", "#E45756", "s")]:
            ss = sub[sub["point_weathering"].eq(state)]
            ax.scatter(ss["pc1"], ss["pc2"], label=state, color=color, marker=marker, alpha=0.8)
        ax.set_xlabel("Aitchison-PCA 第一轴")
        ax.set_ylabel("Aitchison-PCA 第二轴")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
        ax.legend()
    save_figure(fig, "q1_2_aitchison_pca_weathering.pdf", point_plot)

    # 图2：ILR效应森林图
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=False)
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = effects[effects["glass_type"].eq(glass_type)].reset_index(drop=True)
        y = np.arange(len(sub))
        x = sub["beta_weathering"].to_numpy()
        ax.errorbar(x, y, xerr=np.vstack([x - sub["ci_low"], sub["ci_high"] - x]), fmt="o", capsize=2)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(y, sub["ilr_coordinate"])
        ax.set_xlabel("风化系数（95% Bootstrap区间）")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q1_2_ilr_effect_forest.pdf", effects)

    # 图3：主要成分反变换变化
    ranked_contrast = contrast.assign(abs_change=contrast["difference_pct_point"].abs())
    top = pd.concat(
        [sub.nlargest(min(8, len(sub)), "abs_change") for _, sub in ranked_contrast.groupby("glass_type")],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = top[top["glass_type"].eq(glass_type)].sort_values("difference_pct_point")
        y = np.arange(len(sub))
        x = sub["difference_pct_point"].to_numpy()
        ax.errorbar(x, y, xerr=np.vstack([x - sub["ci_low"], sub["ci_high"] - x]), fmt="o", capsize=2)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(y, [COMPONENT_CN.get(c, c) for c in sub["component"]])
        ax.set_xlabel("风化－未风化（百分点）")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q1_2_composition_change_intervals.pdf", contrast)

    # 图4：主要成分描述性分布（使用闭合后的图源重新从主模型构造）
    dist_rows = []
    for glass_type in ["高钾", "铅钡"]:
        points = valid[valid["glass_type"].eq(glass_type)].copy()
        model = saved_models[glass_type]["main"]
        tr = model["preprocessor"].transform(points[COMPONENTS], points["glass_type"])
        key = contrast[contrast["glass_type"].eq(glass_type)].assign(a=lambda x: x["difference_pct_point"].abs()).nlargest(4, "a")["component"]
        for comp in key:
            for idx, value in tr.closed[comp].items():
                dist_rows.append({"glass_type": glass_type, "component": comp, "weathering": points.loc[idx, "point_weathering"], "pct": value})
    dist = pd.DataFrame(dist_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = dist[dist["glass_type"].eq(glass_type)]
        comps = list(sub["component"].drop_duplicates())
        positions = []
        data = []
        colors = []
        labels = []
        pos = 1
        for comp in comps:
            for state, color in [("无风化", "#4C78A8"), ("风化", "#E45756")]:
                data.append(sub[(sub["component"].eq(comp)) & (sub["weathering"].eq(state))]["pct"].to_numpy())
                positions.append(pos)
                colors.append(color)
                labels.append(f"{comp}\n{state}")
                pos += 1
            pos += 0.5
        bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("闭合后质量百分比（%）")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q1_2_component_distribution_baseline.pdf", dist)

    # 图5：留一文物影响
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = loo_df[loo_df["glass_type"].eq(glass_type)].sort_values("effect_norm")
        ax.scatter(np.arange(len(sub)), sub["effect_norm"], c=sub["direction_agreement"], cmap="viridis", vmin=0, vmax=1)
        main_norm = float(overall.loc[overall["glass_type"].eq(glass_type), "effect_norm"].iloc[0])
        ax.axhline(main_norm, color="#E45756", ls="--", lw=1)
        ax.set_xlabel("逐件留一序号")
        ax.set_ylabel("整体风化效应范数")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q1_2_leave_one_artifact_influence.pdf", loo_df)

    logger.info("问题1.2完成：%s", overall.to_dict(orient="records"))


if __name__ == "__main__":
    main()
