from __future__ import annotations

import logging
import os
import sys
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
    COMPONENT_CN,
    CONFIG,
    MASTER_SEED,
    ROOT,
    CompositionPreprocessor,
    artifact_weights,
    load_joblib,
    load_known_data,
    save_csv,
    save_figure,
    weighted_ols,
)


LOG_PATH = ROOT / "logs" / "12_q1_3_counterfactual.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def fit_effect(points: pd.DataFrame, fixed_active: list[str], zero_c: float = 0.5) -> dict:
    prep = CompositionPreprocessor(zero_c=zero_c, fixed_active=fixed_active).fit(points[COMPONENTS], points["glass_type"])
    tr = prep.transform(points[COMPONENTS], points["glass_type"])
    wstate = points["point_weathering"].map({"无风化": 0.0, "风化": 1.0}).to_numpy()
    X = np.column_stack([np.ones(len(points)), wstate])
    coef, _ = weighted_ols(X, tr.ilr, artifact_weights(points))
    return {"preprocessor": prep, "intercept": coef[0], "beta": coef[1]}


def nearest_reference_info(z: np.ndarray, reference_z: np.ndarray) -> tuple[float, float, bool]:
    distances = np.linalg.norm(reference_z - z[None, :], axis=1)
    nearest = float(distances.min()) if len(distances) else np.nan
    if len(reference_z) >= 2:
        pair = pairwise_distances(reference_z)
        np.fill_diagonal(pair, np.inf)
        threshold = float(np.quantile(pair.min(axis=1), 0.95))
    else:
        threshold = np.nan
    return nearest, threshold, bool(np.isfinite(threshold) and nearest <= threshold)


def main() -> None:
    _, t2 = load_known_data()
    valid = t2[t2["valid_sum_flag"]].copy()
    weathered = valid[valid["point_weathering"].eq("风化")].copy()
    saved = load_joblib("models/q1_2_weathering_models.joblib")
    long_rows = []
    wide_rows = []
    comparison_rows = []
    sensitivity_rows = []
    failure_rows = []
    arrow_rows = []
    rng = np.random.default_rng(MASTER_SEED + 130)

    for glass_type in ["高钾", "铅钡"]:
        type_points = valid[valid["glass_type"].eq(glass_type)].copy()
        type_weathered = weathered[weathered["glass_type"].eq(glass_type)].copy()
        model_pack = saved[glass_type]
        main = model_pack["main"]
        prep = main["preprocessor"]
        beta = np.asarray(main["beta"])
        tr_all = prep.transform(type_points[COMPONENTS], type_points["glass_type"])
        ilr_all = pd.DataFrame(tr_all.ilr, index=type_points.index)

        # 未风化参考以文物中心计
        reference_rows = []
        reference_ids = []
        for artifact_id, idx in type_points[type_points["point_weathering"].eq("无风化")].groupby("artifact_id").groups.items():
            reference_rows.append(ilr_all.loc[list(idx)].mean(axis=0).to_numpy())
            reference_ids.append(artifact_id)
        reference_z = np.asarray(reference_rows)
        robust = None
        robust_threshold = np.nan
        if len(reference_z) > reference_z.shape[1] + 1:
            try:
                robust = MinCovDet(random_state=MASTER_SEED).fit(reference_z)
                robust_d = np.sqrt(robust.mahalanobis(reference_z))
                robust_threshold = float(np.quantile(robust_d, 0.95))
            except Exception:
                robust = None

        # 中位比基准使用主闭合组成
        closed_all = tr_all.closed.copy()
        ratios = {}
        for comp in prep.active_components_:
            med_w = closed_all.loc[type_points["point_weathering"].eq("风化").to_numpy(), comp].median()
            med_u = closed_all.loc[type_points["point_weathering"].eq("无风化").to_numpy(), comp].median()
            ratios[comp] = float(med_w / med_u) if med_u > 0 else np.nan

        for idx, row in type_weathered.iterrows():
            one = row.to_frame().T
            tr_one = prep.transform(one[COMPONENTS], one["glass_type"])
            z_w = tr_one.ilr[0]
            z_0 = z_w - beta
            estimate = prep.inverse_ilr(z_0).iloc[0]
            nearest, domain_threshold, in_domain = nearest_reference_info(z_0, reference_z)
            robust_distance = float(np.sqrt(robust.mahalanobis(z_0[None, :])[0])) if robust is not None else np.nan
            if np.isfinite(robust_threshold):
                in_domain = in_domain and robust_distance <= robust_threshold

            boot_values = []
            successful_models = 0
            for bmodel in model_pack["bootstrap"]:
                try:
                    bprep = bmodel["preprocessor"]
                    btr = bprep.transform(one[COMPONENTS], one["glass_type"])
                    bz0 = btr.ilr[0] - np.asarray(bmodel["beta"])
                    bcomp = bprep.inverse_ilr(bz0).iloc[0]
                    boot_values.append(bcomp[prep.active_components_].to_numpy(dtype=float))
                    successful_models += 1
                except Exception as exc:
                    failure_rows.append({
                        "iteration": bmodel.get("iteration", -1),
                        "glass_type": glass_type,
                        "sample_point": row["sample_point"],
                        "failure_stage": "bootstrap_counterfactual_transform",
                        "reason": str(exc),
                    })
            boot_arr = np.asarray(boot_values)
            if len(boot_arr):
                boot_clr = np.log(boot_arr / 100.0) - np.log(boot_arr / 100.0).mean(axis=1, keepdims=True)
                est_arr = estimate[prep.active_components_].to_numpy(dtype=float)
                est_clr = np.log(est_arr / 100.0) - np.log(est_arr / 100.0).mean()
                uncertainty_radius = float(np.quantile(np.linalg.norm(boot_clr - est_clr, axis=1), 0.95))
            else:
                uncertainty_radius = np.nan
            wide = {
                "artifact_id": row["artifact_id"],
                "sample_point": row["sample_point"],
                "glass_type": glass_type,
                "sum_pct": float(estimate.sum()),
                "closure_error": float(abs(estimate.sum() - 100.0)),
                "aitchison_uncertainty": uncertainty_radius,
                "nearest_unweathered_distance": nearest,
                "nearest_threshold": domain_threshold,
                "robust_mahalanobis": robust_distance,
                "robust_threshold": robust_threshold,
                "applicability_flag": "in_domain" if in_domain else "out_of_domain_or_unidentifiable",
                "bootstrap_success": successful_models,
            }
            for comp in COMPONENTS:
                wide[f"estimated_{comp}_pct"] = float(estimate[comp]) if comp in estimate.index else np.nan
            wide_rows.append(wide)

            for j, comp in enumerate(prep.active_components_):
                values = boot_arr[:, j] if len(boot_arr) else np.array([])
                lo, med, hi = np.quantile(values, [0.025, 0.5, 0.975]) if len(values) else (np.nan, np.nan, np.nan)
                long_rows.append({
                    "artifact_id": row["artifact_id"],
                    "sample_point": row["sample_point"],
                    "glass_type": glass_type,
                    "component": comp,
                    "observed_weathered_pct": row[comp],
                    "estimated_unweathered_pct": float(estimate[comp]),
                    "median_pct": float(med),
                    "ci_low_pct": float(lo),
                    "ci_high_pct": float(hi),
                    "modeled": 1,
                    "bootstrap_success": successful_models,
                })
            for comp in COMPONENTS:
                if comp not in prep.active_components_:
                    long_rows.append({
                        "artifact_id": row["artifact_id"],
                        "sample_point": row["sample_point"],
                        "glass_type": glass_type,
                        "component": comp,
                        "observed_weathered_pct": row[comp],
                        "estimated_unweathered_pct": np.nan,
                        "median_pct": np.nan,
                        "ci_low_pct": np.nan,
                        "ci_high_pct": np.nan,
                        "modeled": 0,
                        "bootstrap_success": successful_models,
                    })

            # 中位比基准
            obs_closed = tr_one.closed.iloc[0]
            baseline = np.array([obs_closed[c] / ratios[c] if np.isfinite(ratios[c]) and ratios[c] > 0 else obs_closed[c] for c in prep.active_components_])
            baseline = baseline / baseline.sum() * 100.0
            baseline_z = np.log(baseline / 100.0) @ prep.basis_.T
            main_dist = float(np.linalg.norm(z_0 - reference_z.mean(axis=0)))
            base_dist = float(np.linalg.norm(baseline_z - reference_z.mean(axis=0)))
            comparison_rows.extend([
                {
                    "sample_point": row["sample_point"],
                    "glass_type": glass_type,
                    "method": "ILR_counterfactual_shift",
                    "distance_to_unweathered_center": main_dist,
                    "within_reference_envelope": int(in_domain),
                    "numerical_failure": 0,
                },
                {
                    "sample_point": row["sample_point"],
                    "glass_type": glass_type,
                    "method": "component_median_ratio_baseline",
                    "distance_to_unweathered_center": base_dist,
                    "within_reference_envelope": int(base_dist <= domain_threshold) if np.isfinite(domain_threshold) else 0,
                    "numerical_failure": 0,
                },
            ])

            arrow_rows.extend([
                {"sample_point": row["sample_point"], "glass_type": glass_type, "state": "观测风化", **{f"ilr_{k+1}": z_w[k] for k in range(len(z_w))}},
                {"sample_point": row["sample_point"], "glass_type": glass_type, "state": "估计未风化", **{f"ilr_{k+1}": z_0[k] for k in range(len(z_0))}},
            ])

            # 输入扰动：对成分交替施加正负扰动后重新闭合
            for rate in [0.05, 0.10]:
                for phase in [0, 1]:
                    perturbed = one.copy()
                    for cidx, comp in enumerate(COMPONENTS):
                        value = perturbed.iloc[0][comp]
                        if pd.notna(value) and value > 0:
                            sign = 1 if (cidx + phase) % 2 == 0 else -1
                            perturbed.at[perturbed.index[0], comp] = float(value) * (1 + sign * rate)
                    ptr = prep.transform(perturbed[COMPONENTS], perturbed["glass_type"])
                    pest = prep.inverse_ilr(ptr.ilr[0] - beta).iloc[0]
                    for comp in prep.active_components_:
                        sensitivity_rows.append({
                            "scenario": f"componentwise_pm_{int(rate*100)}pct_phase{phase}",
                            "sample_point": row["sample_point"],
                            "component": comp,
                            "estimate": float(pest[comp]),
                            "relative_change": float((pest[comp] - estimate[comp]) / estimate[comp]) if estimate[comp] != 0 else np.nan,
                            "rank_or_direction_stable": int(np.sign(pest[comp] - row.get(comp, np.nan)) == np.sign(estimate[comp] - row.get(comp, np.nan))) if pd.notna(row.get(comp, np.nan)) else np.nan,
                        })

        # 零替代敏感性模型（由已知样品重新拟合，不使用未知数据）
        for c in [0.25, 0.75]:
            try:
                smodel = fit_effect(type_points, prep.active_components_, zero_c=c)
                for idx, row in type_weathered.iterrows():
                    one = row.to_frame().T
                    tr = smodel["preprocessor"].transform(one[COMPONENTS], one["glass_type"])
                    est = smodel["preprocessor"].inverse_ilr(tr.ilr[0] - smodel["beta"]).iloc[0]
                    main_est = next(w for w in wide_rows if w["sample_point"] == row["sample_point"])
                    for comp in prep.active_components_:
                        base_value = main_est[f"estimated_{comp}_pct"]
                        sensitivity_rows.append({
                            "scenario": f"zero_c_{c}",
                            "sample_point": row["sample_point"],
                            "component": comp,
                            "estimate": float(est[comp]),
                            "relative_change": float((est[comp] - base_value) / base_value) if base_value else np.nan,
                            "rank_or_direction_stable": np.nan,
                        })
            except Exception as exc:
                failure_rows.append({
                    "iteration": -1,
                    "glass_type": glass_type,
                    "sample_point": "ALL",
                    "failure_stage": f"zero_c_{c}_sensitivity",
                    "reason": str(exc),
                })

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    failures_df = pd.DataFrame(failure_rows, columns=["iteration", "glass_type", "sample_point", "failure_stage", "reason"])
    save_csv(long_df, "results/01_q1/q1_3_counterfactual_long.csv")
    save_csv(wide_df, "results/01_q1/q1_3_counterfactual_wide.csv")
    save_csv(comparison_df, "results/01_q1/q1_3_model_comparison.csv")
    save_csv(sensitivity_df, "results/01_q1/q1_3_sensitivity.csv")
    save_csv(failures_df, "results/01_q1/q1_3_bootstrap_failures.csv")

    if wide_df["closure_error"].max() > 1e-8:
        raise ValueError("问题1.3逆ILR闭合检查失败")
    if (long_df.loc[long_df["modeled"].eq(1), "estimated_unweathered_pct"] <= 0).any():
        raise ValueError("问题1.3产生非正恢复成分")

    # 图1：各类型主要成分观测与恢复均值哑铃图
    mean_long = long_df[long_df["modeled"].eq(1)].groupby(["glass_type", "component"], as_index=False).agg(
        observed=("observed_weathered_pct", "mean"), estimated=("estimated_unweathered_pct", "mean")
    )
    mean_long["abs_change"] = (mean_long["estimated"] - mean_long["observed"]).abs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = mean_long[mean_long["glass_type"].eq(glass_type)].nlargest(8, "abs_change").sort_values("estimated")
        y = np.arange(len(sub))
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.plot([r["observed"], r["estimated"]], [yi, yi], color="0.65", lw=1.5)
        ax.scatter(sub["observed"], y, label="观测风化", color="#E45756")
        ax.scatter(sub["estimated"], y, label="估计未风化", color="#4C78A8")
        ax.set_yticks(y, [COMPONENT_CN.get(c, c) for c in sub["component"]])
        ax.set_xlabel("平均质量百分比（%）")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
        ax.legend()
    save_figure(fig, "q1_3_observed_vs_counterfactual_dumbbell.pdf", mean_long)

    # 图2：主要成分点位恢复区间
    interval_plot = long_df[long_df["modeled"].eq(1)].copy()
    interval_plot["width"] = interval_plot["ci_high_pct"] - interval_plot["ci_low_pct"]
    interval_plot = interval_plot.nlargest(min(30, len(interval_plot)), "width")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.22 * len(interval_plot))))
    y = np.arange(len(interval_plot))
    x = interval_plot["estimated_unweathered_pct"].to_numpy()
    ax.errorbar(x, y, xerr=np.vstack([x - interval_plot["ci_low_pct"], interval_plot["ci_high_pct"] - x]), fmt="o", capsize=2)
    ax.set_yticks(y, interval_plot["sample_point"].astype(str) + "-" + interval_plot["component"])
    ax.set_xlabel("估计风化前质量百分比（95%区间）")
    save_figure(fig, "q1_3_largest_uncertainty_intervals.pdf", interval_plot)

    # 图3：ILR空间恢复箭头，使用各类型自己的二维PCA
    arrow_df = pd.DataFrame(arrow_rows)
    arrow_plot_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = arrow_df[arrow_df["glass_type"].eq(glass_type)].copy()
        ilr_cols = [c for c in sub.columns if c.startswith("ilr_") and sub[c].notna().any()]
        coords = PCA(n_components=2).fit_transform(sub[ilr_cols].fillna(0).to_numpy())
        sub["pc1"], sub["pc2"] = coords[:, 0], coords[:, 1]
        for sample, pair in sub.groupby("sample_point"):
            if len(pair) != 2:
                continue
            obs = pair[pair["state"].eq("观测风化")].iloc[0]
            est = pair[pair["state"].eq("估计未风化")].iloc[0]
            ax.annotate("", xy=(est["pc1"], est["pc2"]), xytext=(obs["pc1"], obs["pc2"]), arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 0.8})
        for state, color, marker in [("观测风化", "#E45756", "s"), ("估计未风化", "#4C78A8", "o")]:
            ss = sub[sub["state"].eq(state)]
            ax.scatter(ss["pc1"], ss["pc2"], label=state, color=color, marker=marker, s=25)
        ax.set_xlabel("恢复投影第一轴")
        ax.set_ylabel("恢复投影第二轴")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
        ax.legend()
        arrow_plot_rows.append(sub)
    save_figure(fig, "q1_3_counterfactual_migration_arrows.pdf", pd.concat(arrow_plot_rows, ignore_index=True))

    # 图4：点位不确定性排行
    rank = wide_df.sort_values("aitchison_uncertainty", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.24 * len(rank))))
    colors = rank["glass_type"].map({"高钾": "#4C78A8", "铅钡": "#E45756"})
    ax.barh(rank["sample_point"].astype(str), rank["aitchison_uncertainty"], color=colors)
    ax.set_xlabel("95% Aitchison不确定半径")
    ax.set_ylabel("风化采样点")
    save_figure(fig, "q1_3_uncertainty_ranking.pdf", rank)

    logger.info("问题1.3完成：风化点%d个，闭合最大误差%.3g", len(wide_df), wide_df["closure_error"].max())


if __name__ == "__main__":
    main()
