from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression

from common import (
    CONFIG,
    MASTER_SEED,
    ROOT,
    bh_fdr,
    corrected_cramers_v,
    load_known_data,
    save_csv,
    save_figure,
    save_json,
)


LOG_PATH = ROOT / "logs" / "10_q1_1_association.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def chi2_stat_from_codes(x_codes: np.ndarray, y_codes: np.ndarray, r: int, c: int) -> float:
    table = np.bincount(x_codes * c + y_codes, minlength=r * c).reshape(r, c)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / table.sum()
    mask = expected > 0
    return float(np.sum(((table - expected) ** 2)[mask] / expected[mask]))


def association_test(df: pd.DataFrame, attribute: str, rng: np.random.Generator) -> dict:
    table_df = pd.crosstab(df[attribute], df["surface_weathering"], dropna=False)
    table = table_df.to_numpy()
    chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
    sparse = bool((expected < 1).any() or ((expected < 5).mean() > 0.20))
    mc_se = np.nan
    if sparse and table.shape == (2, 2):
        _, p = fisher_exact(table, alternative="two-sided")
        test_used = "Fisher_exact_two_sided"
        statistic = chi2
    elif sparse:
        x_cat = pd.Categorical(df[attribute])
        y_cat = pd.Categorical(df["surface_weathering"])
        x_codes = x_cat.codes.astype(int)
        y_codes = y_cat.codes.astype(int)
        observed = chi2_stat_from_codes(x_codes, y_codes, len(x_cat.categories), len(y_cat.categories))
        exceed = 0
        B = int(CONFIG["permutation_B"])
        for _ in range(B):
            perm = rng.permutation(y_codes)
            exceed += chi2_stat_from_codes(x_codes, perm, len(x_cat.categories), len(y_cat.categories)) >= observed - 1e-12
        p = (exceed + 1) / (B + 1)
        mc_se = float(np.sqrt(p * (1 - p) / B))
        test_used = "fixed_margin_label_permutation_10000"
        statistic = observed
    else:
        p = p_chi
        test_used = "Pearson_chi_square"
        statistic = chi2
    return {
        "table": table_df,
        "expected": expected,
        "test_used": test_used,
        "statistic": float(statistic),
        "df": int(dof),
        "p_raw": float(p),
        "cramer_v_corrected": corrected_cramers_v(table),
        "monte_carlo_se": mc_se,
    }


def bootstrap_v(df: pd.DataFrame, attribute: str, rng: np.random.Generator) -> tuple[float, float, float]:
    vals = []
    n = len(df)
    for _ in range(int(CONFIG["bootstrap_B"])):
        boot = df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        table = pd.crosstab(boot[attribute], boot["surface_weathering"], dropna=False).to_numpy()
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        try:
            vals.append(corrected_cramers_v(table))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), float(len(vals) / CONFIG["bootstrap_B"])


def fit_multivariable_logistic(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    y = work["surface_weathering"].map({"无风化": 0, "风化": 1}).astype(int)
    refs = {}
    pieces = []
    for col in ["glass_type", "pattern", "color"]:
        levels = sorted(work[col].astype(str).unique())
        refs[col] = levels[0]
        cat = pd.Categorical(work[col].astype(str), categories=levels)
        dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
        pieces.append(dummies)
    X = pd.concat(pieces, axis=1)
    X = sm.add_constant(X, has_constant="add")
    diagnostics = {"reference_levels": refs, "estimator": "statsmodels_GLM_binomial", "converged": False}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.GLM(y.to_numpy(), X.to_numpy(dtype=float), family=sm.families.Binomial())
            fit = model.fit(maxiter=2000, disp=0)
        ci = fit.conf_int()
        rows = []
        for i, term in enumerate(X.columns):
            beta = float(fit.params[i])
            rows.append({
                "term": term,
                "reference_level": "intercept" if term == "const" else refs.get(term.split("_")[0], ""),
                "beta": beta,
                "se": float(fit.bse[i]),
                "odds_ratio": float(np.exp(np.clip(beta, -30, 30))),
                "ci_low": float(np.exp(np.clip(ci[i, 0], -30, 30))),
                "ci_high": float(np.exp(np.clip(ci[i, 1], -30, 30))),
                "p_value": float(fit.pvalues[i]),
                "estimator": "GLM_binomial_MLE",
                "converged": bool(getattr(fit, "converged", True)),
            })
        diagnostics.update({
            "converged": bool(getattr(fit, "converged", True)),
            "iterations": fit.fit_history.get("iteration", None),
            "condition_number": float(np.linalg.cond(X.to_numpy(dtype=float))),
            "warnings": [str(w.message) for w in caught],
            "fallback_used": False,
        })
        if not diagnostics["converged"] or not np.isfinite(fit.params).all():
            raise RuntimeError("GLM未收敛或参数非有限")
        return pd.DataFrame(rows), diagnostics
    except Exception as exc:
        logger.warning("普通Logistic失败，切换L2弱正则：%s", exc)
        scaler_X = X.drop(columns="const").to_numpy(dtype=float)
        clf = LogisticRegression(C=100.0, penalty="l2", solver="liblinear", max_iter=5000)
        clf.fit(scaler_X, y)
        boot_coefs = []
        for _ in range(CONFIG["bootstrap_B"]):
            idx = rng.integers(0, len(work), len(work))
            if len(np.unique(y.iloc[idx])) < 2:
                continue
            try:
                m = LogisticRegression(C=100.0, penalty="l2", solver="liblinear", max_iter=5000)
                m.fit(scaler_X[idx], y.iloc[idx])
                boot_coefs.append(np.r_[m.intercept_, m.coef_.ravel()])
            except Exception:
                continue
        coef = np.r_[clf.intercept_, clf.coef_.ravel()]
        boot_arr = np.asarray(boot_coefs)
        terms = ["const"] + X.drop(columns="const").columns.tolist()
        rows = []
        for j, term in enumerate(terms):
            lo, hi = (np.quantile(boot_arr[:, j], [0.025, 0.975]) if len(boot_arr) else [np.nan, np.nan])
            rows.append({
                "term": term,
                "reference_level": "intercept" if term == "const" else refs.get(term.split("_")[0], ""),
                "beta": float(coef[j]),
                "se": float(np.std(boot_arr[:, j], ddof=1)) if len(boot_arr) > 1 else np.nan,
                "odds_ratio": float(np.exp(np.clip(coef[j], -30, 30))),
                "ci_low": float(np.exp(np.clip(lo, -30, 30))) if np.isfinite(lo) else np.nan,
                "ci_high": float(np.exp(np.clip(hi, -30, 30))) if np.isfinite(hi) else np.nan,
                "p_value": np.nan,
                "estimator": "L2_stabilized_logistic_bootstrap_interval",
                "converged": True,
            })
        diagnostics.update({
            "estimator": "L2_stabilized_logistic",
            "converged": True,
            "fallback_used": True,
            "fallback_reason": str(exc),
            "bootstrap_success": len(boot_coefs),
        })
        return pd.DataFrame(rows), diagnostics


def main() -> None:
    _, t2 = load_known_data()
    t1 = t2[["artifact_id", "glass_type", "pattern", "color", "surface_weathering"]].drop_duplicates("artifact_id").copy()
    if len(t1) != 58:
        raise ValueError("问题1.1统计单位必须为58件文物")
    t1["color"] = t1["color"].fillna("未知").astype(str)
    rng = np.random.default_rng(MASTER_SEED + 101)
    attrs = ["glass_type", "pattern", "color"]

    contingency_rows = []
    association_rows = []
    raw_results = {}
    for attr in attrs:
        res = association_test(t1, attr, rng)
        raw_results[attr] = res
        lo, hi, success_rate = bootstrap_v(t1, attr, rng)
        table = res["table"]
        expected = res["expected"]
        for i, level in enumerate(table.index):
            for j, weather in enumerate(table.columns):
                contingency_rows.append({
                    "attribute": attr,
                    "level": str(level),
                    "weathering_level": str(weather),
                    "observed_n": int(table.iloc[i, j]),
                    "expected_n": float(expected[i, j]),
                })
        association_rows.append({
            "attribute": attr,
            "test_used": res["test_used"],
            "statistic": res["statistic"],
            "df": res["df"],
            "p_raw": res["p_raw"],
            "p_fdr": np.nan,
            "cramer_v_corrected": res["cramer_v_corrected"],
            "v_ci_low": lo,
            "v_ci_high": hi,
            "bootstrap_success_rate": success_rate,
            "monte_carlo_se": res["monte_carlo_se"],
            "n_artifacts": len(t1),
        })
    assoc = pd.DataFrame(association_rows)
    assoc["p_fdr"] = bh_fdr(assoc["p_raw"])
    save_csv(pd.DataFrame(contingency_rows), "results/01_q1/q1_1_contingency.csv")
    save_csv(assoc, "results/01_q1/q1_1_association.csv")

    logistic, diagnostics = fit_multivariable_logistic(t1, rng)
    save_csv(logistic, "results/01_q1/q1_1_logistic.csv")
    save_json(diagnostics, "results/01_q1/q1_1_diagnostics.json")

    sensitivity_rows = []
    original_t1 = t1.copy()
    scenarios = {"color_unknown_level": original_t1}
    complete = original_t1[original_t1["color"].ne("未知")].copy()
    scenarios["color_complete_case"] = complete
    rare_levels = original_t1["color"].value_counts()
    rare = set(rare_levels[rare_levels < 3].index)
    merged = original_t1.copy()
    merged["color"] = merged["color"].where(~merged["color"].isin(rare), "其他")
    scenarios["rare_color_merged"] = merged
    for scenario, sdf in scenarios.items():
        if sdf["color"].nunique() < 2:
            continue
        res = association_test(sdf, "color", rng)
        loo = []
        for idx in sdf.index:
            table = pd.crosstab(sdf.drop(index=idx)["color"], sdf.drop(index=idx)["surface_weathering"]).to_numpy()
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                loo.append(corrected_cramers_v(table))
        sensitivity_rows.append({
            "scenario": scenario,
            "term_or_attribute": "color",
            "estimate": res["cramer_v_corrected"],
            "p_or_interval": res["p_raw"],
            "sign_consistent": int((min(loo) if loo else 0) >= 0),
            "leave_one_out_range": f"{min(loo):.6g}--{max(loo):.6g}" if loo else "not_computable",
            "n_artifacts": len(sdf),
        })
    for attr in ["glass_type", "pattern"]:
        loo = []
        for idx in t1.index:
            table = pd.crosstab(t1.drop(index=idx)[attr], t1.drop(index=idx)["surface_weathering"]).to_numpy()
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                loo.append(corrected_cramers_v(table))
        sensitivity_rows.append({
            "scenario": "leave_one_artifact_out",
            "term_or_attribute": attr,
            "estimate": float(assoc.loc[assoc["attribute"].eq(attr), "cramer_v_corrected"].iloc[0]),
            "p_or_interval": np.nan,
            "sign_consistent": int(min(loo) >= 0) if loo else np.nan,
            "leave_one_out_range": f"{min(loo):.6g}--{max(loo):.6g}" if loo else "not_computable",
            "n_artifacts": len(t1),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_csv(sensitivity, "results/01_q1/q1_1_sensitivity.csv")

    # 百分比堆积条形图
    label_map = {"glass_type": "玻璃类型", "pattern": "纹饰", "color": "颜色"}
    for attr in attrs:
        counts = pd.crosstab(t1[attr], t1["surface_weathering"])
        props = counts.div(counts.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        props.plot(kind="bar", stacked=True, color=["#4C78A8", "#E45756"], ax=ax)
        ax.set_xlabel(label_map[attr])
        ax.set_ylabel("文物比例")
        ax.legend(title="表面状态")
        ax.tick_params(axis="x", rotation=30)
        for i, total in enumerate(counts.sum(axis=1)):
            ax.text(i, 1.015, f"n={int(total)}", ha="center", va="bottom", fontsize=8)
        source = counts.reset_index().melt(id_vars=attr, var_name="surface_weathering", value_name="count")
        source["proportion"] = source.apply(lambda r: r["count"] / counts.loc[r[attr]].sum(), axis=1)
        save_figure(fig, f"q1_1_{attr}_weathering_stacked.pdf", source)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(assoc))
    x = assoc["cramer_v_corrected"].to_numpy()
    xerr = np.vstack([x - assoc["v_ci_low"].to_numpy(), assoc["v_ci_high"].to_numpy() - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#2A6F97", ecolor="#61A5C2", capsize=3)
    ax.set_yticks(y, [label_map[a] for a in assoc["attribute"]])
    ax.set_xlabel("校正 Cramér's V（95%区间）")
    ax.axvline(0, color="0.5", lw=0.8)
    save_figure(fig, "q1_1_cramers_v_forest.pdf", assoc)

    plot_log = logistic[logistic["term"].ne("const") & logistic["ci_low"].notna() & logistic["ci_high"].notna()].copy()
    if len(plot_log):
        fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.38 * len(plot_log))))
        y = np.arange(len(plot_log))
        orv = plot_log["odds_ratio"].to_numpy()
        lo = plot_log["ci_low"].to_numpy()
        hi = plot_log["ci_high"].to_numpy()
        ax.errorbar(orv, y, xerr=np.vstack([orv - lo, hi - orv]), fmt="o", capsize=3, color="#8C2D04")
        ax.set_xscale("log")
        ax.axvline(1, color="0.5", lw=0.8)
        ax.set_yticks(y, plot_log["term"])
        ax.set_xlabel("调整后优势比 OR（对数尺度，95%区间）")
        save_figure(fig, "q1_1_logistic_or_forest.pdf", plot_log)

    logger.info("问题1.1完成：%s", assoc[["attribute", "p_raw", "p_fdr", "cramer_v_corrected"]].to_dict(orient="records"))


if __name__ == "__main__":
    main()
