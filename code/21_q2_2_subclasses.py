from __future__ import annotations

import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from common import (
    COMPONENTS,
    COMPONENT_CN,
    CONFIG,
    MASTER_SEED,
    ROOT,
    CompositionPreprocessor,
    aggregate_artifact_centers,
    load_known_data,
    save_csv,
    save_figure,
)


LOG_PATH = ROOT / "logs" / "21_q2_2_subclasses.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def robust_mad(x: pd.Series) -> float:
    med = x.median()
    return float((x - med).abs().median())


def variable_selection(centers: pd.DataFrame, detection: dict[str, float], threshold: float = 0.5) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    comps = [c for c in COMPONENTS if c in centers.columns]
    logx = np.log(centers[comps].to_numpy() / 100.0)
    clr = logx - logx.mean(axis=1, keepdims=True)
    corr = pd.DataFrame(clr, columns=comps).corr(method="spearman").fillna(0).to_numpy()
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average") if len(comps) > 1 else np.empty((0, 4))
    labels = fcluster(z, t=threshold, criterion="distance") if len(comps) > 1 else np.ones(1, dtype=int)
    rows = []
    representatives = []
    clr_df = pd.DataFrame(clr, columns=comps)
    for cluster_id in sorted(np.unique(labels)):
        members = [c for c, lab in zip(comps, labels) if lab == cluster_id]
        ranking = sorted(members, key=lambda c: (-detection.get(c, 0.0), -robust_mad(clr_df[c]), COMPONENTS.index(c)))
        rep = ranking[0]
        representatives.append(rep)
        for comp in members:
            rows.append({
                "variable_cluster": int(cluster_id), "component": comp,
                "representative_flag": int(comp == rep), "detection_rate": detection.get(comp, np.nan),
                "robust_mad": robust_mad(clr_df[comp]),
                "selection_reason": "highest_detection_then_robust_MAD_then_fixed_component_order" if comp == rep else "redundant_within_R_cluster",
            })
    if len(representatives) < 2:
        representatives = comps
        for row in rows:
            row["selection_reason"] += ";fallback_all_active_components"
            row["representative_flag"] = int(row["component"] in representatives)
    return representatives, pd.DataFrame(rows), z


def subcomposition_ilr(centers: pd.DataFrame, representatives: list[str]) -> np.ndarray:
    x = centers[representatives].to_numpy(dtype=float)
    x = x / x.sum(axis=1, keepdims=True) * 100.0
    from scipy.linalg import helmert

    basis = helmert(len(representatives), full=False)
    return np.log(x / 100.0) @ basis.T


def cluster_labels(X: np.ndarray, k: int, method: str, seed: int) -> tuple[np.ndarray, float | None]:
    if method == "Ward":
        z = linkage(X, method="ward")
        labels = fcluster(z, t=k, criterion="maxclust") - 1
        return labels, None
    if method == "KMeans":
        model = KMeans(n_clusters=k, init="k-means++", n_init=100, max_iter=1000, random_state=seed)
        return model.fit_predict(X), None
    if method.startswith("GMM"):
        cov = method.split("_")[1]
        model = GaussianMixture(n_components=k, covariance_type=cov, reg_covar=1e-6, n_init=50, random_state=seed)
        labels = model.fit_predict(X)
        return labels, float(model.bic(X))
    raise ValueError(method)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, int]:
    counts = np.bincount(labels)
    min_size = int(counts.min()) if len(counts) else 0
    if len(np.unique(labels)) < 2 or min_size < 2:
        return np.nan, np.nan, np.nan, min_size
    return (
        float(silhouette_score(X, labels)),
        float(calinski_harabasz_score(X, labels)),
        float(davies_bouldin_score(X, labels)),
        min_size,
    )


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, candidate):
        if 0 <= a < k and 0 <= b < k:
            matrix[a, b] += 1
    r, c = linear_sum_assignment(-matrix)
    mapping = {cand: ref for ref, cand in zip(r, c)}
    return np.array([mapping.get(x, x) for x in candidate], dtype=int)


def prediction_strength_proxy(X: np.ndarray, full_labels: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Tibshirani--Walther style split-sample prediction strength.

    The test half is clustered independently.  The train-half centroids then
    predict test labels.  For every predicted cluster we compute the fraction
    of within-cluster test pairs that the independent test clustering also
    places together; the split score is the weakest cluster agreement.
    ``full_labels`` is kept in the signature for backward compatibility only.
    """
    del full_labels
    scores = []
    n = len(X)
    for _ in range(20):
        order = rng.permutation(n)
        train_idx, test_idx = order[: n // 2], order[n // 2 :]
        if len(train_idx) < k or len(test_idx) < k:
            continue
        try:
            seed_train = int(rng.integers(0, 2**31 - 1))
            seed_test = int(rng.integers(0, 2**31 - 1))
            train_fit = KMeans(n_clusters=k, n_init=50, random_state=seed_train).fit(X[train_idx])
            predicted = train_fit.predict(X[test_idx])
            independent = KMeans(n_clusters=k, n_init=50, random_state=seed_test).fit_predict(X[test_idx])
            cluster_scores = []
            for cluster_id in range(k):
                members = np.flatnonzero(predicted == cluster_id)
                if len(members) < 2:
                    continue
                same = independent[members][:, None] == independent[members][None, :]
                upper = np.triu_indices(len(members), k=1)
                cluster_scores.append(float(same[upper].mean()))
            if cluster_scores:
                scores.append(float(min(cluster_scores)))
        except Exception:
            continue
    return float(np.median(scores)) if scores else np.nan


def fit_type_full(points: pd.DataFrame, zero_c: float = 0.5, r_threshold: float = 0.5, fixed_active: list[str] | None = None) -> dict:
    prep = CompositionPreprocessor(zero_c=zero_c, fixed_active=fixed_active).fit(points[COMPONENTS], points["glass_type"])
    tr = prep.transform(points[COMPONENTS], points["glass_type"])
    centers = aggregate_artifact_centers(points, tr, prep, extra_cols=["glass_type", "surface_weathering"])
    detection = {c: float(points[c].notna().mean()) for c in prep.active_components_}
    reps, r_table, r_linkage = variable_selection(centers, detection, threshold=r_threshold)
    X = subcomposition_ilr(centers, reps)
    return {"preprocessor": prep, "centers": centers, "detection": detection, "representatives": reps, "r_table": r_table, "r_linkage": r_linkage, "X": X}


def main() -> None:
    _, t2 = load_known_data()
    valid = t2[t2["valid_sum_flag"]].copy()
    rng = np.random.default_rng(MASTER_SEED + 220)
    k_ranges = {"高钾": list(range(2, 6)), "铅钡": list(range(2, 7))}
    r_rows = []
    metric_rows = []
    boot_rows = []
    subclass_rows = []
    profile_rows = []
    consensus_rows = []
    sensitivity_rows = []
    plot_objects = {}

    for glass_type in ["高钾", "铅钡"]:
        points = valid[valid["glass_type"].eq(glass_type)].copy()
        full = fit_type_full(points)
        centers = full["centers"].reset_index(drop=True)
        X = full["X"]
        rtab = full["r_table"].copy()
        rtab.insert(0, "glass_type", glass_type)
        r_rows.append(rtab)
        candidate_k = [k for k in k_ranges[glass_type] if k <= len(centers) // 2]
        main_labels: dict[int, np.ndarray] = {}
        full_metric_index: dict[int, int] = {}
        for method in ["Ward", "KMeans", "GMM_diag", "GMM_tied", "GMM_full"]:
            for k in candidate_k:
                try:
                    labels, bic = cluster_labels(X, k, method, MASTER_SEED + k)
                    sil, ch, db, min_size = internal_metrics(X, labels)
                    feasible = int(min_size >= 3 and len(np.unique(labels)) == k)
                    row = {
                        "glass_type": glass_type, "feature_version": "+".join(full["representatives"]),
                        "method": method, "k": k, "silhouette": sil, "calinski_harabasz": ch,
                        "davies_bouldin": db, "bic": bic, "bootstrap_ari_median": np.nan,
                        "bootstrap_ari_p10": np.nan, "prediction_strength": np.nan,
                        "min_cluster_size": min_size, "feasible": feasible, "selected": 0,
                        "selection_reason": "pending_stability", "infeasible_reason": "" if feasible else "min_cluster_lt_3_or_missing_cluster",
                    }
                    if method == "Ward":
                        main_labels[k] = labels
                        row["prediction_strength"] = prediction_strength_proxy(X, labels, k, rng)
                        full_metric_index[k] = len(metric_rows)
                    metric_rows.append(row)
                except Exception as exc:
                    metric_rows.append({
                        "glass_type": glass_type, "feature_version": "+".join(full["representatives"]),
                        "method": method, "k": k, "silhouette": np.nan, "calinski_harabasz": np.nan,
                        "davies_bouldin": np.nan, "bic": np.nan, "bootstrap_ari_median": np.nan,
                        "bootstrap_ari_p10": np.nan, "prediction_strength": np.nan,
                        "min_cluster_size": 0, "feasible": 0, "selected": 0,
                        "selection_reason": "fit_failed", "infeasible_reason": str(exc),
                    })

        # Bootstrap一次重做预处理、R型筛选，并同时评价所有候选k
        ids = centers["artifact_id"].tolist()
        assignment_counts = {k: {aid: Counter() for aid in ids} for k in candidate_k}
        pair_joint = {k: defaultdict(int) for k in candidate_k}
        pair_same = {k: defaultdict(int) for k in candidate_k}
        for iteration in range(CONFIG["bootstrap_B"]):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            chunks = []
            sampled_original = []
            for draw, artifact_id in enumerate(sampled):
                chunk = points[points["artifact_id"].eq(artifact_id)].copy()
                chunk["original_artifact_id"] = artifact_id
                chunk["artifact_id"] = f"b{draw:03d}_{artifact_id}"
                chunks.append(chunk)
                sampled_original.append(artifact_id)
            boot = pd.concat(chunks, ignore_index=True)
            try:
                bfit = fit_type_full(boot, fixed_active=full["preprocessor"].active_components_)
                bX = bfit["X"]
                feature_jaccard = len(set(full["representatives"]) & set(bfit["representatives"])) / len(set(full["representatives"]) | set(bfit["representatives"]))
                for k in candidate_k:
                    try:
                        blabels, _ = cluster_labels(bX, k, "Ward", MASTER_SEED + iteration + k)
                        ref = np.array([main_labels[k][ids.index(a)] for a in sampled_original])
                        ari = float(adjusted_rand_score(ref, blabels))
                        aligned = align_labels(ref, blabels, k)
                        boot_rows.append({
                            "glass_type": glass_type, "method": "Ward", "k": k, "iteration": iteration,
                            "ari": ari, "selected_features": "+".join(bfit["representatives"]),
                            "feature_jaccard": feature_jaccard, "failed": 0, "failure_reason": "",
                        })
                        per_original = {}
                        for aid, lab in zip(sampled_original, aligned):
                            assignment_counts[k][aid][int(lab)] += 1
                            per_original.setdefault(aid, []).append(int(lab))
                        collapsed = {aid: Counter(labs).most_common(1)[0][0] for aid, labs in per_original.items()}
                        present = sorted(collapsed)
                        for i, a in enumerate(present):
                            for b in present[i:]:
                                key = (a, b)
                                pair_joint[k][key] += 1
                                pair_same[k][key] += int(collapsed[a] == collapsed[b])
                    except Exception as exc:
                        boot_rows.append({
                            "glass_type": glass_type, "method": "Ward", "k": k, "iteration": iteration,
                            "ari": np.nan, "selected_features": "+".join(bfit["representatives"]),
                            "feature_jaccard": feature_jaccard, "failed": 1, "failure_reason": str(exc),
                        })
            except Exception as exc:
                for k in candidate_k:
                    boot_rows.append({
                        "glass_type": glass_type, "method": "Ward", "k": k, "iteration": iteration,
                        "ari": np.nan, "selected_features": "", "feature_jaccard": np.nan,
                        "failed": 1, "failure_reason": str(exc),
                    })
            if (iteration + 1) % 200 == 0:
                logger.info("问题2.2 %s Bootstrap %d/%d", glass_type, iteration + 1, CONFIG["bootstrap_B"])

        boot_df_type = pd.DataFrame([r for r in boot_rows if r["glass_type"] == glass_type])
        # 把稳定性写回Ward候选行
        stable_candidates = []
        consensus_stats = {}
        for k in candidate_k:
            vals = boot_df_type[(boot_df_type["k"].eq(k)) & (boot_df_type["failed"].eq(0))]["ari"].dropna()
            med = float(vals.median()) if len(vals) else np.nan
            p10 = float(vals.quantile(0.10)) if len(vals) else np.nan
            idx = full_metric_index[k]
            metric_rows[idx]["bootstrap_ari_median"] = med
            metric_rows[idx]["bootstrap_ari_p10"] = p10
            # 共识矩阵及簇内/簇间均值
            pair_values = []
            for a in ids:
                for b in ids:
                    key = tuple(sorted((a, b)))
                    joint = pair_joint[k].get(key, 0)
                    same = pair_same[k].get(key, 0)
                    value = same / joint if joint else np.nan
                    pair_values.append((a, b, value, joint))
            main = main_labels[k]
            intra = [v for a, b, v, _ in pair_values if a != b and main[ids.index(a)] == main[ids.index(b)] and np.isfinite(v)]
            inter = [v for a, b, v, _ in pair_values if main[ids.index(a)] != main[ids.index(b)] and np.isfinite(v)]
            intra_mean = float(np.mean(intra)) if intra else np.nan
            inter_mean = float(np.mean(inter)) if inter else np.nan
            consensus_stats[k] = (intra_mean, inter_mean, pair_values)
            feasible = bool(metric_rows[idx]["feasible"])
            pred_strength = metric_rows[idx]["prediction_strength"]
            stable = feasible and med >= 0.75 and p10 >= 0.50 and intra_mean >= 0.75 and inter_mean <= 0.25 and (np.isnan(pred_strength) or pred_strength >= 0.80)
            if stable:
                stable_candidates.append(k)

        if stable_candidates:
            selected_k = min(stable_candidates)
            stable_supported = True
        else:
            ward_rows = [metric_rows[full_metric_index[k]] for k in candidate_k if metric_rows[full_metric_index[k]]["feasible"]]
            selected_k = int(max(ward_rows, key=lambda r: (np.nan_to_num(r["bootstrap_ari_median"], nan=-1), np.nan_to_num(r["silhouette"], nan=-1)))["k"])
            stable_supported = False
        for k in candidate_k:
            idx = full_metric_index[k]
            metric_rows[idx]["selected"] = int(k == selected_k and stable_supported)
            metric_rows[idx]["selection_reason"] = (
                "passed_all_stability_gates_smallest_k" if k == selected_k and stable_supported
                else "exploratory_best_but_no_stable_subclass" if k == selected_k
                else "not_selected"
            )

        # 最终/探索标签、归属频率和边界标记
        labels = main_labels[selected_k]
        for aid, label, n_points in zip(centers["artifact_id"], labels, centers["n_points"]):
            counts = assignment_counts[selected_k][aid]
            total = sum(counts.values())
            frequencies = sorted([v / total for v in counts.values()], reverse=True) if total else []
            top1 = frequencies[0] if frequencies else np.nan
            top2 = frequencies[1] if len(frequencies) > 1 else 0.0
            boundary = bool((np.isfinite(top1) and top1 < 0.80) or (np.isfinite(top1) and top1 - top2 < 0.20))
            subclass_rows.append({
                "artifact_id": aid, "glass_type": glass_type,
                "subclass_label": f"{glass_type}-亚类{label+1}" if stable_supported else "NO_STABLE_SUBCLASS",
                "exploratory_label": f"{glass_type}-探索组{label+1}",
                "selected_k": selected_k, "stable_subclass_supported": int(stable_supported),
                "assignment_frequency": top1, "second_best_frequency": top2,
                "boundary_flag": int(boundary), "n_points": int(n_points),
            })

        # 成分画像
        centers_profile = centers.copy()
        centers_profile["label"] = labels
        for label in sorted(np.unique(labels)):
            sub = centers_profile[centers_profile["label"].eq(label)]
            for comp in full["preprocessor"].active_components_:
                profile_rows.append({
                    "glass_type": glass_type,
                    "subclass_label": f"{glass_type}-亚类{label+1}" if stable_supported else f"{glass_type}-探索组{label+1}",
                    "component": comp, "center_pct": float(np.exp(np.log(sub[comp] / 100).mean())),
                    "median_pct": float(sub[comp].median()), "q25_pct": float(sub[comp].quantile(0.25)),
                    "q75_pct": float(sub[comp].quantile(0.75)), "n_artifacts": len(sub),
                    "stable_subclass_supported": int(stable_supported),
                })
        # center_pct重新闭合到100
        prof_idx = [i for i, r in enumerate(profile_rows) if r["glass_type"] == glass_type]
        for sublabel in set(profile_rows[i]["subclass_label"] for i in prof_idx):
            ids_prof = [i for i in prof_idx if profile_rows[i]["subclass_label"] == sublabel]
            total = sum(profile_rows[i]["center_pct"] for i in ids_prof)
            for i in ids_prof:
                profile_rows[i]["center_pct"] = profile_rows[i]["center_pct"] / total * 100.0

        # 选定k的共识矩阵
        for a, b, value, joint in consensus_stats[selected_k][2]:
            consensus_rows.append({
                "glass_type": glass_type, "artifact_i": a, "artifact_j": b,
                "co_cluster_frequency": value, "n_joint_resamples": joint,
                "k": selected_k, "stable_subclass_supported": int(stable_supported),
            })

        # 灵敏度：零替代、R切割、纳入无效点、输入扰动
        scenarios = [("main", points.copy(), 0.5, 0.5), ("zero_c_0.25", points.copy(), 0.25, 0.5), ("zero_c_0.75", points.copy(), 0.75, 0.5), ("r_threshold_0.4", points.copy(), 0.5, 0.4), ("r_threshold_0.6", points.copy(), 0.5, 0.6)]
        scenarios.append(("include_invalid_15_17", t2[t2["glass_type"].eq(glass_type)].copy(), 0.5, 0.5))
        for rate in [0.05, 0.10]:
            perturbed = points.copy()
            for ridx, idx in enumerate(perturbed.index):
                for cidx, comp in enumerate(COMPONENTS):
                    v = perturbed.at[idx, comp]
                    if pd.notna(v) and v > 0:
                        perturbed.at[idx, comp] = float(v) * (1 + (1 if (ridx + cidx) % 2 == 0 else -1) * rate)
            scenarios.append((f"componentwise_pm_{int(rate*100)}pct", perturbed, 0.5, 0.5))
        for scenario, spoints, c, rt in scenarios:
            try:
                sf = fit_type_full(spoints, zero_c=c, r_threshold=rt)
                if len(sf["X"]) != len(centers):
                    # 纳入无效点会多出文物，按共同文物比较
                    common_ids = [a for a in ids if a in set(sf["centers"]["artifact_id"])]
                    sx = sf["X"][[sf["centers"]["artifact_id"].tolist().index(a) for a in common_ids]]
                    slabels, _ = cluster_labels(sx, selected_k, "Ward", MASTER_SEED)
                    ref = np.array([labels[ids.index(a)] for a in common_ids])
                else:
                    slabels, _ = cluster_labels(sf["X"], selected_k, "Ward", MASTER_SEED)
                    ref = labels
                ari = float(adjusted_rand_score(ref, slabels))
                feature_jaccard = len(set(full["representatives"]) & set(sf["representatives"])) / len(set(full["representatives"]) | set(sf["representatives"]))
                sensitivity_rows.append({
                    "scenario": scenario, "glass_type": glass_type, "k": selected_k,
                    "ari_vs_main": ari, "feature_jaccard": feature_jaccard,
                    "conclusion": "stable" if ari >= 0.75 else "sensitive",
                })
            except Exception as exc:
                sensitivity_rows.append({
                    "scenario": scenario, "glass_type": glass_type, "k": selected_k,
                    "ari_vs_main": np.nan, "feature_jaccard": np.nan,
                    "conclusion": f"not_computable:{exc}",
                })

        plot_objects[glass_type] = {
            "full": full, "selected_k": selected_k, "labels": labels,
            "stable_supported": stable_supported,
            "q_linkage": linkage(X, method="ward"),
        }

    r_df = pd.concat(r_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    boot_df = pd.DataFrame(boot_rows)
    subclasses_df = pd.DataFrame(subclass_rows)
    profiles_df = pd.DataFrame(profile_rows)
    consensus_df = pd.DataFrame(consensus_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    save_csv(r_df, "results/02_q2/q2_2_r_clusters.csv")
    save_csv(metrics_df, "results/02_q2/q2_2_cluster_metrics.csv")
    save_csv(subclasses_df, "results/02_q2/q2_2_subclasses.csv")
    save_csv(profiles_df, "results/02_q2/q2_2_subclass_profiles.csv")
    save_csv(boot_df, "results/02_q2/q2_2_bootstrap_stability.csv")
    save_csv(consensus_df, "results/02_q2/q2_2_consensus_matrix.csv")
    save_csv(sensitivity_df, "results/02_q2/q2_2_sensitivity.csv")

    # 图1：R型变量树状图
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        obj = plot_objects[glass_type]["full"]
        comps = [c for c in COMPONENTS if c in obj["centers"].columns]
        dendrogram(obj["r_linkage"], labels=comps, orientation="right", ax=ax, color_threshold=0.5)
        ax.axvline(0.5, color="#E45756", ls="--", lw=1)
        ax.set_xlabel("1-|Spearman ρ|")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q2_2_r_variable_dendrograms.pdf", r_df)

    # 图2：Q型文物树状图
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        obj = plot_objects[glass_type]
        dendrogram(obj["q_linkage"], labels=obj["full"]["centers"]["artifact_id"].tolist(), orientation="right", ax=ax)
        ax.set_xlabel("Ward合并距离")
        suffix = "稳定亚类" if obj["stable_supported"] else "仅探索分组"
        ax.text(0.02, 0.98, f"{glass_type}，k={obj['selected_k']}（{suffix}）", transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q2_2_q_artifact_dendrograms.pdf", subclasses_df)

    # 图3：k与内部/稳定性指标
    ward_metrics = metrics_df[metrics_df["method"].eq("Ward")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric, ylabel in zip(axes.ravel(), ["silhouette", "calinski_harabasz", "davies_bouldin", "bootstrap_ari_median"], ["轮廓系数", "CH指数", "DB指数（越低越好）", "Bootstrap ARI中位数"]):
        for glass_type, color in [("高钾", "#4C78A8"), ("铅钡", "#E45756")]:
            sub = ward_metrics[ward_metrics["glass_type"].eq(glass_type)]
            ax.plot(sub["k"], sub[metric], marker="o", color=color, label=glass_type)
        ax.set_xlabel("候选亚类数k")
        ax.set_ylabel(ylabel)
        ax.legend()
    save_figure(fig, "q2_2_k_metrics_and_stability.pdf", ward_metrics)

    # 图4：共识矩阵
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = consensus_df[consensus_df["glass_type"].eq(glass_type)]
        matrix = sub.pivot(index="artifact_i", columns="artifact_j", values="co_cluster_frequency")
        order = subclasses_df[subclasses_df["glass_type"].eq(glass_type)].sort_values(["exploratory_label", "artifact_id"])["artifact_id"]
        matrix = matrix.reindex(index=order, columns=order)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), order, fontsize=6)
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top", color="white")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    save_figure(fig, "q2_2_consensus_matrices.pdf", consensus_df)

    # 图5：二维投影
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    projection_rows = []
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        obj = plot_objects[glass_type]
        coords = PCA(n_components=2).fit_transform(obj["full"]["X"])
        labels = obj["labels"]
        aids = obj["full"]["centers"]["artifact_id"].tolist()
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=45)
        for x, y, aid in zip(coords[:, 0], coords[:, 1], aids):
            ax.text(x, y, aid, fontsize=6, ha="center", va="bottom")
            projection_rows.append({"glass_type": glass_type, "artifact_id": aid, "pc1": x, "pc2": y, "label": int(labels[aids.index(aid)])})
        ax.set_xlabel("ILR-PCA第一轴")
        ax.set_ylabel("ILR-PCA第二轴")
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    save_figure(fig, "q2_2_subclass_projection.pdf", pd.DataFrame(projection_rows))

    # 图6：亚类/探索组画像
    top_profiles = profiles_df.groupby("component")["center_pct"].max().nlargest(10).index
    pp = profiles_df[profiles_df["component"].isin(top_profiles)].copy()
    pivot = pp.pivot(index="subclass_label", columns="component", values="center_pct").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("成分")
    fig.colorbar(im, ax=ax, label="Aitchison中心（%）")
    save_figure(fig, "q2_2_subclass_profiles_heatmap.pdf", profiles_df)

    logger.info("问题2.2完成：%s", subclasses_df.groupby("glass_type")["stable_subclass_supported"].max().to_dict())


if __name__ == "__main__":
    main()
