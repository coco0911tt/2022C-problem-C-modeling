from __future__ import annotations

import itertools
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from common import (
    COMPONENTS,
    COMPONENT_CN,
    CONFIG,
    MASTER_SEED,
    ROOT,
    CompositionPreprocessor,
    aggregate_artifact_centers,
    bh_fdr,
    load_known_data,
    save_csv,
    save_figure,
)


LOG_PATH = ROOT / "logs" / "40_q4_network.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def rho_p(logx: np.ndarray, j: int, k: int) -> float:
    a = logx[:, j]
    b = logx[:, k]
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    denom = va + vb
    return float(2 * np.cov(a, b, ddof=1)[0, 1] / denom) if denom > 0 else np.nan


def edge_values(centers: pd.DataFrame, nodes: list[str]) -> dict[tuple[str, str], tuple[float, float]]:
    logx = np.log(centers[nodes].to_numpy(dtype=float) / 100.0)
    out = {}
    for j, k in itertools.combinations(range(len(nodes)), 2):
        variation = float(np.var(logx[:, j] - logx[:, k], ddof=1))
        out[(nodes[j], nodes[k])] = (variation, rho_p(logx, j, k))
    return out


def fit_type(points: pd.DataFrame, zero_c: float = 0.5, fixed_active: list[str] | None = None) -> dict:
    prep = CompositionPreprocessor(zero_c=zero_c, fixed_active=fixed_active).fit(points[COMPONENTS], points["glass_type"])
    tr = prep.transform(points[COMPONENTS], points["glass_type"])
    centers = aggregate_artifact_centers(points, tr, prep, extra_cols=["glass_type", "surface_weathering"])
    detection = {c: float(points[c].notna().mean()) for c in COMPONENTS}
    nodes = [c for c in prep.active_components_ if detection[c] >= CONFIG["network_detection_rate"]]
    return {"preprocessor": prep, "transform": tr, "centers": centers, "detection": detection, "nodes": nodes}


def bootstrap_edges(points: pd.DataFrame, main: dict, rng: np.random.Generator) -> tuple[dict[tuple[str, str], list[float]], list[dict]]:
    ids = points["artifact_id"].unique()
    values = defaultdict(list)
    failures = []
    for iteration in range(CONFIG["bootstrap_B"]):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        chunks = []
        for draw, artifact_id in enumerate(sampled):
            chunk = points[points["artifact_id"].eq(artifact_id)].copy()
            chunk["original_artifact_id"] = chunk["artifact_id"]
            chunk["artifact_id"] = f"b{draw:03d}_{artifact_id}"
            chunks.append(chunk)
        boot = pd.concat(chunks, ignore_index=True)
        try:
            bfit = fit_type(boot, fixed_active=main["preprocessor"].active_components_)
            ev = edge_values(bfit["centers"], main["nodes"])
            for edge, (_, rho) in ev.items():
                values[edge].append(rho)
        except Exception as exc:
            failures.append({"iteration": iteration, "reason": str(exc)})
        if (iteration + 1) % 250 == 0:
            logger.info("问题4 %s Bootstrap %d/%d", points["glass_type"].iloc[0], iteration + 1, CONFIG["bootstrap_B"])
    return values, failures


def fast_permutation_deltas(points: pd.DataFrame, nodes: list[str], observed: dict, rng: np.random.Generator) -> tuple[dict, int]:
    # 文物标签置换；每次按置换组重算分组中位数、闭合和文物中心。
    artifacts = points[["artifact_id", "glass_type"]].drop_duplicates("artifact_id").reset_index(drop=True)
    artifact_ids = artifacts["artifact_id"].tolist()
    artifact_index = {a: i for i, a in enumerate(artifact_ids)}
    point_art = points["artifact_id"].map(artifact_index).to_numpy(dtype=int)
    labels = artifacts["glass_type"].to_numpy()
    X_raw = points[nodes].to_numpy(dtype=float)
    global_median = np.nanmedian(X_raw, axis=0)
    positive_min = np.array([np.nanmin(X_raw[:, j][X_raw[:, j] > 0]) for j in range(len(nodes))])
    delta = CONFIG["zero_c"] * positive_min
    pairs = list(itertools.combinations(range(len(nodes)), 2))
    pair_names = [(nodes[j], nodes[k]) for j, k in pairs]
    observed_abs = np.array([abs(observed[e]) for e in pair_names])
    exceed = np.zeros(len(pairs), dtype=int)
    direction = np.zeros(len(pairs), dtype=int)
    successful = 0
    B = int(CONFIG["permutation_B"])
    for iteration in range(B):
        perm_labels_art = rng.permutation(labels)
        perm_labels_point = perm_labels_art[point_art]
        X = X_raw.copy()
        for group in ["高钾", "铅钡"]:
            mask = perm_labels_point == group
            for j in range(len(nodes)):
                obs = X_raw[mask, j]
                obs = obs[np.isfinite(obs)]
                med = np.median(obs) if len(obs) >= CONFIG["min_group_observed"] else global_median[j]
                missing = mask & ~np.isfinite(X[:, j])
                X[missing, j] = med
        for j in range(len(nodes)):
            X[X[:, j] == 0, j] = delta[j]
        if not np.isfinite(X).all() or (X <= 0).any():
            continue
        X = X / X.sum(axis=1, keepdims=True)
        clr = np.log(X) - np.log(X).mean(axis=1, keepdims=True)
        art_clr = np.zeros((len(artifact_ids), len(nodes)))
        for aidx in range(len(artifact_ids)):
            art_clr[aidx] = clr[point_art == aidx].mean(axis=0)
        group_values = {}
        for group in ["高钾", "铅钡"]:
            sub = art_clr[perm_labels_art == group]
            vals = []
            for j, k in pairs:
                vals.append(rho_p(sub, j, k))
            group_values[group] = np.asarray(vals)
        diff = group_values["高钾"] - group_values["铅钡"]
        if not np.isfinite(diff).all():
            continue
        exceed += np.abs(diff) >= observed_abs - 1e-12
        obs_sign = np.sign([observed[e] for e in pair_names])
        direction += np.sign(diff) == obs_sign
        successful += 1
    return {
        edge: {
            "p": float((exceed[i] + 1) / (successful + 1)) if successful else np.nan,
            "permutation_direction_frequency": float(direction[i] / successful) if successful else np.nan,
        }
        for i, edge in enumerate(pair_names)
    }, successful


def partial_correlation(centers: pd.DataFrame, nodes: list[str]) -> tuple[np.ndarray, float]:
    x = np.log(centers[nodes].to_numpy(dtype=float) / 100.0)
    clr = x - x.mean(axis=1, keepdims=True)
    model = LedoitWolf().fit(clr)
    precision = model.precision_
    denom = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / denom
    np.fill_diagonal(partial, 1.0)
    return partial, float(model.shrinkage_)


def weather_adjusted_edges(centers: pd.DataFrame, nodes: list[str]) -> dict[tuple[str, str], float]:
    logx = np.log(centers[nodes].to_numpy(dtype=float) / 100.0)
    w = centers["surface_weathering"].map({"无风化": 0.0, "风化": 1.0}).to_numpy()
    design = np.column_stack([np.ones(len(w)), w])
    coef, *_ = np.linalg.lstsq(design, logx, rcond=None)
    resid = logx - design @ coef
    out = {}
    for j, k in itertools.combinations(range(len(nodes)), 2):
        out[(nodes[j], nodes[k])] = rho_p(resid, j, k)
    return out


def core_edge_set(edge_df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, edge_df.loc[edge_df["core_edge"].eq(1), ["component_j", "component_k"]].to_numpy()))


def main() -> None:
    _, t2 = load_known_data()
    valid = t2[t2["valid_sum_flag"]].copy()
    rng = np.random.default_rng(MASTER_SEED + 400)
    fits = {}
    node_rows = []
    edge_rows = []
    weather_rows = []
    partial_rows = []
    stability_rows = []
    sensitivity_rows = []
    failure_rows = []
    boot_by_type = {}

    for glass_type in ["高钾", "铅钡"]:
        points = valid[valid["glass_type"].eq(glass_type)].copy()
        fit = fit_type(points)
        fits[glass_type] = fit
        for comp in COMPONENTS:
            node_rows.append({
                "glass_type": glass_type, "component": comp,
                "detection_rate": fit["detection"][comp], "missing_rate": 1 - fit["detection"][comp],
                "active_main": int(comp in fit["nodes"]),
                "exclusion_reason": "" if comp in fit["nodes"] else ("lt5_observed" if comp not in fit["preprocessor"].active_components_ else "detection_rate_lt_0.30"),
            })
        main_edges = edge_values(fit["centers"], fit["nodes"])
        boot_values, failures = bootstrap_edges(points, fit, rng)
        boot_by_type[glass_type] = boot_values
        for failure in failures:
            failure_rows.append({"glass_type": glass_type, **failure})
        for edge, (variation, rho) in main_edges.items():
            vals = np.asarray(boot_values[edge], dtype=float)
            ci_low, ci_high = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
            selection = float(np.mean(np.abs(vals) >= CONFIG["network_edge_threshold"])) if len(vals) else np.nan
            direction = float(np.mean(np.sign(vals) == np.sign(rho))) if len(vals) else np.nan
            core = bool(abs(rho) >= CONFIG["network_edge_threshold"] and selection >= CONFIG["network_selection_frequency"] and not (ci_low <= 0 <= ci_high))
            edge_rows.append({
                "glass_type": glass_type, "component_j": edge[0], "component_k": edge[1],
                "variation_logratio": variation, "rho_p": rho, "ci_low": ci_low, "ci_high": ci_high,
                "selection_frequency": selection, "direction_frequency": direction, "core_edge": int(core),
            })
        adjusted = weather_adjusted_edges(fit["centers"], fit["nodes"])
        for edge, adj in adjusted.items():
            raw = main_edges[edge][1]
            weather_rows.append({
                "glass_type": glass_type, "component_j": edge[0], "component_k": edge[1],
                "rho_raw": raw, "rho_adjusted": adj,
                "direction_same": int(np.sign(raw) == np.sign(adj)),
                "selection_frequency_adjusted": np.nan,
            })
        partial, shrinkage = partial_correlation(fit["centers"], fit["nodes"])
        partial_boot = defaultdict(list)
        X_center = fit["centers"]
        for _ in range(CONFIG["bootstrap_B"]):
            idx = rng.integers(0, len(X_center), len(X_center))
            try:
                pb, _ = partial_correlation(X_center.iloc[idx].reset_index(drop=True), fit["nodes"])
                for j, k in itertools.combinations(range(len(fit["nodes"])), 2):
                    partial_boot[(j, k)].append(pb[j, k])
            except Exception:
                continue
        for j, k in itertools.combinations(range(len(fit["nodes"])), 2):
            vals = np.asarray(partial_boot[(j, k)])
            lo, hi = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
            partial_rows.append({
                "glass_type": glass_type, "component_j": fit["nodes"][j], "component_k": fit["nodes"][k],
                "partial_corr": float(partial[j, k]), "shrinkage_parameter": shrinkage,
                "ci_low": lo, "ci_high": hi,
            })

    edges_df = pd.DataFrame(edge_rows)
    common_nodes = [c for c in COMPONENTS if c in fits["高钾"]["nodes"] and c in fits["铅钡"]["nodes"]]
    high_map = {(r.component_j, r.component_k): r.rho_p for r in edges_df[edges_df["glass_type"].eq("高钾")].itertuples()}
    lead_map = {(r.component_j, r.component_k): r.rho_p for r in edges_df[edges_df["glass_type"].eq("铅钡")].itertuples()}
    observed_delta = {}
    for edge in itertools.combinations(common_nodes, 2):
        observed_delta[edge] = high_map[edge] - lead_map[edge]
    perm_results, successful_permutations = fast_permutation_deltas(valid, common_nodes, observed_delta, rng)
    diff_rows = []
    for edge, delta in observed_delta.items():
        hv = np.asarray(boot_by_type["高钾"][edge])
        lv = np.asarray(boot_by_type["铅钡"][edge])
        n = min(len(hv), len(lv))
        delta_boot = hv[:n] - lv[:n]
        direction_freq = float(np.mean(np.sign(delta_boot) == np.sign(delta))) if n else np.nan
        p = perm_results[edge]["p"]
        diff_rows.append({
            "component_j": edge[0], "component_k": edge[1],
            "rho_high_k": high_map[edge], "rho_lead_barium": lead_map[edge], "delta": delta,
            "delta_ci_low": float(np.quantile(delta_boot, 0.025)) if n else np.nan,
            "delta_ci_high": float(np.quantile(delta_boot, 0.975)) if n else np.nan,
            "permutation_p": p, "p_fdr": np.nan,
            "effect_pass": int(abs(delta) >= CONFIG["network_delta_threshold"]),
            "direction_frequency": direction_freq, "core_differential_edge": 0,
            "permutation_success": successful_permutations,
        })
    diff_df = pd.DataFrame(diff_rows)
    diff_df["p_fdr"] = bh_fdr(diff_df["permutation_p"])
    diff_df["core_differential_edge"] = (
        (diff_df["p_fdr"] < CONFIG["fdr_q"])
        & (diff_df["delta"].abs() >= CONFIG["network_delta_threshold"])
        & (diff_df["direction_frequency"] >= CONFIG["network_direction_frequency"])
    ).astype(int)

    # 网络层Bootstrap Jaccard稳定性
    for glass_type in ["高钾", "铅钡"]:
        sub_edges = edges_df[edges_df["glass_type"].eq(glass_type)]
        main_set = core_edge_set(sub_edges)
        nodes = fits[glass_type]["nodes"]
        pairs = list(itertools.combinations(nodes, 2))
        for iteration in range(CONFIG["bootstrap_B"]):
            eset = {edge for edge in pairs if iteration < len(boot_by_type[glass_type][edge]) and abs(boot_by_type[glass_type][edge][iteration]) >= CONFIG["network_edge_threshold"]}
            union = main_set | eset
            jaccard = len(main_set & eset) / len(union) if union else 1.0
            stability_rows.append({
                "scenario_or_iteration": iteration, "glass_type": glass_type,
                "edge_count": len(eset), "density": len(eset) / max(1, len(pairs)),
                "jaccard_vs_main": jaccard, "failed": 0, "failure_reason": "",
            })
        # Graphical Lasso切换条件审计
        n = len(fits[glass_type]["centers"])
        p = len(nodes)
        stability_rows.append({
            "scenario_or_iteration": "graphical_lasso_gate", "glass_type": glass_type,
            "edge_count": np.nan, "density": np.nan, "jaccard_vs_main": np.nan,
            "failed": int(n / max(p, 1) < 5),
            "failure_reason": f"disabled_n_over_p={n/max(p,1):.3f}_lt_5" if n / max(p, 1) < 5 else "gate_passed_not_selected_as_main",
        })

    # 灵敏度：零替代、检出阈值、边阈值、纳入无效点、风化调整
    main_core = {g: core_edge_set(edges_df[edges_df["glass_type"].eq(g)]) for g in ["高钾", "铅钡"]}
    for glass_type in ["高钾", "铅钡"]:
        points_main = valid[valid["glass_type"].eq(glass_type)].copy()
        scenarios = [("zero_c_0.25", points_main, 0.25, 0.30, 0.50), ("zero_c_0.75", points_main, 0.75, 0.30, 0.50), ("detection_0.20", points_main, 0.5, 0.20, 0.50), ("detection_0.40", points_main, 0.5, 0.40, 0.50), ("edge_threshold_0.40", points_main, 0.5, 0.30, 0.40), ("edge_threshold_0.60", points_main, 0.5, 0.30, 0.60), ("include_invalid_15_17", t2[t2["glass_type"].eq(glass_type)].copy(), 0.5, 0.30, 0.50)]
        for scenario, spoints, c, det, eth in scenarios:
            try:
                sf = fit_type(spoints, zero_c=c)
                nodes = [node for node in sf["preprocessor"].active_components_ if sf["detection"][node] >= det]
                ev = edge_values(sf["centers"], nodes)
                eset = {edge for edge, (_, rho) in ev.items() if abs(rho) >= eth}
                union = main_core[glass_type] | eset
                jaccard = len(main_core[glass_type] & eset) / len(union) if union else 1.0
                sign_common = []
                main_rho = {(r.component_j, r.component_k): r.rho_p for r in edges_df[edges_df["glass_type"].eq(glass_type)].itertuples()}
                for edge in set(main_rho) & set(ev):
                    sign_common.append(np.sign(main_rho[edge]) == np.sign(ev[edge][1]))
                sensitivity_rows.append({
                    "scenario": scenario, "edge_type": glass_type,
                    "retained_core_edges": len(main_core[glass_type] & eset),
                    "edge_jaccard": jaccard, "sign_agreement": float(np.mean(sign_common)) if sign_common else np.nan,
                    "conclusion_grade": "stable" if jaccard >= 0.70 else "sensitive",
                })
            except Exception as exc:
                sensitivity_rows.append({
                    "scenario": scenario, "edge_type": glass_type,
                    "retained_core_edges": np.nan, "edge_jaccard": np.nan, "sign_agreement": np.nan,
                    "conclusion_grade": f"not_computable:{exc}",
                })

    nodes_df = pd.DataFrame(node_rows)
    weather_df = pd.DataFrame(weather_rows)
    partial_df = pd.DataFrame(partial_rows)
    stability_df = pd.DataFrame(stability_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    failures_df = pd.DataFrame(failure_rows, columns=["glass_type", "iteration", "reason"])
    save_csv(nodes_df, "results/04_q4/q4_nodes.csv")
    save_csv(edges_df, "results/04_q4/q4_edges_by_type.csv")
    save_csv(diff_df, "results/04_q4/q4_differential_edges.csv")
    save_csv(weather_df, "results/04_q4/q4_weather_adjusted_edges.csv")
    save_csv(partial_df, "results/04_q4/q4_partial_correlation_baseline.csv")
    save_csv(stability_df, "results/04_q4/q4_network_stability.csv")
    save_csv(sensitivity_df, "results/04_q4/q4_sensitivity.csv")
    save_csv(failures_df, "results/04_q4/q4_bootstrap_failures.csv")

    # 图1：两类核心比例性网络，统一布局
    union_nodes = sorted(set(fits["高钾"]["nodes"]) | set(fits["铅钡"]["nodes"]), key=COMPONENTS.index)
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(union_nodes)
    for _, r in edges_df.iterrows():
        if r["core_edge"]:
            layout_graph.add_edge(r["component_j"], r["component_k"])
    pos = nx.spring_layout(layout_graph, seed=MASTER_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        sub = edges_df[(edges_df["glass_type"].eq(glass_type)) & (edges_df["core_edge"].eq(1))]
        g = nx.Graph()
        g.add_nodes_from(fits[glass_type]["nodes"])
        for _, r in sub.iterrows():
            g.add_edge(r["component_j"], r["component_k"], weight=abs(r["rho_p"]), sign=np.sign(r["rho_p"]))
        nx.draw_networkx_nodes(g, pos, node_color="#D9EAF7", edgecolors="#2A6F97", node_size=700, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
        pos_edges = [(u, v) for u, v, d in g.edges(data=True) if d["sign"] >= 0]
        neg_edges = [(u, v) for u, v, d in g.edges(data=True) if d["sign"] < 0]
        nx.draw_networkx_edges(g, pos, edgelist=pos_edges, edge_color="#E45756", width=[1 + 3 * g[u][v]["weight"] for u, v in pos_edges], ax=ax)
        nx.draw_networkx_edges(g, pos, edgelist=neg_edges, edge_color="#4C78A8", style="dashed", width=[1 + 3 * g[u][v]["weight"] for u, v in neg_edges], ax=ax)
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
        ax.axis("off")
    save_figure(fig, "q4_core_proportionality_networks.pdf", edges_df)

    # 图2：差异网络
    core_diff = diff_df[diff_df["core_differential_edge"].eq(1)]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    g = nx.Graph()
    g.add_nodes_from(common_nodes)
    for _, r in core_diff.iterrows():
        g.add_edge(r["component_j"], r["component_k"], weight=abs(r["delta"]), sign=np.sign(r["delta"]))
    nx.draw_networkx_nodes(g, pos, node_color="#F4E3C1", edgecolors="#8C2D04", node_size=700, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    if len(g.edges):
        nx.draw_networkx_edges(g, pos, edge_color=["#E45756" if g[u][v]["sign"] > 0 else "#4C78A8" for u, v in g.edges], width=[1 + 5 * g[u][v]["weight"] for u, v in g.edges], ax=ax)
    ax.axis("off")
    save_figure(fig, "q4_core_differential_network.pdf", diff_df)

    # 图3：比例性矩阵热图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    heat_rows = []
    for ax, glass_type in zip(axes, ["高钾", "铅钡"]):
        nodes = fits[glass_type]["nodes"]
        mat = np.eye(len(nodes))
        sub = edges_df[edges_df["glass_type"].eq(glass_type)]
        for _, r in sub.iterrows():
            i, j = nodes.index(r["component_j"]), nodes.index(r["component_k"])
            mat[i, j] = mat[j, i] = r["rho_p"]
            heat_rows.append({"glass_type": glass_type, "component_i": r["component_j"], "component_j": r["component_k"], "rho_p": r["rho_p"]})
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(nodes)), nodes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(nodes)), nodes, fontsize=7)
        ax.text(0.02, 0.98, glass_type, transform=ax.transAxes, ha="left", va="top")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, label="比例性系数 ρp")
    save_figure(fig, "q4_proportionality_heatmaps.pdf", pd.DataFrame(heat_rows))

    # 图4：选择频率与效应
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for glass_type, color in [("高钾", "#4C78A8"), ("铅钡", "#E45756")]:
        sub = edges_df[edges_df["glass_type"].eq(glass_type)]
        ax.scatter(sub["rho_p"].abs(), sub["selection_frequency"], label=glass_type, color=color, alpha=0.75)
    ax.axvline(CONFIG["network_edge_threshold"], color="0.5", ls="--")
    ax.axhline(CONFIG["network_selection_frequency"], color="0.5", ls="--")
    ax.set_xlabel("|ρp|")
    ax.set_ylabel("Bootstrap选择频率")
    ax.legend()
    save_figure(fig, "q4_effect_vs_selection_frequency.pdf", edges_df)

    # 图5：风化调整前后
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for glass_type, color in [("高钾", "#4C78A8"), ("铅钡", "#E45756")]:
        sub = weather_df[weather_df["glass_type"].eq(glass_type)]
        ax.scatter(sub["rho_raw"], sub["rho_adjusted"], label=glass_type, color=color, alpha=0.7)
    ax.plot([-1, 1], [-1, 1], color="0.5", ls="--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("未调整ρp")
    ax.set_ylabel("风化残差调整后ρp")
    ax.legend()
    save_figure(fig, "q4_weathering_adjustment_comparison.pdf", weather_df)

    logger.info("问题4完成：permutations=%d, core_diff_edges=%d", successful_permutations, int(diff_df["core_differential_edge"].sum()))


if __name__ == "__main__":
    main()
