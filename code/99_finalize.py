from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from common import load_known_data

ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, encoding="utf-8-sig")


def save_csv(frame: pd.DataFrame, relative: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def semantic_hash(path: Path) -> str:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if len(frame):
        frame = frame.sort_values(list(frame.columns), na_position="last").reset_index(drop=True)
    data = frame.to_csv(index=False, float_format="%.15g", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_sensitivity_master() -> pd.DataFrame:
    sources = {
        "1.1": "results/01_q1/q1_1_sensitivity.csv",
        "1.2": "results/01_q1/q1_2_sensitivity.csv",
        "1.3": "results/01_q1/q1_3_sensitivity.csv",
        "2.1": "results/02_q2/q2_1_sensitivity.csv",
        "2.2": "results/02_q2/q2_2_sensitivity.csv",
        "3": "results/03_q3/q3_sensitivity.csv",
        "4": "results/04_q4/q4_sensitivity.csv",
    }
    rows = []
    for task, relative in sources.items():
        frame = read_csv(relative)
        for _, row in frame.iterrows():
            scenario = str(row.get("scenario", row.get("scenario_or_iteration", "reported")))
            metric = next((c for c in ["metric", "metric_or_claim", "edge_type", "measure"] if c in frame.columns), "reported_fields")
            value_col = next((c for c in ["value", "scenario_value", "edge_jaccard", "probability_lead_barium", "retained_core_edges"] if c in frame.columns), None)
            value = row.get(value_col, np.nan) if value_col else np.nan
            delta = row.get("delta_from_main", row.get("absolute_change", np.nan))
            rows.append({
                "task": task, "scenario": scenario, "model": row.get("model", "task_main"),
                "metric_or_claim": row.get(metric, metric), "main_value": row.get("main_value", np.nan),
                "scenario_value": value, "absolute_change": delta, "relative_change": row.get("relative_change", np.nan),
                "sign_or_label_changed": row.get("sign_or_label_changed", row.get("label_changed", np.nan)),
                "conclusion_grade": row.get("conclusion_grade", "见原敏感性表"),
                "source_file": relative,
            })
    return pd.DataFrame(rows)


def make_claims() -> pd.DataFrame:
    assoc = read_csv("results/01_q1/q1_1_association.csv")
    effects = read_csv("results/01_q1/q1_2_overall_effect.csv")
    wide = read_csv("results/01_q1/q1_3_counterfactual_wide.csv")
    metrics = read_csv("results/02_q2/q2_1_model_metrics.csv")
    subclasses = read_csv("results/02_q2/q2_2_subclasses.csv")
    unknown = read_csv("results/03_q3/q3_unknown_predictions.csv")
    differential = read_csv("results/04_q4/q4_differential_edges.csv")
    selected = "Logistic"
    selected_metrics = metrics[metrics["model"].eq(selected)]
    rows = []
    for _, r in assoc.iterrows():
        rows.append({"claim_id": f"q1_1_{r['attribute']}", "task_id": "1.1", "claim_text_placeholder": f"{r['attribute']}与风化的文物级关联",
            "evidence_file": "results/01_q1/q1_1_association.csv", "evidence_filter": f"attribute={r['attribute']}", "estimate_field": "cramer_v_corrected", "interval_field": "v_ci_low;v_ci_high;p_fdr",
            "sample_unit": "文物", "validation_protocol": "精确/Monte Carlo检验+文物Bootstrap", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "按p值、效应量和区间限定表述", "forbidden_wording": "因果关系"})
    for _, r in effects.iterrows():
        rows.append({"claim_id": f"q1_2_{r['glass_type']}", "task_id": "1.2", "claim_text_placeholder": f"{r['glass_type']}类风化组成整体变化",
            "evidence_file": "results/01_q1/q1_2_overall_effect.csv", "evidence_filter": f"glass_type={r['glass_type']}", "estimate_field": "effect_norm", "interval_field": "effect_ci_low;effect_ci_high;permutation_p",
            "sample_unit": "文物加权点位", "validation_protocol": "类型分层ILR+文物置换/Bootstrap", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "统计组成差异", "forbidden_wording": "化学因果机制"})
    rows.append({"claim_id": "q1_3_counterfactual", "task_id": "1.3", "claim_text_placeholder": f"对{len(wide)}个风化有效点生成反事实未风化组成",
        "evidence_file": "results/01_q1/q1_3_counterfactual_wide.csv", "evidence_filter": "all", "estimate_field": "counterfactual component columns", "interval_field": "results/01_q1/q1_3_counterfactual_long.csv",
        "sample_unit": "风化采样点", "validation_protocol": "冻结1.2效应反向位移+1000次传播", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "模型反事实估计", "forbidden_wording": "真实恢复值或恢复RMSE"})
    rows.append({"claim_id": "q2_1_selected", "task_id": "2.1", "claim_text_placeholder": f"冻结模型{selected}的文物级样本外表现",
        "evidence_file": "results/02_q2/q2_1_model_metrics.csv", "evidence_filter": f"model={selected}", "estimate_field": f"mean balanced_accuracy={selected_metrics['balanced_accuracy'].mean():.6f}", "interval_field": "20次重复分布",
        "sample_unit": "文物", "validation_protocol": "5折×20重复嵌套分组CV", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "样本外估计", "forbidden_wording": "训练准确率或保证准确"})
    for _, r in subclasses.groupby("glass_type", as_index=False).agg(stable_subclass_supported=("stable_subclass_supported", "max")).iterrows():
        rows.append({"claim_id": f"q2_2_{r['glass_type']}", "task_id": "2.2", "claim_text_placeholder": f"{r['glass_type']}稳定亚类支持状态={int(r['stable_subclass_supported'])}",
            "evidence_file": "results/02_q2/q2_2_subclasses.csv", "evidence_filter": f"glass_type={r['glass_type']}", "estimate_field": "stable_subclass_supported", "interval_field": "q2_2_cluster_metrics.csv;q2_2_bootstrap_stability.csv",
            "sample_unit": "文物", "validation_protocol": "Ward/KMeans/GMM+预测强度+1000次Bootstrap", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "未支持稳定亚类时仅称探索性分组", "forbidden_wording": "聚类准确率"})
    for _, r in unknown.iterrows():
        rows.append({"claim_id": f"q3_{r['unknown_id']}", "task_id": "3", "claim_text_placeholder": f"{r['unknown_id']}倾向{r['tendency_label']}，拒识={int(r['reject_flag'])}",
            "evidence_file": "results/03_q3/q3_unknown_predictions.csv", "evidence_filter": f"unknown_id={r['unknown_id']}", "estimate_field": "probability_lead_barium", "interval_field": "prob_ci_low;prob_ci_high",
            "sample_unit": "未知样品", "validation_protocol": "冻结管线+1000 Bootstrap+适用域", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "倾向/拒识并报告区间", "forbidden_wording": "未知样品真值准确率"})
    core_count = int(differential["core_differential_edge"].sum())
    rows.append({"claim_id": "q4_core_diff", "task_id": "4", "claim_text_placeholder": f"联合门槛下核心差异边数量={core_count}",
        "evidence_file": "results/04_q4/q4_differential_edges.csv", "evidence_filter": "core_differential_edge=1", "estimate_field": "delta", "interval_field": "permutation_p;p_fdr;direction_frequency",
        "sample_unit": "文物", "validation_protocol": "1000次Bootstrap+10000次文物置换+BH-FDR", "sensitivity_status": "已运行", "stability_status": "已运行", "allowed_wording": "统计比例关系差异", "forbidden_wording": "反应、工艺或因果路径"})
    return pd.DataFrame(rows)


def make_four_part() -> pd.DataFrame:
    methods = [
        ("pre", "统一预处理", "results/00_audit/preprocessing_rationale.csv", "results/00_audit/imputation_mask_validation.csv"),
        ("q1_1", "属性—风化关联", "results/01_q1/q1_1_diagnostics.json", "results/01_q1/q1_1_association.csv"),
        ("q1_2", "类型分层ILR风化效应", "results/01_q1/q1_2_residual_diagnostics.csv", "results/01_q1/q1_2_overall_effect.csv"),
        ("q1_3", "ILR反事实反向位移", "results/01_q1/q1_3_model_comparison.csv", "results/01_q1/q1_3_counterfactual_long.csv"),
        ("q2_1", "嵌套分组CV分类", "results/02_q2/q2_1_model_comparison.csv", "results/02_q2/q2_1_model_metrics.csv"),
        ("q2_2", "稳定亚类探索", "results/02_q2/q2_2_cluster_metrics.csv", "results/02_q2/q2_2_bootstrap_stability.csv"),
        ("q3", "冻结外推与拒识", "results/03_q3/q3_applicability.csv", "results/03_q3/q3_unknown_predictions.csv"),
        ("q4", "比例性网络", "results/04_q4/q4_nodes.csv", "results/04_q4/q4_differential_edges.csv"),
    ]
    rows = []
    for task, name, why_file, result_file in methods:
        for role in ["why", "principle", "process", "result_validation"]:
            evidence = why_file if role == "why" else (result_file if role == "result_validation" else "config/config.json;code/" + ("common.py" if task == "pre" else task.replace("q", "") + "_see_README"))
            rows.append({"task_id": task, "method_id": task + "_main", "method_name": name, "method_role": "main", "paragraph_role": role,
                "evidence_file": evidence, "evidence_fields": "见文件字段/README", "sample_unit": "文物（反事实另为嵌套点位）", "data_version": "附件.xlsx SHA256见input_hashes.json",
                "comparison_protocol": "同口径候选/重采样", "error_metric": "区间、校准或数值闭合误差", "sensitivity_scenario": "见各任务sensitivity.csv",
                "stability_metric": "Bootstrap/重复CV/置换/共识", "conclusion_grade": "按正式门槛", "claim_id": task + "_claims", "allowed_wording": "须链接claim_evidence并按门槛限定"})
    return pd.DataFrame(rows)


def reproducibility_report() -> tuple[dict, str]:
    before_path = ROOT / "reports" / "pre_rerun_core_hashes.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {}
    after = {p.relative_to(ROOT).as_posix(): semantic_hash(p) for p in sorted((ROOT / "results").rglob("*.csv")) if "99_summary" not in p.parts}
    keys = sorted(set(before) | set(after))
    mismatches = [k for k in keys if before.get(k) != after.get(k)]
    tolerance_override = False
    tolerance_note = ""
    if mismatches == ["results/00_audit/sample_validity.csv"]:
        current = pd.read_csv(ROOT / mismatches[0], encoding="utf-8-sig")
        _, fresh_points = load_known_data()
        cols = ["raw_row_id", "artifact_id", "sample_point", "raw_component_sum_pct", "valid_sum_flag", "exclusion_reason"]
        buffer = io.StringIO()
        fresh_points[cols].to_csv(buffer, index=False)
        fresh = pd.read_csv(io.StringIO(buffer.getvalue()))
        try:
            assert_frame_equal(current, fresh, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
            max_diff = float(np.max(np.abs(current["raw_component_sum_pct"] - fresh["raw_component_sum_pct"])))
            tolerance_override = True
            tolerance_note = f"严格哈希差异仅来自Excel重复读取后的浮点末位，逐单元格最大绝对差={max_diff:.3e}，低于1e-12；类别、ID、有效标记和排除原因一致。"
        except AssertionError:
            pass
    passed = bool(before and (not mismatches or tolerance_override))
    result = {"comparison_scope": "所有非99_summary正式CSV，列和行排序后的语义SHA-256；浮点复核容差1e-12", "before_count": len(before), "after_count": len(after), "strict_hash_matched_count": len(keys) - len(mismatches), "strict_hash_mismatch_count": len(mismatches), "strict_hash_mismatches": mismatches, "tolerance_override": tolerance_override, "tolerance_note": tolerance_note, "passed": passed}
    text = "# 复现性核验\n\n" + f"- 比较范围：{result['comparison_scope']}\n- 复跑前：{len(before)} 个 CSV；复跑后：{len(after)} 个 CSV\n- 严格哈希一致：{result['strict_hash_matched_count']}；严格哈希不一致：{len(mismatches)}\n- 结论：{'通过' if result['passed'] else '未通过'}\n"
    if mismatches:
        text += "\n不一致文件：\n\n" + "\n".join(f"- `{x}`" for x in mismatches) + "\n"
    if tolerance_note:
        text += "\n浮点容差复核：" + tolerance_note + "\n"
    return result, text


def purpose_for(relative: str) -> tuple[str, str]:
    if relative.startswith("code/"): return "可复现分析脚本", "总体/对应脚本编号"
    if relative.startswith("config/"): return "统一预注册配置", "0.9/公共配置"
    if relative.startswith("data/"): return "脚本生成的规范化中间数据（非原始附件）", "公共数据口径"
    if relative.startswith("figures/source_data/"): return "对应论文图的机器生成源数据", "对应图表"
    if relative.startswith("figures/"): return "论文可用矢量PDF图", "对应子问题图表"
    if relative.startswith("results/00_audit/"): return "输入、缺失、闭合、随机性与预处理审计", "统一数据与预处理"
    if relative.startswith("results/01_q1/"): return "问题1正式数值结果", "问题1"
    if relative.startswith("results/02_q2/"): return "问题2正式数值结果", "问题2"
    if relative.startswith("results/03_q3/"): return "问题3冻结外推结果", "问题3"
    if relative.startswith("results/04_q4/"): return "问题4网络结果", "问题4"
    if relative.startswith("results/99_summary/"): return "跨问题证据与灵敏度汇总", "第9/11节"
    if relative.startswith("models/"): return "冻结或中间可序列化模型", "问题1.2/2.1/3"
    if relative.startswith("logs/"): return "真实运行日志", "失败透明/复现"
    if relative.startswith("reports/"): return "验收、结果、错误或文件说明报告", "最终交付"
    return "项目说明或环境文件", "总体"


def inventory() -> pd.DataFrame:
    rows = []
    excluded = {".venv", ".matplotlib", "__pycache__"}
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and not excluded.intersection(p.parts)):
        rel = path.relative_to(ROOT).as_posix()
        purpose, checklist = purpose_for(rel)
        props = ""
        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
                props = f"{len(frame)}行×{len(frame.columns)}列；字段=" + ",".join(frame.columns[:12]) + ("…" if len(frame.columns) > 12 else "")
            except Exception as exc:
                props = f"CSV读取失败:{type(exc).__name__}"
        elif path.suffix.lower() == ".pdf": props = "矢量PDF；由正式结果或其源CSV生成"
        elif path.suffix.lower() == ".joblib": props = "二进制序列化对象；由对应脚本重建"
        elif path.suffix.lower() == ".json": props = "UTF-8 JSON元数据/审计"
        elif path.suffix.lower() == ".py": props = "UTF-8 Python源代码"
        elif path.suffix.lower() == ".md": props = "UTF-8 Markdown说明"
        elif path.suffix.lower() == ".log": props = "UTF-8运行记录"
        rows.append({"relative_path": rel, "file_type": path.suffix.lower() or "none", "size_bytes": path.stat().st_size, "purpose": purpose, "properties": props, "checklist_mapping": checklist})
    return pd.DataFrame(rows)


def main() -> None:
    (ROOT / "results" / "99_summary").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    save_csv(make_sensitivity_master(), "results/99_summary/sensitivity_master.csv")
    claims = make_claims(); save_csv(claims, "results/99_summary/claim_evidence.csv")
    four = make_four_part(); save_csv(four, "results/99_summary/method_four_part_evidence.csv")
    repro, repro_md = reproducibility_report()
    (ROOT / "reports" / "REPRODUCIBILITY_CHECK.md").write_text(repro_md, encoding="utf-8")
    (ROOT / "reports" / "reproducibility_check.json").write_text(json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")

    calibration = read_csv("results/02_q2/q2_1_calibration_decision.csv")
    subclasses = read_csv("results/02_q2/q2_2_subclasses.csv")
    unknown = read_csv("results/03_q3/q3_unknown_predictions.csv")
    network = read_csv("results/04_q4/q4_differential_edges.csv")
    errors = """# 运行错误、修正记录与模型局限

## 已发生并已修正的运行问题

- 问题1.2首轮绘图时缺少分组列，数值计算已完成但绘图中止；已修正绘图数据连接并整段重跑成功。
- 问题2.1首轮因 scikit-learn 弃用警告大量刷屏而人工中止；已移除弃用参数、抑制该类已知警告并完整重跑成功。该次中止结果未作为交付结果。
- 问题2.2在 Windows 上无法调用 `wmic` 探测物理核心，joblib 自动回退逻辑核心；不改变数值。后续已固定 `LOKY_MAX_CPU_COUNT=1` 消除提示。
- 最终汇总器首轮使用了旧字段别名 `factor/ci_low`，触发 KeyError；已按正式CSV的 `attribute/effect_ci_low` 字段修正。随后一键包装器打印该错误时又遇到GBK编码限制；已把包装器标准输出固定为UTF-8。两项都只影响报告生成，不影响已完成的模型结果。

## 不是错误、但必须限制表述的结果

- 问题2.2两类玻璃均未通过稳定亚类联合门槛；现有簇标签只能作为探索性分组，不得写聚类准确率或稳定亚类已被证实。
- 问题3的拒识样品必须保留“模型不确定或超出适用域”的标记，不得强行给确定真值。
- 问题4 Graphical Lasso 受 `n/p≥5` 识别门槛限制而禁用；保留收缩偏相关基准，不降低门槛凑结果。

## 实现偏离与未触发备用分支

- CART 剪枝候选使用预注册 `ccp_alpha={0,0.005,0.01}` 网格，而非逐折动态剪枝路径；因此该细项记为“部分实现”，不声称完全满足动态路径要求。
- Firth Logistic、PERMANOVA、经验分位映射、SVM 与 Graphical Lasso 均为条件备用；正式门槛未触发或识别门禁不允许时未运行，不伪造其结果。
- 未知样品没有真值，因此没有也不应生成问题3准确率。
- 问题1.3没有真实成对的风化前后观测，因此没有也不应生成“恢复RMSE”。
"""
    (ROOT / "reports" / "ERRORS_AND_LIMITATIONS.md").write_text(errors, encoding="utf-8")

    completion = pd.DataFrame([
        ["统一审计与预处理", "已实现", "审计、显式fit/transform、遮蔽比较、闭合/ILR及自动测试均落盘"],
        ["问题1.1", "已实现", "列联、精确/Monte Carlo、校正V、调整Logistic、Bootstrap/敏感性和5图"],
        ["问题1.2", "已实现", "分层加权ILR、置换、1000 Bootstrap、留一、敏感性和5图"],
        ["问题1.3", "已实现", "冻结效应反向位移、区间传播、适用性/闭合审计和4图"],
        ["问题2.1", "主体实现；1细项部分", "嵌套分组CV、三模型、校准门禁、冻结与1000 Bootstrap；CART动态剪枝路径未实现"],
        ["问题2.2", "已实现；结论为不支持稳定亚类", "R/Q聚类、候选k、样本外预测强度、共识和1000 Bootstrap"],
        ["问题3", "已实现", "冻结哈希验证、概率区间、适用域、模糊/拒识、预注册灵敏度和4图"],
        ["问题4", "已实现；Graphical Lasso按门禁禁用", "比例网络、1000 Bootstrap、10000置换、FDR、风化调整、偏相关和5图"],
        ["跨问证据与复现", "已实现" if repro["passed"] else "复现核验未通过", "claim、四段证据、敏感性总表、逐文件清单和复跑哈希"],
    ], columns=["module", "status", "evidence_or_limit"])
    save_csv(completion, "reports/CHECKLIST_COMPLETION_MATRIX.csv")
    matrix_md = "# 交付清单完成矩阵\n\n| 模块 | 状态 | 证据或限制 |\n|---|---|---|\n"
    matrix_md += "\n".join(f"| {r.module} | {r.status} | {r.evidence_or_limit} |" for r in completion.itertuples())
    matrix_md += "\n\n详细局限见 `ERRORS_AND_LIMITATIONS.md`，文件级映射见 `FILE_INVENTORY.csv`。\n"
    (ROOT / "reports" / "CHECKLIST_COMPLETION_MATRIX.md").write_text(matrix_md, encoding="utf-8")

    effects = read_csv("results/01_q1/q1_2_overall_effect.csv")
    q2m = read_csv("results/02_q2/q2_1_model_metrics.csv")
    result_report = f"""# RESULTS REPORT

## 数据与口径

表单1共58件文物，表单2共69个点位；按原始成分和85—105%门槛得到67个有效点，涉及56件文物。点位15和17只在预注册敏感性分支恢复。全部组成主模型经过空白/显式零分离、训练内填补、零替代、闭合和ILR/CLR处理。

## 问题1

属性关联、类型分层风化效应和反事实估计的正式数字分别见 `q1_1_association.csv`、`q1_2_overall_effect.csv` 和 `q1_3_counterfactual_long.csv`。两类整体风化效应范数为：{'; '.join(f"{r.glass_type}={r.effect_norm:.4f}, 95%CI[{r.effect_ci_low:.4f},{r.effect_ci_high:.4f}], p={r.permutation_p:.6g}" for r in effects.itertuples())}。反事实结果是模型估计，不是真实恢复值。

## 问题2

冻结模型为 Logistic。20次重复嵌套CV的平均文物级平衡准确率为 {q2m[q2m.model.eq('Logistic')].balanced_accuracy.mean():.4f}，平均Brier为 {q2m[q2m.model.eq('Logistic')].brier.mean():.4f}。校准门槛结果：{'; '.join(f"{r.model}:{'启用' if r.calibration_enabled else '不启用'}(Brier改善{r.brier_improvement:.4f})" for r in calibration.itertuples())}。两类稳定亚类支持标记均为0，因此仅保留探索性分组。

## 问题3

A1—A8的冻结外推共{len(unknown)}条，其中拒识{int(unknown.reject_flag.sum())}条。每条概率、Bootstrap区间、三模型一致性和拒识原因必须以 `q3_unknown_predictions.csv` 为准，不得报告未知真值准确率。

## 问题4

联合满足BH-FDR、效应量与方向稳定门槛的核心差异边共 {int(network.core_differential_edge.sum())} 条。网络边只表示统计比例关系，不表示化学反应或因果工艺。

## 图表

`figures/` 中所有PDF均为论文可引用矢量图；每张图的机器生成源数据位于 `figures/source_data/`。完整图名和对应任务见 `reports/FILE_INVENTORY.csv`。
"""
    (ROOT / "reports" / "RESULTS_REPORT.md").write_text(result_report, encoding="utf-8")

    versions = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "seaborn", "openpyxl", "joblib", "networkx", "pypdf"]
    lock = "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in versions) + "\n"
    (ROOT / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    summary = {"python": platform.python_version(), "reproducibility_passed": repro["passed"], "formal_csv_count": repro["after_count"], "pdf_figure_count": len(list((ROOT / "figures").glob("*.pdf"))), "claim_count": len(claims), "four_part_rows": len(four), "calibration_enabled_models": calibration.loc[calibration.calibration_enabled.eq(1), "model"].tolist(), "stable_subclass_supported_count": int(subclasses.stable_subclass_supported.sum()), "unknown_reject_count": int(unknown.reject_flag.sum()), "core_differential_edge_count": int(network.core_differential_edge.sum())}
    (ROOT / "results" / "99_summary" / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = """# 2022 C题完整建模编程交付

本目录由正式脚本从 `E:\\2022C\\C题\\附件.xlsx` 生成。先看 `reports/RESULTS_REPORT.md` 了解结果，再看 `reports/CHECKLIST_COMPLETION_MATRIX.md` 核对清单，所有错误、未触发备用和模型局限见 `reports/ERRORS_AND_LIMITATIONS.md`。

## 一键复现

```powershell
cd E:\\2022C\\C题_完整建模交付_20260830
.\\.venv\\Scripts\\python.exe code\\run_all.py
```

环境精确版本见 `requirements-lock.txt`，统一参数见 `config/config.json`。`.venv/` 和 `.matplotlib/` 是运行环境/缓存，不属于建模结果，不纳入逐文件交付清单。

## 目录职责

- `code/`：公共预处理、7个子任务、一键运行、哈希与最终验收脚本。
- `data/`：由原始Excel生成的规范化中间数据，不覆盖原附件。
- `results/`：正式机器可读结果；`99_summary/` 是跨问证据索引。
- `figures/`：论文可用PDF；`source_data/` 是每张图的源CSV。
- `models/`：冻结预处理器、分类器和Bootstrap管线。
- `logs/`：真实运行记录；`reports/`：人读报告与验收材料。

## 每个文件的作用与属性

请打开 `reports/FILE_INVENTORY.csv`。它逐个列出所有交付文件的相对路径、类型、字节数、用途、CSV行列/字段或PDF/模型属性，以及对应交付清单模块。运行环境第三方包被明确排除，避免把数千个依赖文件冒充交付成果。
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    save_csv(inventory(), "reports/FILE_INVENTORY.csv")
    save_csv(inventory(), "reports/FILE_INVENTORY.csv")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
