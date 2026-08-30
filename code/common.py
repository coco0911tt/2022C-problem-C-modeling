from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_THIS_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_THIS_ROOT / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from statsmodels.stats.multitest import multipletests


ROOT = _THIS_ROOT
SOURCE_ROOT = ROOT.parent
SOURCE_XLSX = SOURCE_ROOT / "C题" / "附件.xlsx"
CHECKLIST = SOURCE_ROOT / "交付清单_编程手_逐问.md"
CONFIG_PATH = ROOT / "config" / "config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = json.load(fh)

MASTER_SEED = int(CONFIG["master_seed"])

COMPONENT_LABELS = [
    "二氧化硅(SiO2)", "氧化钠(Na2O)", "氧化钾(K2O)", "氧化钙(CaO)",
    "氧化镁(MgO)", "氧化铝(Al2O3)", "氧化铁(Fe2O3)", "氧化铜(CuO)",
    "氧化铅(PbO)", "氧化钡(BaO)", "五氧化二磷(P2O5)", "氧化锶(SrO)",
    "氧化锡(SnO2)", "二氧化硫(SO2)",
]
COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
COMPONENT_RENAME = dict(zip(COMPONENT_LABELS, COMPONENTS))
COMPONENT_CN = dict(zip(COMPONENTS, [
    "二氧化硅", "氧化钠", "氧化钾", "氧化钙", "氧化镁", "氧化铝", "氧化铁",
    "氧化铜", "氧化铅", "氧化钡", "五氧化二磷", "氧化锶", "氧化锡", "二氧化硫",
]))

BALANCE_LIBRARY = [
    {
        "name": "钾_相对_铅钡",
        "numerator": ["K2O"],
        "denominator": ["PbO", "BaO"],
        "interpretation": "钾系助熔成分相对于铅钡重金属成分",
    },
    {
        "name": "铅钡_相对_硅铝",
        "numerator": ["PbO", "BaO"],
        "denominator": ["SiO2", "Al2O3"],
        "interpretation": "铅钡组分相对于硅铝骨架",
    },
    {
        "name": "硅铝_相对_总助熔",
        "numerator": ["SiO2", "Al2O3"],
        "denominator": ["Na2O", "K2O", "CaO", "MgO", "PbO", "BaO"],
        "interpretation": "玻璃骨架相对于常见助熔/稳定组分",
    },
]


def ensure_dirs() -> None:
    for rel in [
        "data", "models", "figures", "reports", "logs",
        "results/00_audit", "results/01_q1", "results/02_q2",
        "results/03_q3", "results/04_q4", "results/99_summary",
        "figures/source_data",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, rel: str, index: bool = False) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_json(obj: Any, rel: str) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=json_default)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, filename: str, source_df: pd.DataFrame | None = None) -> None:
    configure_plotting()
    fig.savefig(ROOT / "figures" / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)
    if source_df is not None:
        source_name = Path(filename).stem + ".csv"
        save_csv(source_df, f"figures/source_data/{source_name}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(SOURCE_XLSX, sheet_name=None, engine="openpyxl")
    required = {"表单1", "表单2", "表单3"}
    if set(sheets) != required:
        raise ValueError(f"工作表不匹配：{list(sheets)}")

    t1 = sheets["表单1"].copy()
    t2 = sheets["表单2"].copy()
    t3 = sheets["表单3"].copy()
    t1.columns = [str(c).strip() for c in t1.columns]
    t2.columns = [str(c).strip() for c in t2.columns]
    t3.columns = [str(c).strip() for c in t3.columns]

    required_t1 = ["文物编号", "纹饰", "类型", "颜色", "表面风化"]
    required_t2 = ["文物采样点"] + COMPONENT_LABELS
    required_t3 = ["文物编号", "表面风化"] + COMPONENT_LABELS
    for name, df, cols in [("表单1", t1, required_t1), ("表单2", t2, required_t2), ("表单3", t3, required_t3)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}缺少字段：{missing}")

    t1 = t1[required_t1].rename(columns={
        "文物编号": "artifact_id", "纹饰": "pattern", "类型": "glass_type",
        "颜色": "color", "表面风化": "surface_weathering",
    })
    t1["artifact_id"] = t1["artifact_id"].astype(str).str.zfill(2)
    t1["raw_row_id"] = [f"表单1:{i}" for i in range(2, len(t1) + 2)]

    t2 = t2[required_t2].rename(columns={"文物采样点": "sample_point", **COMPONENT_RENAME})
    t2["sample_point"] = t2["sample_point"].astype(str).str.strip()
    t2["artifact_id"] = t2["sample_point"].str.extract(r"^(\d+)", expand=False)
    if t2["artifact_id"].isna().any():
        bad = t2.loc[t2["artifact_id"].isna(), "sample_point"].tolist()
        raise ValueError(f"采样点编号解析失败：{bad}")
    t2["artifact_id"] = t2["artifact_id"].str.zfill(2)
    t2["raw_row_id"] = [f"表单2:{i}" for i in range(2, len(t2) + 2)]
    for comp in COMPONENTS:
        t2[comp] = pd.to_numeric(t2[comp], errors="coerce")

    t2 = t2.merge(t1.drop(columns="raw_row_id"), on="artifact_id", how="left", validate="many_to_one")
    if t2["glass_type"].isna().any():
        raise ValueError("表单2存在无法关联到表单1的文物编号")

    def point_weathering(row: pd.Series) -> tuple[str, str]:
        point = str(row["sample_point"])
        if "未风化点" in point:
            return "无风化", "point_unweathered"
        if "严重风化点" in point:
            return "风化", "point_severe"
        return str(row["surface_weathering"]), "artifact_inherited"

    parsed = t2.apply(point_weathering, axis=1, result_type="expand")
    parsed.columns = ["point_weathering", "weathering_source"]
    t2 = pd.concat([t2, parsed], axis=1)
    t2["raw_component_sum_pct"] = t2[COMPONENTS].sum(axis=1, skipna=True)
    t2["valid_sum_flag"] = t2["raw_component_sum_pct"].between(
        CONFIG["valid_sum_low"], CONFIG["valid_sum_high"], inclusive="both"
    )
    t2["exclusion_reason"] = np.where(t2["valid_sum_flag"], "", "component_sum_outside_85_105")
    for comp in COMPONENTS:
        t2[f"miss_{comp}"] = t2[comp].isna().astype(int)
        t2[f"zero_{comp}"] = t2[comp].eq(0).fillna(False).astype(int)

    t3 = t3[required_t3].rename(columns={
        "文物编号": "unknown_id", "表面风化": "surface_weathering", **COMPONENT_RENAME,
    })
    t3["unknown_id"] = t3["unknown_id"].astype(str).str.strip()
    t3["raw_row_id"] = [f"表单3:{i}" for i in range(2, len(t3) + 2)]
    for comp in COMPONENTS:
        t3[comp] = pd.to_numeric(t3[comp], errors="coerce")
        t3[f"miss_{comp}"] = t3[comp].isna().astype(int)
        t3[f"zero_{comp}"] = t3[comp].eq(0).fillna(False).astype(int)
    t3["raw_component_sum_pct"] = t3[COMPONENTS].sum(axis=1, skipna=True)
    t3["valid_sum_flag"] = t3["raw_component_sum_pct"].between(
        CONFIG["valid_sum_low"], CONFIG["valid_sum_high"], inclusive="both"
    )
    t3["exclusion_reason"] = np.where(t3["valid_sum_flag"], "", "component_sum_outside_85_105")
    return t1, t2, t3


def load_known_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """只读取表单1和表单2；问题2冻结前禁止通过此函数接触表单3。"""
    sheets = pd.read_excel(SOURCE_XLSX, sheet_name=["表单1", "表单2"], engine="openpyxl")
    t1 = sheets["表单1"].copy()
    t2 = sheets["表单2"].copy()
    t1.columns = [str(c).strip() for c in t1.columns]
    t2.columns = [str(c).strip() for c in t2.columns]
    required_t1 = ["文物编号", "纹饰", "类型", "颜色", "表面风化"]
    required_t2 = ["文物采样点"] + COMPONENT_LABELS
    for name, df, cols in [("表单1", t1, required_t1), ("表单2", t2, required_t2)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}缺少字段：{missing}")
    t1 = t1[required_t1].rename(columns={
        "文物编号": "artifact_id", "纹饰": "pattern", "类型": "glass_type",
        "颜色": "color", "表面风化": "surface_weathering",
    })
    t1["artifact_id"] = t1["artifact_id"].astype(str).str.zfill(2)
    t1["raw_row_id"] = [f"表单1:{i}" for i in range(2, len(t1) + 2)]
    t2 = t2[required_t2].rename(columns={"文物采样点": "sample_point", **COMPONENT_RENAME})
    t2["sample_point"] = t2["sample_point"].astype(str).str.strip()
    t2["artifact_id"] = t2["sample_point"].str.extract(r"^(\d+)", expand=False)
    if t2["artifact_id"].isna().any():
        raise ValueError("表单2存在无法解析的采样点编号")
    t2["artifact_id"] = t2["artifact_id"].str.zfill(2)
    t2["raw_row_id"] = [f"表单2:{i}" for i in range(2, len(t2) + 2)]
    for comp in COMPONENTS:
        t2[comp] = pd.to_numeric(t2[comp], errors="coerce")
    t2 = t2.merge(t1.drop(columns="raw_row_id"), on="artifact_id", how="left", validate="many_to_one")
    if t2["glass_type"].isna().any():
        raise ValueError("表单2存在无法关联到表单1的文物编号")
    point_states = []
    sources = []
    for _, row in t2.iterrows():
        point = str(row["sample_point"])
        if "未风化点" in point:
            point_states.append("无风化")
            sources.append("point_unweathered")
        elif "严重风化点" in point:
            point_states.append("风化")
            sources.append("point_severe")
        else:
            point_states.append(str(row["surface_weathering"]))
            sources.append("artifact_inherited")
    t2["point_weathering"] = point_states
    t2["weathering_source"] = sources
    t2["raw_component_sum_pct"] = t2[COMPONENTS].sum(axis=1, skipna=True)
    t2["valid_sum_flag"] = t2["raw_component_sum_pct"].between(
        CONFIG["valid_sum_low"], CONFIG["valid_sum_high"], inclusive="both"
    )
    t2["exclusion_reason"] = np.where(t2["valid_sum_flag"], "", "component_sum_outside_85_105")
    for comp in COMPONENTS:
        t2[f"miss_{comp}"] = t2[comp].isna().astype(int)
        t2[f"zero_{comp}"] = t2[comp].eq(0).fillna(False).astype(int)
    return t1, t2


def load_unknown_data() -> pd.DataFrame:
    """仅在问题2.1冻结后调用表单3。"""
    t3 = pd.read_excel(SOURCE_XLSX, sheet_name="表单3", engine="openpyxl")
    t3.columns = [str(c).strip() for c in t3.columns]
    required_t3 = ["文物编号", "表面风化"] + COMPONENT_LABELS
    missing = [c for c in required_t3 if c not in t3.columns]
    if missing:
        raise ValueError(f"表单3缺少字段：{missing}")
    t3 = t3[required_t3].rename(columns={
        "文物编号": "unknown_id", "表面风化": "surface_weathering", **COMPONENT_RENAME,
    })
    t3["unknown_id"] = t3["unknown_id"].astype(str).str.strip()
    t3["raw_row_id"] = [f"表单3:{i}" for i in range(2, len(t3) + 2)]
    for comp in COMPONENTS:
        t3[comp] = pd.to_numeric(t3[comp], errors="coerce")
        t3[f"miss_{comp}"] = t3[comp].isna().astype(int)
        t3[f"zero_{comp}"] = t3[comp].eq(0).fillna(False).astype(int)
    t3["raw_component_sum_pct"] = t3[COMPONENTS].sum(axis=1, skipna=True)
    t3["valid_sum_flag"] = t3["raw_component_sum_pct"].between(
        CONFIG["valid_sum_low"], CONFIG["valid_sum_high"], inclusive="both"
    )
    t3["exclusion_reason"] = np.where(t3["valid_sum_flag"], "", "component_sum_outside_85_105")
    return t3


@dataclass
class TransformResult:
    filled: pd.DataFrame
    closed: pd.DataFrame
    clr: pd.DataFrame
    ilr: np.ndarray
    closure_error: np.ndarray
    fallback_used: pd.DataFrame


class CompositionPreprocessor:
    def __init__(
        self,
        components: list[str] | None = None,
        min_group_observed: int | None = None,
        zero_c: float | None = None,
        fixed_active: list[str] | None = None,
    ) -> None:
        self.components = list(components or COMPONENTS)
        self.min_group_observed = int(min_group_observed or CONFIG["min_group_observed"])
        self.zero_c = float(zero_c if zero_c is not None else CONFIG["zero_c"])
        self.fixed_active = list(fixed_active) if fixed_active is not None else None

    def fit(self, X: pd.DataFrame, groups: Iterable[Any]) -> "CompositionPreprocessor":
        X = X[self.components].apply(pd.to_numeric, errors="coerce").copy()
        groups_s = pd.Series(list(groups), index=X.index).astype(str)
        self.global_medians_: dict[str, float] = {}
        self.global_counts_: dict[str, int] = {}
        for comp in self.components:
            obs = X[comp].dropna()
            self.global_counts_[comp] = int(obs.shape[0])
            self.global_medians_[comp] = float(obs.median()) if len(obs) else np.nan
        if self.fixed_active is None:
            self.active_components_ = [
                comp for comp in self.components if self.global_counts_[comp] >= self.min_group_observed
            ]
        else:
            self.active_components_ = list(self.fixed_active)
        if len(self.active_components_) < 2:
            raise ValueError("活动成分少于2个，不能建立ILR")
        for comp in self.active_components_:
            if not np.isfinite(self.global_medians_[comp]):
                raise ValueError(f"活动成分{comp}在训练集中全缺失")

        self.group_medians_: dict[str, dict[str, float]] = {}
        self.group_counts_: dict[str, dict[str, int]] = {}
        self.group_fallback_: dict[str, dict[str, str]] = {}
        for group_name in sorted(groups_s.unique()):
            mask = groups_s.eq(group_name)
            self.group_medians_[group_name] = {}
            self.group_counts_[group_name] = {}
            self.group_fallback_[group_name] = {}
            for comp in self.active_components_:
                obs = X.loc[mask, comp].dropna()
                n = int(len(obs))
                self.group_counts_[group_name][comp] = n
                if n >= self.min_group_observed:
                    value = float(obs.median())
                    level = "group"
                else:
                    value = self.global_medians_[comp]
                    level = "global"
                self.group_medians_[group_name][comp] = float(value)
                self.group_fallback_[group_name][comp] = level

        filled_training = self._fill(X, groups_s)[0]
        self.delta_: dict[str, float] = {}
        for comp in self.active_components_:
            positives = filled_training.loc[filled_training[comp] > 0, comp]
            if positives.empty:
                raise ValueError(f"活动成分{comp}没有正值，不能进行零替代")
            self.delta_[comp] = float(self.zero_c * positives.min())
        self.basis_ = helmert(len(self.active_components_), full=False)
        self.fitted_ = True
        return self

    def _fill(self, X: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        filled = X[self.active_components_].copy()
        fallback = pd.DataFrame(False, index=X.index, columns=self.active_components_)
        for idx in filled.index:
            group = str(groups.loc[idx])
            for comp in self.active_components_:
                if pd.isna(filled.at[idx, comp]):
                    if group in self.group_medians_:
                        filled.at[idx, comp] = self.group_medians_[group][comp]
                        fallback.at[idx, comp] = self.group_fallback_[group][comp] != "group"
                    else:
                        filled.at[idx, comp] = self.global_medians_[comp]
                        fallback.at[idx, comp] = True
        return filled.astype(float), fallback

    def transform(self, X: pd.DataFrame, groups: Iterable[Any]) -> TransformResult:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("预处理器尚未fit")
        X = X[self.components].apply(pd.to_numeric, errors="coerce").copy()
        groups_s = pd.Series(list(groups), index=X.index).astype(str)
        filled, fallback = self._fill(X, groups_s)
        replaced = filled.copy()
        for comp in self.active_components_:
            if (replaced[comp] < 0).any():
                raise ValueError(f"成分{comp}出现负值")
            replaced.loc[replaced[comp] == 0, comp] = self.delta_[comp]
        row_sum = replaced.sum(axis=1)
        if (row_sum <= 0).any():
            raise ValueError("闭合前存在非正行和")
        closed = replaced.div(row_sum, axis=0) * 100.0
        closure_error = np.abs(closed.sum(axis=1).to_numpy() - 100.0)
        logx = np.log(closed.to_numpy() / 100.0)
        clr_arr = logx - logx.mean(axis=1, keepdims=True)
        ilr_arr = logx @ self.basis_.T
        if not np.isfinite(ilr_arr).all():
            raise ValueError("ILR出现NaN/Inf")
        return TransformResult(
            filled=filled,
            closed=pd.DataFrame(closed.to_numpy(), index=X.index, columns=self.active_components_),
            clr=pd.DataFrame(clr_arr, index=X.index, columns=self.active_components_),
            ilr=ilr_arr,
            closure_error=closure_error,
            fallback_used=fallback,
        )

    def inverse_ilr(self, z: np.ndarray) -> pd.DataFrame:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        logx = z @ self.basis_
        x = np.exp(logx - logx.max(axis=1, keepdims=True))
        x = x / x.sum(axis=1, keepdims=True) * 100.0
        return pd.DataFrame(x, columns=self.active_components_)

    def parameter_rows(self, task_id: str, split_id: str = "full") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group, medians in self.group_medians_.items():
            for comp, value in medians.items():
                rows.append({
                    "task_id": task_id,
                    "split_id": split_id,
                    "group_name": group,
                    "component": comp,
                    "n_observed": self.group_counts_[group][comp],
                    "median_value": value,
                    "fallback_level": self.group_fallback_[group][comp],
                    "active_feature": int(comp in self.active_components_),
                })
        return rows


class KNNCompositionPreprocessor:
    """仅用于预注册敏感性分支；标准化、KNN和零替代均只在训练集fit。"""

    def __init__(self, n_neighbors: int = 5, zero_c: float = 0.5, fixed_active: list[str] | None = None) -> None:
        self.n_neighbors = int(n_neighbors)
        self.zero_c = float(zero_c)
        self.fixed_active = list(fixed_active) if fixed_active is not None else None
        self.components = list(COMPONENTS)

    def fit(self, X: pd.DataFrame, groups: Iterable[Any] | None = None) -> "KNNCompositionPreprocessor":
        from sklearn.impute import KNNImputer

        X = X[self.components].apply(pd.to_numeric, errors="coerce").copy()
        counts = X.notna().sum()
        self.active_components_ = self.fixed_active or [c for c in self.components if counts[c] >= CONFIG["min_group_observed"]]
        if len(self.active_components_) < 2:
            raise ValueError("KNN敏感性活动成分少于2个")
        train = X[self.active_components_].astype(float)
        self.mean_ = train.mean(axis=0)
        self.scale_ = train.std(axis=0, ddof=0).replace(0, 1.0)
        scaled = (train - self.mean_) / self.scale_
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        imputed_scaled = self.imputer_.fit_transform(scaled)
        imputed = imputed_scaled * self.scale_.to_numpy() + self.mean_.to_numpy()
        imputed = np.maximum(imputed, 0.0)
        self.delta_ = {}
        for j, comp in enumerate(self.active_components_):
            positive = imputed[:, j][imputed[:, j] > 0]
            if len(positive) == 0:
                raise ValueError(f"KNN敏感性成分{comp}无正值")
            self.delta_[comp] = float(self.zero_c * positive.min())
        self.basis_ = helmert(len(self.active_components_), full=False)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, groups: Iterable[Any] | None = None) -> TransformResult:
        X = X[self.components].apply(pd.to_numeric, errors="coerce").copy()
        train = X[self.active_components_].astype(float)
        scaled = (train - self.mean_) / self.scale_
        imputed_scaled = self.imputer_.transform(scaled)
        arr = imputed_scaled * self.scale_.to_numpy() + self.mean_.to_numpy()
        arr = np.maximum(arr, 0.0)
        filled = pd.DataFrame(arr, index=X.index, columns=self.active_components_)
        replaced = filled.copy()
        for comp in self.active_components_:
            replaced.loc[replaced[comp] == 0, comp] = self.delta_[comp]
        closed = replaced.div(replaced.sum(axis=1), axis=0) * 100.0
        logx = np.log(closed.to_numpy() / 100.0)
        clr = logx - logx.mean(axis=1, keepdims=True)
        ilr = logx @ self.basis_.T
        return TransformResult(
            filled=filled,
            closed=closed,
            clr=pd.DataFrame(clr, index=X.index, columns=self.active_components_),
            ilr=ilr,
            closure_error=np.abs(closed.sum(axis=1).to_numpy() - 100.0),
            fallback_used=pd.DataFrame(X[self.active_components_].isna().to_numpy(), index=X.index, columns=self.active_components_),
        )

    def inverse_ilr(self, z: np.ndarray) -> pd.DataFrame:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        logx = z @ self.basis_
        x = np.exp(logx - logx.max(axis=1, keepdims=True))
        x = x / x.sum(axis=1, keepdims=True) * 100.0
        return pd.DataFrame(x, columns=self.active_components_)


def aitchison_center(closed: pd.DataFrame) -> pd.Series:
    logx = np.log(closed.to_numpy() / 100.0)
    center = np.exp(logx.mean(axis=0))
    center = center / center.sum() * 100.0
    return pd.Series(center, index=closed.columns)


def aggregate_artifact_centers(
    meta: pd.DataFrame,
    transform: TransformResult,
    preprocessor: CompositionPreprocessor,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_cols = list(extra_cols or [])
    rows: list[dict[str, Any]] = []
    ilr_df = pd.DataFrame(transform.ilr, index=meta.index)
    for artifact_id, idx in meta.groupby("artifact_id").groups.items():
        idx_list = list(idx)
        z = ilr_df.loc[idx_list].mean(axis=0).to_numpy()
        composition = preprocessor.inverse_ilr(z)[0:1].iloc[0]
        row: dict[str, Any] = {"artifact_id": artifact_id, "n_points": len(idx_list)}
        for col in extra_cols:
            row[col] = meta.loc[idx_list[0], col]
        for comp, value in composition.items():
            row[comp] = float(value)
        for j, value in enumerate(z):
            row[f"ilr_{j+1}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_balances(closed: pd.DataFrame, library: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    library = library or BALANCE_LIBRARY
    result: dict[str, np.ndarray] = {}
    for spec in library:
        numerator = [c for c in spec["numerator"] if c in closed.columns]
        denominator = [c for c in spec["denominator"] if c in closed.columns]
        if not numerator or not denominator:
            result[spec["name"]] = np.full(len(closed), np.nan)
            continue
        r, s = len(numerator), len(denominator)
        scale = math.sqrt(r * s / (r + s))
        log_num = np.log(closed[numerator].to_numpy()).mean(axis=1)
        log_den = np.log(closed[denominator].to_numpy()).mean(axis=1)
        result[spec["name"]] = scale * (log_num - log_den)
    return pd.DataFrame(result, index=closed.index)


def bh_fdr(pvalues: Iterable[float], alpha: float | None = None) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    valid = np.isfinite(p)
    out = np.full_like(p, np.nan, dtype=float)
    if valid.any():
        out[valid] = multipletests(p[valid], alpha=alpha or CONFIG["fdr_q"], method="fdr_bh")[1]
    return out


def weighted_ols(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sw = np.sqrt(weights)[:, None]
    coef, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
    residual = Y - X @ coef
    return coef, residual


def artifact_weights(meta: pd.DataFrame) -> np.ndarray:
    counts = meta.groupby("artifact_id")["artifact_id"].transform("count").astype(float)
    return (1.0 / counts).to_numpy()


def corrected_cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency

    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n <= 1:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return math.sqrt(phi2corr / denom) if denom > 0 else 0.0


def environment_info() -> dict[str, Any]:
    import matplotlib
    import scipy
    import sklearn
    import statsmodels

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "source_xlsx": str(SOURCE_XLSX),
        "source_xlsx_sha256": hash_file(SOURCE_XLSX),
        "checklist_sha256": hash_file(CHECKLIST),
    }


def dump_joblib(obj: Any, rel: str) -> Path:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    return path


def load_joblib(rel: str) -> Any:
    return joblib.load(ROOT / rel)


ensure_dirs()
