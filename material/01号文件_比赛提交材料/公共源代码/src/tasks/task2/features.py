"""任务二标签、安全特征与机制特征组 F1–F7。"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import (
    TASK2_BASE_FEATURES,
    TASK2_EXCLUDED_FEATURES,
    TASK2_MECHANISM_FEATURES,
    TASK2_SAFE_CANDIDATE_FEATURES,
    TASK2_TARGET,
)

logger = logging.getLogger(__name__)

# 机制特征分组（F1–F7）：仅当源字段存在时纳入
TASK2_MECH_GROUPS: Dict[str, List[str]] = {
    "F1_charging": [
        "charging_gap",
        "charging_convenience",
        "log_station_distance",
        "charging_gap_index",
        "charging_convenience_index",
        "distance_access_interaction",
    ],
    "F2_economy": [
        "log_annual_income",
        "monthly_income",
        "fuel_income_pressure",
        "income_over_near",
        "electricity_x_income",
    ],
    "F3_travel": [
        "commute_ratio",
        "noncommute_km",
    ],
    "F4_vehicle": [
        "replacement_pressure",
    ],
    "F5_city_demo": [
        "city_is_urban",
        "city_is_suburban",
        "city_is_rural",
        "income_x_urban",
        "income_x_suburban",
        "income_x_rural",
        "near_x_rural",
        "access_x_prev_ev",
    ],
    "F6_readiness": [
        "readiness_mean",
        "eco_tech_mean",
        "knowledge_deficit",
        "incentive_knowledge",
    ],
    "F7_housing_proxy": [
        "income_ge_train_median",
        "urban_high_income",
        "suburban_high_income",
    ],
}

# 数据集中缺失、但影响家庭充电的关键结构变量（文档用）
UNOBSERVED_HOUSING_FIELDS: List[str] = [
    "private_parking_available",
    "garage_available",
    "housing_tenure",
    "housing_type",
    "property_management_restriction",
    "home_electrical_capacity",
    "condo_charging_policy",
]


def unobserved_housing_fields() -> List[str]:
    """返回数据集中缺失的结构/住房相关变量列表（供论文说明）。"""
    return list(UNOBSERVED_HOUSING_FIELDS)


def resolve_safe_features(df: pd.DataFrame) -> List[str]:
    """从安全候选中筛选实际存在的列；缺失则警告并跳过。"""
    available: List[str] = []
    for col in TASK2_SAFE_CANDIDATE_FEATURES:
        if col in TASK2_EXCLUDED_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("任务二安全候选特征缺失，已跳过：%s", col)
            continue
        if col not in available:
            available.append(col)
    return available


def check_task2_base_columns(df: pd.DataFrame) -> None:
    """官方基础字段必须存在。"""
    required = TASK2_BASE_FEATURES + [TASK2_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("任务二缺少官方基础字段：" + ", ".join(missing))


def build_target(df: pd.DataFrame) -> pd.Series:
    """构造任务二标签 0/1。"""
    y_raw = pd.to_numeric(df[TASK2_TARGET], errors="coerce")
    if y_raw.isna().any():
        raise ValueError("任务二目标列存在缺失或无法解析为数值的值。")
    unique_vals = set(y_raw.unique().tolist())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"任务二目标值异常，期望为 0/1，实际出现：{sorted(unique_vals)}")
    return y_raw.astype(int)


def to_num(series: pd.Series) -> Tuple[pd.Series, int]:
    """清洗数值：去逗号/空格后 to_numeric，返回转换失败数。"""
    return _to_numeric_series(series)


def _to_numeric_series(series: pd.Series) -> Tuple[pd.Series, int]:
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        fail = int(series.notna().sum() - converted.notna().sum())
        return converted, fail

    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    as_str = as_str.str.replace(r"[$¥€]", "", regex=True)
    converted = pd.to_numeric(as_str, errors="coerce")
    fail = int((series.notna() & converted.isna()).sum())
    return converted, fail


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造机制特征与住房/充电代理特征。

    仅使用行内信息（不加目标、不用全表统计），避免泄露。
    依赖训练集分位数的相对特征在 preprocessor 中另行生成。
    """
    out = df.copy()
    eps = 1.0

    nearest = access = income = daily = weekly = fuel = None
    know = tech = eco = incentive = None
    has_nearest = "nearest_charging_station_km" in out.columns
    has_access = "charging_station_accessibility" in out.columns

    if has_nearest:
        nearest, _ = _to_numeric_series(out["nearest_charging_station_km"])
    if has_access:
        access, _ = _to_numeric_series(out["charging_station_accessibility"])
    if has_nearest and has_access:
        out["charging_gap_index"] = nearest / (access + eps)
        out["charging_convenience_index"] = access / (nearest + eps)
        out["charging_gap"] = out["charging_gap_index"]
        out["charging_convenience"] = out["charging_convenience_index"]
        out["distance_access_interaction"] = nearest * (11.0 - access)
    if has_nearest:
        out["log_station_distance"] = np.log1p(nearest.clip(lower=0))

    if "daily_commute_km" in out.columns and "weekly_travel_distance_km" in out.columns:
        daily, _ = _to_numeric_series(out["daily_commute_km"])
        weekly, _ = _to_numeric_series(out["weekly_travel_distance_km"])
        out["commute_ratio"] = 7.0 * daily / (weekly + eps)
        out["noncommute_km"] = (weekly - 7.0 * daily).clip(lower=0)

    if "annual_income" in out.columns:
        income, _ = _to_numeric_series(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))
        out["monthly_income"] = income / 12.0

    if "fuel_expense_per_month" in out.columns and income is not None:
        fuel, _ = _to_numeric_series(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)

    if "vehicle_age_years" in out.columns and "fuel_expense_per_month" in out.columns:
        vage, _ = _to_numeric_series(out["vehicle_age_years"])
        if fuel is None:
            fuel, _ = _to_numeric_series(out["fuel_expense_per_month"])
        out["replacement_pressure"] = vage * fuel

    if "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["city_is_urban"] = (city == "urban").astype(float)
        out["city_is_suburban"] = (city == "suburban").astype(float)
        out["city_is_rural"] = (city == "rural").astype(float)
        if income is not None:
            out["income_x_urban"] = income * out["city_is_urban"]
            out["income_x_suburban"] = income * out["city_is_suburban"]
            out["income_x_rural"] = income * out["city_is_rural"]
        if nearest is not None:
            out["near_x_rural"] = nearest * out["city_is_rural"]

    if has_access and "previous_ev_experience" in out.columns:
        prev, _ = _to_numeric_series(out["previous_ev_experience"])
        out["access_x_prev_ev"] = access * prev

    if income is not None and nearest is not None:
        out["income_over_near"] = income / (nearest + eps)

    if "electricity_cost_per_kwh" in out.columns and income is not None:
        elec, _ = _to_numeric_series(out["electricity_cost_per_kwh"])
        out["electricity_x_income"] = elec * income

    if "ev_knowledge_score" in out.columns:
        know, _ = _to_numeric_series(out["ev_knowledge_score"])
        out["knowledge_deficit"] = 10.0 - know
    if "technology_affinity_score" in out.columns:
        tech, _ = _to_numeric_series(out["technology_affinity_score"])
    if "environmental_awareness_score" in out.columns:
        eco, _ = _to_numeric_series(out["environmental_awareness_score"])
    if "government_incentive_awareness" in out.columns:
        incentive, _ = _to_numeric_series(out["government_incentive_awareness"])

    readiness_parts = [s for s in (know, tech, eco, incentive) if s is not None]
    if readiness_parts:
        out["readiness_mean"] = sum(readiness_parts) / len(readiness_parts)
    if eco is not None and tech is not None:
        out["eco_tech_mean"] = (eco + tech) / 2.0
    if know is not None and incentive is not None:
        out["incentive_knowledge"] = incentive * know

    return out


def _add_train_relative_features(
    df: pd.DataFrame,
    income_median: float | None,
) -> pd.DataFrame:
    """基于训练集收入中位数的相对特征（防泄露：中位数只来自当前训练折）。"""
    out = df.copy()
    if income_median is None or "annual_income" not in out.columns:
        return out
    income, _ = _to_numeric_series(out["annual_income"])
    high = (income >= income_median).astype(float)
    out["income_ge_train_median"] = high
    if "city_is_urban" in out.columns:
        out["urban_high_income"] = out["city_is_urban"] * high
    elif "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["urban_high_income"] = ((city == "urban").astype(float) * high)
    if "city_is_suburban" in out.columns:
        out["suburban_high_income"] = out["city_is_suburban"] * high
    elif "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["suburban_high_income"] = ((city == "suburban").astype(float) * high)
    return out


def _collect_mechanism_pool(probe: pd.DataFrame) -> List[str]:
    pool: List[str] = []
    for cols in TASK2_MECH_GROUPS.values():
        for c in cols:
            if c in probe.columns and c not in pool and c not in TASK2_EXCLUDED_FEATURES:
                pool.append(c)
    for c in TASK2_MECHANISM_FEATURES:
        if c in probe.columns and c not in pool:
            pool.append(c)
    return pool


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """返回 A/B 及 B+各机制组特征列表。"""
    official = list(TASK2_BASE_FEATURES)
    safe = resolve_safe_features(df)
    probe = add_mechanism_features(df.head(3))
    mech_pool = _collect_mechanism_pool(probe)

    out: Dict[str, List[str]] = {
        "A_official": official,
        "B_safe_raw": safe,
    }
    for gname, gcols in TASK2_MECH_GROUPS.items():
        feats = list(safe)
        for c in gcols:
            if c in mech_pool and c not in feats:
                feats.append(c)
        out[f"B+{gname}"] = feats

    mech_all = list(safe)
    for c in mech_pool:
        if c not in mech_all:
            mech_all.append(c)
    out["B+all_mechanism"] = mech_all
    out["mechanism_only_pool"] = mech_pool

    for key, cols in out.items():
        if key != "mechanism_only_pool":
            assert_no_leakage(cols)
    return out


def assert_no_leakage(features: Sequence[str]) -> None:
    """禁止目标、任务一标签源及后验字段进入正式模型。"""
    bad = [c for c in features if c in TASK2_EXCLUDED_FEATURES]
    if bad:
        raise ValueError(f"分类特征含禁止字段：{bad}")
