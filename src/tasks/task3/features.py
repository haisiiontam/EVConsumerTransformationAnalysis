"""任务三标签、安全特征与机制特征组。"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import (
    TASK3_EXCLUDED_FEATURES,
    TASK3_MECH_GROUPS,
    TASK3_MECHANISM_FEATURES,
    TASK3_OFFICIAL_FEATURES,
    TASK3_SAFE_CANDIDATE_FEATURES,
    TASK3_SOURCE_SCORE,
    TASK3_STRONG_FEATURES,
    TASK3_THRESHOLD,
)

logger = logging.getLogger(__name__)


def build_target(df: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(df[TASK3_SOURCE_SCORE], errors="coerce")
    if score.isna().any():
        raise ValueError("任务三原始评分列存在缺失或无法解析的值。")
    return (score > TASK3_THRESHOLD).astype(int)


def raw_score(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df[TASK3_SOURCE_SCORE], errors="coerce")


def check_task3_base_columns(df: pd.DataFrame) -> None:
    required = TASK3_OFFICIAL_FEATURES + [TASK3_SOURCE_SCORE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("任务三缺少必需字段：" + ", ".join(missing))


def to_num(series: pd.Series) -> Tuple[pd.Series, int]:
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        fail = int(series.notna().sum() - converted.notna().sum())
        return converted, fail
    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    converted = pd.to_numeric(as_str, errors="coerce")
    fail = int((series.notna() & converted.isna()).sum())
    return converted, fail


def resolve_safe_features(df: pd.DataFrame) -> List[str]:
    available: List[str] = []
    for col in TASK3_SAFE_CANDIDATE_FEATURES:
        if col in TASK3_EXCLUDED_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("任务三安全候选特征缺失，已跳过：%s", col)
            continue
        if col not in available:
            available.append(col)
    for c in TASK3_STRONG_FEATURES:
        if c in df.columns and c not in available and c not in TASK3_EXCLUDED_FEATURES:
            available.append(c)
    return available


def assert_no_leakage(features: Sequence[str]) -> None:
    bad = [c for c in features if c in TASK3_EXCLUDED_FEATURES or c == TASK3_SOURCE_SCORE]
    if bad:
        raise ValueError(f"分类特征含禁止字段：{bad}")


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """行内机制特征；绝不写入 range_anxiety_score 派生分类特征。"""
    out = df.copy()
    eps = 1.0

    daily = weekly = know = tech = eco = concern = income = None
    if "daily_commute_km" in out.columns:
        daily, _ = to_num(out["daily_commute_km"])
        out["log_daily_commute"] = np.log1p(daily.clip(lower=0))
    if "weekly_travel_distance_km" in out.columns:
        weekly, _ = to_num(out["weekly_travel_distance_km"])
        out["log_weekly_travel"] = np.log1p(weekly.clip(lower=0))
    if daily is not None and weekly is not None:
        out["avg_daily_weekly_travel"] = weekly / 7.0
        out["travel_pressure"] = daily + weekly / 7.0
        out["commute_share"] = 7.0 * daily / (weekly + eps)
        out["commute_ratio"] = out["commute_share"]
        out["weekly_minus_commute"] = weekly - 7.0 * daily
        out["long_trip_pressure"] = weekly / (daily + eps)

    if "nearest_charging_station_km" in out.columns and "charging_station_accessibility" in out.columns:
        dist, _ = to_num(out["nearest_charging_station_km"])
        acc, _ = to_num(out["charging_station_accessibility"])
        out["charging_gap"] = dist / (acc + eps)
        out["charging_convenience"] = acc / (dist + eps)
        out["distance_access_interaction"] = dist * (11.0 - acc)
        out["log_station_distance"] = np.log1p(dist.clip(lower=0))

    if "ev_knowledge_score" in out.columns:
        know, _ = to_num(out["ev_knowledge_score"])
        out["knowledge_deficit"] = 10.0 - know
    if "technology_affinity_score" in out.columns:
        tech, _ = to_num(out["technology_affinity_score"])
    if know is not None and tech is not None:
        out["knowledge_tech_gap"] = tech - know
        out["knowledge_tech_mean"] = (tech + know) / 2.0
        out["knowledge_tech_interaction"] = know * tech
    if "previous_ev_experience" in out.columns and know is not None:
        exp, _ = to_num(out["previous_ev_experience"])
        out["experience_knowledge"] = exp * know

    if "battery_replacement_concern" in out.columns:
        concern, _ = to_num(out["battery_replacement_concern"])
    if concern is not None and know is not None:
        out["concern_knowledge_ratio"] = concern / (know + eps)
        out["concern_knowledge_gap"] = concern - know
    if concern is not None and tech is not None:
        out["concern_tech_ratio"] = concern / (tech + eps)
        out["concern_tech_gap"] = concern - tech
    if concern is not None and daily is not None:
        out["concern_commute"] = concern * daily
    if concern is not None and weekly is not None:
        out["concern_weekly_travel"] = concern * weekly
    if "charging_station_accessibility" in out.columns and "travel_pressure" in out.columns:
        acc, _ = to_num(out["charging_station_accessibility"])
        out["charging_travel_pressure"] = out["travel_pressure"] / (acc + eps)
    if "nearest_charging_station_km" in out.columns and "travel_pressure" in out.columns:
        dist, _ = to_num(out["nearest_charging_station_km"])
        out["distance_travel_interaction"] = dist * out["travel_pressure"]

    if "annual_income" in out.columns:
        income, _ = to_num(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))
        out["log_income"] = out["log_annual_income"]
        out["monthly_income"] = income / 12.0
    if income is not None and "fuel_expense_per_month" in out.columns:
        fuel, _ = to_num(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)
        out["fuel_monthly_income_ratio"] = fuel / (out["monthly_income"] + eps)

    if "environmental_awareness_score" in out.columns:
        eco, _ = to_num(out["environmental_awareness_score"])
    if eco is not None and tech is not None:
        out["eco_tech_mean"] = (eco + tech) / 2.0
        out["eco_tech_interaction"] = eco * tech
        out["eco_minus_tech"] = eco - tech

    if "vehicle_age_years" in out.columns and "age" in out.columns:
        vage, _ = to_num(out["vehicle_age_years"])
        age, _ = to_num(out["age"])
        out["vehicle_age_ratio"] = vage / (age + eps)
        if "previous_ev_experience" in out.columns:
            exp, _ = to_num(out["previous_ev_experience"])
            out["experience_age_interaction"] = exp * age
            out["vehicle_familiarity"] = exp / (vage + eps)

    return out


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    official = list(TASK3_OFFICIAL_FEATURES)
    safe = resolve_safe_features(df)
    probe = add_mechanism_features(df.head(3))
    all_mech: List[str] = []
    for cols in TASK3_MECH_GROUPS.values():
        for c in cols:
            if c in probe.columns and c not in all_mech and c not in TASK3_EXCLUDED_FEATURES:
                all_mech.append(c)
    # 兼容旧稳定机制列名
    for c in TASK3_MECHANISM_FEATURES:
        if c in probe.columns and c not in all_mech:
            all_mech.append(c)
    mech = list(safe)
    for c in all_mech:
        if c not in mech:
            mech.append(c)
    assert_no_leakage(official)
    assert_no_leakage(safe)
    assert_no_leakage(mech)
    return {
        "A_official": official,
        "B_safe_raw": safe,
        "C_safe_plus_mechanism": mech,
        "mechanism_only_pool": all_mech,
    }


def group_feature_lists(base: Sequence[str], pool: Sequence[str]) -> Dict[str, List[str]]:
    """B + 各组 的消融候选。"""
    out: Dict[str, List[str]] = {"B_safe_raw": list(base)}
    for gname, gcols in TASK3_MECH_GROUPS.items():
        feats = list(base)
        for c in gcols:
            if c in pool and c not in feats:
                feats.append(c)
        out[f"B+{gname}"] = feats
    all_f = list(base)
    for c in pool:
        if c not in all_f:
            all_f.append(c)
    out["B+all_mechanism"] = all_f
    return out
