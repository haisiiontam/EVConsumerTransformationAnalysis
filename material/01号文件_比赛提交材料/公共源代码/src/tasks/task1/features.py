from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import (
    TASK1_EXCLUDED_FEATURES,
    TASK1_MECH_GROUPS,
    TASK1_OFFICIAL_FEATURES,
    TASK1_SAFE_CANDIDATE_FEATURES,
    TASK1_STRONG_FEATURES,
    TASK1_UNCERTAIN_FEATURES,
)

logger = logging.getLogger(__name__)


def to_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    return pd.to_numeric(as_str, errors="coerce")


def resolve_safe_raw(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in TASK1_SAFE_CANDIDATE_FEATURES:
        if col in TASK1_EXCLUDED_FEATURES or col in TASK1_UNCERTAIN_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("安全特征缺失，跳过：%s", col)
            continue
        if col not in cols:
            cols.append(col)
    for col in TASK1_STRONG_FEATURES:
        if (
            col in df.columns
            and col not in cols
            and col not in TASK1_EXCLUDED_FEATURES
            and col not in TASK1_UNCERTAIN_FEATURES
        ):
            cols.append(col)
    return cols


def _experience_01(series: pd.Series) -> pd.Series | None:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        uniq = set(v.dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return v.fillna(0).clip(0, 1)
        return None
    mapped = s.astype(str).str.strip().str.lower().map(
        {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
        }
    )
    if mapped.notna().mean() < 0.8:
        return None
    return mapped.fillna(0)


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造全部 C1–C6 机制特征（行内，无目标/全表统计）。"""
    out = df.copy()
    eps = 1.0
    income = know = eco = tech = anxiety = battery = None

    if "annual_income" in out.columns:
        income = to_num(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))
        out["monthly_income"] = income / 12.0

    if "fuel_expense_per_month" in out.columns and income is not None:
        fuel = to_num(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)
        monthly = out["monthly_income"]
        out["fuel_monthly_income_ratio"] = fuel / (monthly + eps)

    if (
        "environmental_awareness_score" in out.columns
        and "technology_affinity_score" in out.columns
    ):
        eco = to_num(out["environmental_awareness_score"])
        tech = to_num(out["technology_affinity_score"])
        out["eco_tech_mean"] = (eco + tech) / 2.0
        out["eco_tech_interaction"] = eco * tech
        out["eco_minus_tech"] = eco - tech

    if "ev_knowledge_score" in out.columns:
        know = to_num(out["ev_knowledge_score"])
        out["knowledge_deficit"] = 10.0 - know

    if tech is not None and know is not None:
        out["knowledge_tech_gap"] = tech - know
        out["knowledge_tech_mean"] = (tech + know) / 2.0

    if "previous_ev_experience" in out.columns and tech is not None:
        exp01 = _experience_01(out["previous_ev_experience"])
        if exp01 is not None:
            out["experience_tech_interaction"] = exp01 * tech

    if "range_anxiety_score" in out.columns:
        anxiety = to_num(out["range_anxiety_score"])
    if "battery_replacement_concern" in out.columns:
        battery = to_num(out["battery_replacement_concern"])
    if anxiety is not None and battery is not None:
        out["anxiety_concern_interaction"] = anxiety * battery
    if anxiety is not None and know is not None:
        out["anxiety_knowledge_ratio"] = anxiety / (know + eps)
        out["anxiety_knowledge_gap"] = anxiety * (10.0 - know)
    if battery is not None and know is not None:
        out["concern_knowledge_ratio"] = battery / (know + eps)

    if "daily_commute_km" in out.columns and "weekly_travel_distance_km" in out.columns:
        daily = to_num(out["daily_commute_km"])
        weekly = to_num(out["weekly_travel_distance_km"])
        out["average_daily_travel"] = weekly / 7.0
        out["travel_pressure"] = daily + weekly / 7.0
        out["commute_ratio"] = 7.0 * daily / (weekly + eps)

    if (
        "nearest_charging_station_km" in out.columns
        and "charging_station_accessibility" in out.columns
    ):
        near = to_num(out["nearest_charging_station_km"])
        access = to_num(out["charging_station_accessibility"])
        out["charging_gap"] = near / (access + eps)
        out["charging_convenience"] = access / (near + eps)

    if "government_incentive_awareness" in out.columns and "log_annual_income" in out.columns:
        incent = to_num(out["government_incentive_awareness"])
        out["incentive_income_interaction"] = incent * out["log_annual_income"]

    return out


def available_group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    probe = add_mechanism_features(df.head(5))
    out: Dict[str, List[str]] = {}
    for gname, feats in TASK1_MECH_GROUPS.items():
        out[gname] = [f for f in feats if f in probe.columns]
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """构建消融用特征集字典。"""
    a = [c for c in TASK1_OFFICIAL_FEATURES if c in df.columns]
    b = resolve_safe_raw(df)
    groups = available_group_features(df)
    sets: Dict[str, List[str]] = {"A_official": a, "B_safe_raw": list(b)}
    for gname, gfeats in groups.items():
        sets[f"B+{gname}"] = list(b) + [f for f in gfeats if f not in b]
    all_mech: List[str] = []
    for gfeats in groups.values():
        for f in gfeats:
            if f not in all_mech:
                all_mech.append(f)
    sets["B+all_mech"] = list(b) + [f for f in all_mech if f not in b]
    return sets


def uncertain_feature_report(df: pd.DataFrame) -> Dict[str, object]:
    present = [c for c in TASK1_UNCERTAIN_FEATURES if c in df.columns]
    return {
        "uncertain_features": TASK1_UNCERTAIN_FEATURES,
        "present_in_data": present,
        "policy": "listed but never auto-added to models",
        "also_excluded": list(TASK1_EXCLUDED_FEATURES),
    }


def select_kept_groups(
    ablation_df: pd.DataFrame, base_name: str = "B_safe_raw", eps: float = 1e-12
) -> Tuple[List[str], List[str]]:
    """根据消融表选出相对 B 有提升的机制组。"""
    base_row = ablation_df[ablation_df["feature_set"] == base_name]
    if base_row.empty:
        return [], []
    base_acc = float(base_row.iloc[0]["oof_accuracy"])
    kept, rejected = [], []
    for _, row in ablation_df.iterrows():
        name = str(row["feature_set"])
        if not name.startswith("B+C"):
            continue
        g = name.replace("B+", "")
        if float(row["oof_accuracy"]) > base_acc + eps:
            kept.append(g)
        else:
            rejected.append(g)
    return kept, rejected


def features_from_kept_groups(df: pd.DataFrame, kept_groups: Sequence[str]) -> List[str]:
    b = resolve_safe_raw(df)
    groups = available_group_features(df)
    feats = list(b)
    for g in kept_groups:
        for f in groups.get(g, []):
            if f not in feats:
                feats.append(f)
    return feats
