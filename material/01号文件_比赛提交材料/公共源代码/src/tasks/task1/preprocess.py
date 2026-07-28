from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TASK1_KNOWN_CATEGORICAL
from src.tasks.task1.features import add_mechanism_features, to_num


@dataclass
class CatPrepState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)
    fill_numeric_median: bool = True


@dataclass
class TabularPrepState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    ohe: Optional[OneHotEncoder] = None
    scaler: Optional[StandardScaler] = None
    feature_names: List[str] = field(default_factory=list)


def identify_types(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Tuple[List[str], List[str]]:
    cat, num = [], []
    known = set(TASK1_KNOWN_CATEGORICAL)
    for col in feature_cols:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        is_cat = (
            col in known
            or pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        )
        (cat if is_cat else num).append(col)
    return num, cat


def fit_cat_preprocessor(
    X_train: pd.DataFrame,
    feature_cols: Sequence[str],
    fill_numeric_median: bool = True,
) -> CatPrepState:
    Xw = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in Xw.columns]
    num, cat = identify_types(Xw, cols)
    med = {
        c: float(s.median()) if (s := to_num(Xw[c])).notna().any() else 0.0 for c in num
    }
    return CatPrepState(num, cat, med, cols, fill_numeric_median)


def transform_cat(X: pd.DataFrame, state: CatPrepState) -> pd.DataFrame:
    Xw = add_mechanism_features(X)
    out = pd.DataFrame(index=Xw.index)
    for col in state.final_feature_order:
        if col not in Xw.columns:
            out[col] = (
                state.numeric_medians.get(col, 0.0)
                if col in state.numeric_columns
                else "Unknown"
            )
            continue
        if col in state.numeric_columns:
            s = to_num(Xw[col])
            if state.fill_numeric_median:
                out[col] = s.fillna(state.numeric_medians.get(col, 0.0))
            else:

                out[col] = s
        else:
            miss = Xw[col].isna()
            as_str = Xw[col].astype(str).str.strip().replace(
                {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
            )
            out[col] = as_str.where(~miss, "Unknown")
    return out.loc[:, state.final_feature_order]


def cat_indices(state: CatPrepState) -> List[int]:
    s = set(state.categorical_columns)
    return [i for i, c in enumerate(state.final_feature_order) if c in s]


def fit_tabular_preprocessor(
    X_train: pd.DataFrame,
    feature_cols: Sequence[str],
    scale: bool = False,
) -> TabularPrepState:
    Xw = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in Xw.columns]
    num, cat = identify_types(Xw, cols)
    med = {
        c: float(s.median()) if (s := to_num(Xw[c])).notna().any() else 0.0 for c in num
    }
    Xn = pd.DataFrame({c: to_num(Xw[c]).fillna(med[c]) for c in num}, index=Xw.index)
    Xc = pd.DataFrame(index=Xw.index)
    for c in cat:
        miss = Xw[c].isna()
        as_str = Xw[c].astype(str).str.strip().replace(
            {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
        )
        Xc[c] = as_str.where(~miss, "Unknown")

    ohe = None
    parts = [Xn] if num else []
    if cat:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh = ohe.fit_transform(Xc[cat])
        oh_names = list(ohe.get_feature_names_out(cat))
        parts.append(pd.DataFrame(oh, columns=oh_names, index=Xw.index))
    mat = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=Xw.index)
    scaler = None
    if scale and mat.shape[1] > 0:
        scaler = StandardScaler()
        mat_v = scaler.fit_transform(mat)
        mat = pd.DataFrame(mat_v, columns=mat.columns, index=mat.index)
    return TabularPrepState(num, cat, med, ohe, scaler, list(mat.columns))


def transform_tabular(X: pd.DataFrame, state: TabularPrepState) -> pd.DataFrame:
    Xw = add_mechanism_features(X)
    Xn = pd.DataFrame(
        {
            c: to_num(Xw[c]).fillna(state.numeric_medians.get(c, 0.0))
            for c in state.numeric_columns
        },
        index=Xw.index,
    )
    parts = [Xn] if state.numeric_columns else []
    if state.categorical_columns and state.ohe is not None:
        Xc = pd.DataFrame(index=Xw.index)
        for c in state.categorical_columns:
            if c not in Xw.columns:
                Xc[c] = "Unknown"
                continue
            miss = Xw[c].isna()
            as_str = Xw[c].astype(str).str.strip().replace(
                {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
            )
            Xc[c] = as_str.where(~miss, "Unknown")
        oh = state.ohe.transform(Xc[state.categorical_columns])
        oh_names = list(state.ohe.get_feature_names_out(state.categorical_columns))
        parts.append(pd.DataFrame(oh, columns=oh_names, index=Xw.index))
    mat = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=Xw.index)
    mat = mat.reindex(columns=state.feature_names, fill_value=0.0)
    if state.scaler is not None and mat.shape[1] > 0:
        mat = pd.DataFrame(
            state.scaler.transform(mat), columns=state.feature_names, index=mat.index
        )
    return mat
