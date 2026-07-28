from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TASK3_KNOWN_CATEGORICAL
from src.tasks.task3.features import add_mechanism_features, to_num


@dataclass
class CatPrepState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)


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
    known = set(TASK3_KNOWN_CATEGORICAL)
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


def fit_cat_preprocessor(X_train: pd.DataFrame, feature_cols: Sequence[str]) -> CatPrepState:
    Xw = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in Xw.columns]
    num, cat = identify_types(Xw, cols)
    med = {
        c: float(s.median()) if (s := to_num(Xw[c])[0]).notna().any() else 0.0 for c in num
    }
    return CatPrepState(num, cat, med, cols)


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
            series, _ = to_num(Xw[col])
            out[col] = series.fillna(state.numeric_medians.get(col, 0.0))
        else:
            mask_missing = Xw[col].isna()
            as_str = Xw[col].astype(str).str.strip()
            as_str = as_str.replace(
                {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
            )
            out[col] = as_str.where(~mask_missing, "Unknown")
    return out.loc[:, state.final_feature_order]


def cat_indices(state: CatPrepState) -> List[int]:
    cat_set = set(state.categorical_columns)
    return [i for i, c in enumerate(state.final_feature_order) if c in cat_set]


def fit_tabular_preprocessor(
    X_train: pd.DataFrame, feature_cols: Sequence[str], scale: bool = False
) -> TabularPrepState:
    Xw = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in Xw.columns]
    num, cat = identify_types(Xw, cols)
    med = {
        c: float(s.median()) if (s := to_num(Xw[c])[0]).notna().any() else 0.0 for c in num
    }
    ohe = None
    if cat:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(Xw[cat].astype(str).fillna("Unknown"))
    scaler = None
    Xn = np.column_stack(
        [to_num(Xw[c])[0].fillna(med[c]).to_numpy() for c in num]
    ) if num else np.zeros((len(Xw), 0))
    if scale and Xn.shape[1] > 0:
        scaler = StandardScaler()
        scaler.fit(Xn)
    names = list(num)
    if ohe is not None:
        names.extend(list(ohe.get_feature_names_out(cat)))
    return TabularPrepState(num, cat, med, ohe, scaler, names)


def transform_tabular(X: pd.DataFrame, state: TabularPrepState) -> np.ndarray:
    Xw = add_mechanism_features(X)
    parts = []
    if state.numeric_columns:
        Xn = np.column_stack(
            [
                to_num(Xw[c])[0].fillna(state.numeric_medians.get(c, 0.0)).to_numpy()
                if c in Xw.columns
                else np.full(len(Xw), state.numeric_medians.get(c, 0.0))
                for c in state.numeric_columns
            ]
        )
        if state.scaler is not None:
            Xn = state.scaler.transform(Xn)
        parts.append(Xn)
    if state.ohe is not None and state.categorical_columns:
        Xc = Xw.reindex(columns=state.categorical_columns).astype(str).fillna("Unknown")
        parts.append(state.ohe.transform(Xc))
    if not parts:
        return np.zeros((len(Xw), 0))
    return np.hstack(parts)
