\
"""
PUBG WinPlacePerc - productionized pipeline utilities.

This module is designed to be imported by a Streamlit app (app.py).

Key decisions (important for Colab / low-RAM environments):
- Drop high-cardinality IDs (Id/matchId/groupId) BEFORE one-hot encoding.
- Keep OneHotEncoder sparse output (default). Dense one-hot can easily exhaust RAM.
- Use Ridge as the default model because it supports sparse matrices and is fast.

You can extend this to LightGBM/XGBoost later if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

TARGET_COL = "winPlacePerc"
ID_COL = "Id"
GROUP_COL = "groupId"
MATCH_COL = "matchId"

DROP_ID_COLS: List[str] = [ID_COL, MATCH_COL, GROUP_COL]


def add_group_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering based on common PUBG baselines:
    - rankPoints: replace -1 with 0
    - group_size, match_size (if Id/matchId/groupId exist)
    - group mean & max for base numeric features
    - a few ratio features
    """
    df = df.copy()

    if "rankPoints" in df.columns:
        df["rankPoints"] = df["rankPoints"].replace(-1, 0)

    # base numeric columns (exclude IDs + target)
    exclude = set([ID_COL, MATCH_COL, GROUP_COL, TARGET_COL])
    base_num = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]

    if MATCH_COL in df.columns and GROUP_COL in df.columns and ID_COL in df.columns:
        df["group_size"] = df.groupby([MATCH_COL, GROUP_COL])[ID_COL].transform("count")
        df["match_size"] = df.groupby(MATCH_COL)[ID_COL].transform("count")

        if base_num:
            gmean = df.groupby([MATCH_COL, GROUP_COL])[base_num].transform("mean")
            gmean.columns = [f"{c}_gmean" for c in base_num]
            df = pd.concat([df, gmean], axis=1)

            gmax = df.groupby([MATCH_COL, GROUP_COL])[base_num].transform("max")
            gmax.columns = [f"{c}_gmax" for c in base_num]
            df = pd.concat([df, gmax], axis=1)

    # safe ratio features
    if "kills" in df.columns and "walkDistance" in df.columns:
        df["kills_per_walk"] = df["kills"] / (df["walkDistance"] + 1.0)
    if "damageDealt" in df.columns and "walkDistance" in df.columns:
        df["damage_per_walk"] = df["damageDealt"] / (df["walkDistance"] + 1.0)
    if "heals" in df.columns and "boosts" in df.columns:
        df["heals_boosts"] = df["heals"] + df["boosts"]

    return df


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_ID_COLS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessing:
    - Numeric: median impute + standardize (with_mean=False keeps sparse compatibility)
    - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore') (sparse)
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # sparse output
        ]
    )

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
    )


def build_model(model_name: str = "Ridge"):
    name = model_name.strip().lower()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    raise ValueError(f"Unsupported model_name={model_name!r}. Only 'Ridge' is included in this template.")


@dataclass
class TrainResult:
    pipe: Pipeline
    valid_mae: float


def train_pipeline(
    train_df: pd.DataFrame,
    model_name: str = "Ridge",
    valid_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """
    End-to-end training:
    1) Feature engineering
    2) Group-aware split by matchId (prevents leakage)
    3) Drop Id/matchId/groupId before one-hot
    4) Fit pipeline and compute validation MAE
    """
    df = add_group_match_features(train_df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Train CSV must contain target column: {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])

    if MATCH_COL in df.columns:
        groups = df[MATCH_COL]
        gss = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
        tr_idx, va_idx = next(gss.split(X, y, groups=groups))
    else:
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        cut = int(len(df) * (1 - valid_size))
        tr_idx, va_idx = idx[:cut], idx[cut:]

    X_train = drop_id_columns(X.iloc[tr_idx].copy())
    X_valid = drop_id_columns(X.iloc[va_idx].copy())
    y_train = y.iloc[tr_idx].copy()
    y_valid = y.iloc[va_idx].copy()

    preprocess = build_preprocess(X_train)
    model = build_model(model_name)

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_valid)
    valid_mae = float(mean_absolute_error(y_valid, pred))
    return TrainResult(pipe=pipe, valid_mae=valid_mae)


def predict_winplace(pipe: Pipeline, test_df: pd.DataFrame) -> np.ndarray:
    df = add_group_match_features(test_df)
    X = drop_id_columns(df)
    preds = pipe.predict(X)
    return np.clip(preds, 0.0, 1.0)


def make_submission(test_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    if ID_COL not in test_df.columns:
        raise ValueError(f"Test CSV must contain '{ID_COL}' column.")
    return pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: preds})


def save_model(pipe: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(pipe, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)
