from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.tasks.task1.catboost_models import default_catboost_params, oof_catboost_multiclass
from src.tasks.task1.checkpoints import fold_is_complete
from src.tasks.task1.pipeline import make_folds


def main() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    labels = np.array(["Low", "Medium", "High"])[y]
    y_s = pd.Series(labels)
    folds = make_folds(y_s, n_folds=3, seed=42)
    feats = list(df.columns)
    params = default_catboost_params(42)
    params["iterations"] = 30

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "m1"
        sample_index = np.arange(len(df))
        oof1, pred1, rows1, _ = oof_catboost_multiclass(
            df,
            y_s,
            feats,
            folds,
            params,
            tag="tiny",
            checkpoint_dir=ckpt,
            sample_index=sample_index,
            resume_folds=True,
        )
        assert all(fold_is_complete(ckpt, i) for i in range(3))

        oof2, pred2, rows2, _ = oof_catboost_multiclass(
            df,
            y_s,
            feats,
            folds,
            params,
            tag="tiny",
            checkpoint_dir=ckpt,
            sample_index=sample_index,
            resume_folds=True,
        )
        assert np.allclose(oof1, oof2)
        assert list(pred1) == list(pred2)
        assert len(rows2) == 3
        print("OK tiny resume checkpoint test")


if __name__ == "__main__":
    main()
