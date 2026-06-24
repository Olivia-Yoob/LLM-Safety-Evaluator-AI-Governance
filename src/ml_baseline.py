"""
ml_baseline.py — the classical-ML control: TF-IDF + One-vs-Rest Logistic Regression.

Why a classical baseline matters for LLM governance
---------------------------------------------------
Before you can claim an LLM is a *good* safety classifier, you need a credible
*floor* to measure it against. A cheap, fast, fully-interpretable TF-IDF + linear
model is that floor. In governance terms it is the "champion" the LLM
"challenger" must beat to justify its cost, latency, and opacity — the same
champion/challenger discipline used for credit and fraud models in Projects 1–2,
now applied to an LLM control. It is also the fallback control: if the Claude API
is unavailable, this model still runs on-prem with no external dependency.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .data_loader import LABELS, repo_root


def build_pipeline(max_features: int = 30000, ngram_range=(1, 2),
                   C: float = 4.0, class_weight: str | None = "balanced") -> Pipeline:
    """
    TF-IDF (word 1-2 grams) → One-vs-Rest Logistic Regression.

    class_weight='balanced' is a deliberate governance choice: the corpus is ~90%
    clean, so without it the model optimises for the majority (clean) class and
    under-detects the rare, high-severity harms we care most about.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=3,
            stop_words="english",
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=1000,
                solver="liblinear",
            ),
            # n_jobs=1: only 6 binary heads (fast), and avoids a numpy
            # read-only-writeback error under joblib's parallel backend.
            n_jobs=1,
        )),
    ])


def split_xy(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratify-free split on text X and the 6-column label matrix Y."""
    X = df["comment_text"].fillna("")
    Y = df[LABELS].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, Y_train, Y_test


def train_baseline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                   **pipe_kwargs):
    """
    Fit the baseline and return (pipeline, splits).
    splits = dict(X_train, X_test, Y_train, Y_test).
    """
    X_train, X_test, Y_train, Y_test = split_xy(df, test_size, random_state)
    pipe = build_pipeline(**pipe_kwargs)
    pipe.fit(X_train, Y_train.values)
    splits = {"X_train": X_train, "X_test": X_test,
              "Y_train": Y_train, "Y_test": Y_test}
    return pipe, splits


def predict(pipe: Pipeline, X, threshold: float = 0.5):
    """Return (proba, pred) DataFrames indexed like X, columns = LABELS."""
    proba = pipe.predict_proba(X)
    proba = pd.DataFrame(np.asarray(proba), columns=LABELS, index=getattr(X, "index", None))
    pred = (proba >= threshold).astype(int)
    return proba, pred


def save_model(pipe: Pipeline, name: str = "tfidf_logreg_baseline.joblib") -> str:
    """Persist the fitted pipeline to <repo>/models/ (gitignored)."""
    mdir = os.path.join(repo_root(), "models")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, name)
    joblib.dump(pipe, path)
    return path


def load_model(name: str = "tfidf_logreg_baseline.joblib") -> Pipeline:
    path = os.path.join(repo_root(), "models", name)
    return joblib.load(path)


def top_features(pipe: Pipeline, label: str, k: int = 15) -> pd.DataFrame:
    """
    Most positively-weighted tokens for one category — the interpretability
    surface that lets a reviewer sanity-check *why* the model flags content.
    """
    if label not in LABELS:
        raise ValueError(f"{label} not in {LABELS}")
    vec = pipe.named_steps["tfidf"]
    ovr = pipe.named_steps["clf"]
    est = ovr.estimators_[LABELS.index(label)]
    names = np.array(vec.get_feature_names_out())
    coefs = est.coef_.ravel()
    top_idx = np.argsort(coefs)[::-1][:k]
    return pd.DataFrame({"token": names[top_idx], "weight": coefs[top_idx].round(3)})
