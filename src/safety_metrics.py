"""
safety_metrics.py — the measurement layer for the safety control.

Why bespoke metrics instead of a bare sklearn classification_report
-------------------------------------------------------------------
A safety classifier is judged differently from an accuracy-maximising model:

  * **Recall on rare, high-severity harms** matters far more than overall accuracy
    (a missed `threat` is a real-world safety event; a missed `obscene` is not).
  * **Per-group false-positive rate** is a fairness control (Project-2 fair-lending
    logic), not an afterthought.
  * Thresholds are a **governance lever**, set per category by severity tier
    (the KB-Life-KRI-inspired logic), not a single global 0.5.

This module makes those first-class so Notebook 5 can express governance policy
directly in code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from .data_loader import LABELS, ANTHROPIC_POLICY_MAP, SEVERITY_ORDER


def per_category_metrics(Y_true, Y_pred, Y_proba=None) -> pd.DataFrame:
    """
    Precision / recall / F1 / support per toxicity category (+ ROC-AUC & PR-AUC
    when probabilities are supplied). Annotated with the Anthropic severity tier
    so the table reads as a governance scorecard, not just an ML report.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    rows = []
    for i, lab in enumerate(LABELS):
        yt, yp = Y_true[:, i], Y_pred[:, i]
        row = {
            "category": lab,
            "severity_tier": ANTHROPIC_POLICY_MAP[lab]["severity_tier"],
            "support_pos": int(yt.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        }
        if Y_proba is not None:
            yp_score = np.asarray(Y_proba)[:, i]
            # AUCs require both classes present
            if len(np.unique(yt)) == 2:
                row["roc_auc"] = roc_auc_score(yt, yp_score)
                row["pr_auc"] = average_precision_score(yt, yp_score)
            else:
                row["roc_auc"] = np.nan
                row["pr_auc"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_sev"] = df["severity_tier"].map(SEVERITY_ORDER)
    return df.sort_values(["_sev", "category"], ascending=[False, True]).drop(columns="_sev").reset_index(drop=True)


def macro_micro_summary(Y_true, Y_pred) -> pd.Series:
    """Single-row macro/micro F1 summary for headline reporting."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    return pd.Series({
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_recall": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
    })


def confusion_per_category(Y_true, Y_pred) -> dict:
    """Return {label: 2x2 confusion matrix} for confusion-matrix plotting."""
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    out = {}
    for i, lab in enumerate(LABELS):
        out[lab] = confusion_matrix(Y_true[:, i], Y_pred[:, i], labels=[0, 1])
    return out


# ---------------------------------------------------------------------------
# Governance thresholds — severity-tiered, the KB-Life-KRI logic applied to LLM.
# A Critical-harm head runs at a lower decision threshold (higher recall, accepts
# more false positives) because the cost of a missed threat dwarfs the cost of an
# over-flag. A Medium-harm head can sit at the standard 0.5.
# ---------------------------------------------------------------------------
DEFAULT_TIER_THRESHOLDS = {"Critical": 0.30, "High": 0.40, "Medium": 0.50}


def tiered_thresholds(tier_map: dict | None = None) -> dict:
    """Map each label → decision threshold based on its severity tier."""
    tier_map = tier_map or DEFAULT_TIER_THRESHOLDS
    return {lab: tier_map[ANTHROPIC_POLICY_MAP[lab]["severity_tier"]] for lab in LABELS}


def apply_tiered_thresholds(Y_proba, tier_map: dict | None = None) -> np.ndarray:
    """Binarise a (n, 6) probability matrix using per-category severity thresholds."""
    thr = tiered_thresholds(tier_map)
    Y_proba = np.asarray(Y_proba)
    out = np.zeros_like(Y_proba, dtype=int)
    for i, lab in enumerate(LABELS):
        out[:, i] = (Y_proba[:, i] >= thr[lab]).astype(int)
    return out


# ---------------------------------------------------------------------------
# Fairness: per-identity-group false-positive rate. The LLM analogue of the
# disparate-impact monitoring from the Project-2 fair-lending work.
# ---------------------------------------------------------------------------
def group_false_positive_rates(texts, y_true, y_pred, identity_terms,
                               label: str = "toxic") -> pd.DataFrame:
    """
    For each identity group, FPR among comments that mention it.
    A clean (y_true==0) comment that the model flags (y_pred==1) is a false
    positive — exactly the failure that silences the group it should protect.
    """
    import re
    texts = pd.Series(texts).reset_index(drop=True).str.lower()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for group, terms in identity_terms.items():
        pat = re.compile(r"\b(?:" + "|".join(terms) + r")\b")
        mask = texts.str.contains(pat).values
        neg = mask & (y_true == 0)            # truly-clean, mentions the group
        n_neg = int(neg.sum())
        fp = int((neg & (y_pred == 1)).sum())
        rows.append({
            "identity_group": group,
            "clean_mentions": n_neg,
            "false_positives": fp,
            "false_positive_rate_pct": 100 * fp / n_neg if n_neg else np.nan,
        })
    overall_neg = (y_true == 0)
    overall_fpr = 100 * ((overall_neg & (y_pred == 1)).sum() / overall_neg.sum())
    df = pd.DataFrame(rows)
    df.attrs["overall_fpr_pct"] = overall_fpr
    df.attrs["label"] = label
    return df.sort_values("false_positive_rate_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drift KRI — statistical-threshold alerting on safety-metric drift over time.
# KB-Life-KRI inspired: a metric that drops more than `tier`-dependent tolerance
# below its established baseline trips an escalation.
# ---------------------------------------------------------------------------
DEFAULT_DRIFT_TOLERANCE = {"Critical": 0.03, "High": 0.05, "Medium": 0.08}


def drift_kri(baseline_metrics: pd.DataFrame, current_metrics: pd.DataFrame,
              metric: str = "recall", tolerance: dict | None = None) -> pd.DataFrame:
    """
    Compare a current scorecard to a baseline and flag categories whose `metric`
    has degraded beyond the severity-tiered tolerance. Returns a KRI table with a
    RAG (red/amber/green) status per category — the executive-accountability view.
    """
    tolerance = tolerance or DEFAULT_DRIFT_TOLERANCE
    base = baseline_metrics.set_index("category")
    cur = current_metrics.set_index("category")
    rows = []
    for lab in LABELS:
        tier = ANTHROPIC_POLICY_MAP[lab]["severity_tier"]
        tol = tolerance[tier]
        b = float(base.loc[lab, metric])
        c = float(cur.loc[lab, metric])
        drop = b - c
        if drop > tol:
            status = "RED — escalate"
        elif drop > tol / 2:
            status = "AMBER — watch"
        else:
            status = "GREEN — ok"
        rows.append({
            "category": lab, "severity_tier": tier,
            f"baseline_{metric}": round(b, 3), f"current_{metric}": round(c, 3),
            "drop": round(drop, 3), "tolerance": tol, "kri_status": status,
        })
    return pd.DataFrame(rows)
