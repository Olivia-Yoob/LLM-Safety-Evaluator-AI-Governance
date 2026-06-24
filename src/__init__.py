"""
LLM Safety Evaluator & Governance — shared library.

Centralises the logic the notebooks rely on so each notebook reads as a thin,
governance-focused narrative rather than a wall of utility code:

    data_loader        – load the Jigsaw corpus, redact harmful text, policy map
    ml_baseline        – TF-IDF + Logistic Regression baseline classifier
    llm_client         – Claude API toxicity classifier (graceful w/o API key)
    safety_metrics     – per-category precision/recall/F1 + governance thresholds
    red_team_generator – systematic adversarial test-case generation

Author: Olivia Kim — Quantitative AI Risk Analyst portfolio, Project 3.
"""

from .data_loader import (
    LABELS,
    ANTHROPIC_POLICY_MAP,
    IDENTITY_TERMS,
    resolve_data_path,
    load_jigsaw,
    redact,
    repo_root,
    outputs_dir,
)

__all__ = [
    "LABELS",
    "ANTHROPIC_POLICY_MAP",
    "IDENTITY_TERMS",
    "resolve_data_path",
    "load_jigsaw",
    "redact",
    "repo_root",
    "outputs_dir",
]
