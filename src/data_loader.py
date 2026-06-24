"""
data_loader.py — load the Jigsaw corpus, redact harmful text, and hold the
canonical Jigsaw-to-Anthropic-Usage-Policy mapping.

Why this module exists for an LLM safety pipeline
-------------------------------------------------
A content-moderation classifier is a *control*. Like any control it needs a
single, auditable definition of (a) what data feeds it, (b) the harm taxonomy it
governs against, and (c) how harmful examples are handled when humans look at
them. Putting all three here — instead of copy-pasting across five notebooks —
means the control has one source of truth that can be reviewed and version-
controlled, which is exactly what a model-risk-management (MRM) standard expects.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The six Jigsaw toxicity labels, ordered the way the notebooks present them.
# ---------------------------------------------------------------------------
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ---------------------------------------------------------------------------
# ⭐ Canonical mapping: Jigsaw 6 categories → Anthropic Usage Policy.
# The Anthropic Usage Policy ("Universal Usage Standards") clauses are PARAPHRASED
# from the public policy at anthropic.com/legal/aup — treat as an analyst mapping,
# not a verbatim quote. EU AI Act articles show the harm also sits in a binding
# regulatory regime. severity_tier drives differential governance downstream.
# ---------------------------------------------------------------------------
ANTHROPIC_POLICY_MAP = {
    "toxic": {
        "definition": "Rude, disrespectful or unreasonable language likely to make "
                      "someone leave a discussion.",
        "anthropic_usage_policy_clause": "Do Not Generate Psychologically or "
                      "Emotionally Harmful Content (harassment / abusive interactions)",
        "policy_harm_family": "Harassment & Abuse",
        "eu_ai_act_touchpoint": "Art. 5 (harmful manipulation) · Art. 15 (robustness)",
        "severity_tier": "Medium",
    },
    "severe_toxic": {
        "definition": "Extreme, aggressive toxicity — a high-intensity version of toxic.",
        "anthropic_usage_policy_clause": "Do Not Generate Psychologically or "
                      "Emotionally Harmful Content (severe abuse)",
        "policy_harm_family": "Harassment & Abuse (severe)",
        "eu_ai_act_touchpoint": "Art. 5 · Art. 15",
        "severity_tier": "High",
    },
    "obscene": {
        "definition": "Profanity / sexually explicit or vulgar language.",
        "anthropic_usage_policy_clause": "Do Not Generate Sexually Explicit / Obscene "
                      "Content; protect minors from adult content",
        "policy_harm_family": "Adult / Explicit Content",
        "eu_ai_act_touchpoint": "Art. 50 (transparency) · child-safety provisions",
        "severity_tier": "Medium",
    },
    "threat": {
        "definition": "Statement of intent to inflict harm or violence on a person "
                      "or group.",
        "anthropic_usage_policy_clause": "Do Not Incite, Threaten or Facilitate "
                      "Violence; Do Not Compromise Personal Safety",
        "policy_harm_family": "Violence & Physical Harm",
        "eu_ai_act_touchpoint": "Art. 5 · Annex III (safety-critical)",
        "severity_tier": "Critical",
    },
    "insult": {
        "definition": "Insulting, inflammatory or demeaning language aimed at a person.",
        "anthropic_usage_policy_clause": "Do Not Generate Psychologically or "
                      "Emotionally Harmful Content (bullying / demeaning attacks)",
        "policy_harm_family": "Harassment & Abuse",
        "eu_ai_act_touchpoint": "Art. 5 · Art. 15",
        "severity_tier": "Medium",
    },
    "identity_hate": {
        "definition": "Hateful or discriminatory content targeting a protected "
                      "attribute (race, religion, gender, sexuality, etc.).",
        "anthropic_usage_policy_clause": "Do Not Generate Hateful Content or Promote "
                      "Discrimination against protected groups",
        "policy_harm_family": "Hate & Discrimination",
        "eu_ai_act_touchpoint": "Art. 5 · Art. 10 (data governance / non-discrimination) "
                      "· Art. 15",
        "severity_tier": "Critical",
    },
}

# Severity ordering used for escalation logic and sorting.
SEVERITY_ORDER = {"Medium": 1, "High": 2, "Critical": 3}


def policy_map_frame() -> pd.DataFrame:
    """Return the Jigsaw→Anthropic policy mapping as a tidy DataFrame."""
    rows = []
    for cat in LABELS:
        row = {"jigsaw_category": cat, **ANTHROPIC_POLICY_MAP[cat]}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Identity-term lexicon for the bias-risk scan. A deliberately conservative,
# documented screening proxy — NOT an exhaustive or perfect list.
# ---------------------------------------------------------------------------
IDENTITY_TERMS = {
    "religion": [r"muslim", r"islam", r"jew", r"jewish", r"christian", r"catholic",
                 r"hindu", r"buddhist", r"atheist", r"sikh"],
    "race_ethnicity": [r"black", r"white", r"asian", r"african", r"hispanic", r"latino",
                       r"arab", r"indian", r"chinese", r"mexican"],
    "gender_sexuality": [r"gay", r"lesbian", r"homosexual", r"trans", r"transgender",
                         r"queer", r"bisexual", r"woman", r"women", r"female"],
    "nationality": [r"american", r"european", r"immigrant", r"foreigner", r"refugee"],
    "disability_age": [r"disabled", r"autistic", r"elderly", r"old man", r"old woman"],
}


# ---------------------------------------------------------------------------
# Path helpers — robust to whether code runs from notebooks/ or the repo root.
# ---------------------------------------------------------------------------
def resolve_data_path(filename: str = "train.csv.zip") -> str:
    """Find a Jigsaw data file whether launched from notebooks/ or the repo root."""
    candidates = [
        os.path.join("..", "jigsaw-toxic-comment-classification-challenge", filename),
        os.path.join("jigsaw-toxic-comment-classification-challenge", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not locate {filename}. Looked in: {candidates}. "
        "Run from the repo root or the notebooks/ directory."
    )


def repo_root() -> str:
    """Absolute path to the project root (the folder holding the jigsaw dataset)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(resolve_data_path())))


def outputs_dir() -> str:
    """Absolute path to <repo>/outputs (created if missing). Shared artefact store."""
    d = os.path.join(repo_root(), "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def results_dir() -> str:
    """Absolute path to <repo>/results (created if missing). Figure/result store."""
    d = os.path.join(repo_root(), "results")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_jigsaw(path: str | None = None, nrows: int | None = None,
                add_length: bool = True) -> pd.DataFrame:
    """
    Load the Jigsaw train corpus and run integrity gates.

    Parameters
    ----------
    path : optional explicit path to train.csv(.zip). Auto-resolved if None.
    nrows : optional row cap (useful for quick API experiments in NB3).
    add_length : add char_len / word_len helper columns.

    Raises on duplicate ids or non-binary labels — a control built on corrupt
    data is worse than no control, so we fail loudly rather than silently.
    """
    path = path or resolve_data_path()
    df = pd.read_csv(path, nrows=nrows)

    missing = [c for c in (["id", "comment_text"] + LABELS) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    assert df["id"].duplicated().sum() == 0, "Duplicate ids found in corpus!"
    label_vals = set(pd.unique(df[LABELS].values.ravel()))
    assert label_vals.issubset({0, 1}), f"Labels not binary: {sorted(label_vals)}"

    if add_length:
        df["char_len"] = df["comment_text"].str.len()
        df["word_len"] = df["comment_text"].str.split().map(len)
    return df


# ---------------------------------------------------------------------------
# Redaction — harm-reduction for display. Masks a slur/profanity core list,
# strips obvious PII, collapses whitespace, truncates. Display-grade scrubbing,
# not a security guarantee — exactly what a T&S reviewer sees instead of raw text.
# ---------------------------------------------------------------------------
_PROFANITY_CORE = [
    "fuck", "shit", "bitch", "cunt", "asshole", "dick", "bastard", "slut",
    "whore", "nigger", "nigga", "faggot", "fag", "retard", "kike", "spic",
    "chink", "tranny", "cock", "pussy", "twat", "wanker", "moron",
]
_prof_re = re.compile(r"\b(" + "|".join(map(re.escape, _PROFANITY_CORE)) + r")\w*", re.I)
_email_re = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_handle_re = re.compile(r"@\w+")
_digits_re = re.compile(r"\b\d{4,}\b")


def redact(text: str, width: int = 240) -> str:
    """Return a redacted, truncated excerpt safe to show in a notebook/report."""
    t = str(text)
    t = _url_re.sub("[URL]", t)
    t = _email_re.sub("[EMAIL]", t)
    t = _handle_re.sub("[USER]", t)
    t = _digits_re.sub("[NUM]", t)
    t = _prof_re.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > width:
        t = t[:width].rstrip() + " […]"
    return t


def identity_mentions(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """
    Per-identity-group prevalence + conditional toxicity, used by the bias scan.
    Returns a tidy DataFrame; the caller decides how to visualise it.
    """
    text_lower = df[text_col].str.lower()
    n = len(df)
    rows = []
    for group, terms in IDENTITY_TERMS.items():
        pat = re.compile(r"\b(?:" + "|".join(terms) + r")\b")
        mentions = text_lower.str.contains(pat)
        k = int(mentions.sum())
        rows.append({
            "identity_group": group,
            "comments_mentioning": k,
            "pct_of_corpus": 100 * k / n if n else np.nan,
            "toxic_rate_given_mention_pct":
                df.loc[mentions, "toxic"].mean() * 100 if k else np.nan,
            "identity_hate_rate_given_mention_pct":
                df.loc[mentions, "identity_hate"].mean() * 100 if k else np.nan,
        })
    return (pd.DataFrame(rows)
            .sort_values("comments_mentioning", ascending=False)
            .reset_index(drop=True))
