"""
red_team_generator.py — systematic adversarial test-case generation.

Why a red-team framework is the project's strongest differentiator
------------------------------------------------------------------
Measuring a classifier on a held-out *random* test set tells you how it does on
average traffic. It tells you nothing about how it fails when an adversary is
*trying* to break it — which is the whole threat model for a safety control.
Systematic red-teaming probes specific, named failure modes:

  * obfuscation evasion (leetspeak/spacing) that should STILL be caught,
  * implicit/coded hate with no slur that should be caught,
  * neutral identity statements that must NOT be caught (fairness),
  * negations/condemnations that must NOT be caught (over-flagging).

This maps directly to Anthropic's red-team practice and to EU AI Act Article 15
(robustness & cybersecurity of high-risk AI systems). The suites live in
prompts/red_team_prompts.json so the test battery is version-controlled and
auditable — a reviewer can see exactly what the control was stress-tested against.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .data_loader import LABELS, repo_root


def _prompts_path() -> str:
    return os.path.join(repo_root(), "prompts", "red_team_prompts.json")


def load_suites(path: str | None = None) -> dict:
    """Load the raw red-team suite definitions from JSON."""
    path = path or _prompts_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expected_vector(expected: dict, should_flag: bool) -> dict:
    """
    Materialise a full 6-label expectation from a (possibly partial) spec.
    Missing labels default to 0. For should_flag suites with no explicit
    'expected', we only assert that SOMETHING is flagged (handled at scoring).
    """
    return {lab: int(expected.get(lab, 0)) for lab in LABELS}


def generate_cases(path: str | None = None) -> pd.DataFrame:
    """
    Expand the suite definitions into a flat DataFrame of adversarial test cases.

    Columns:
        case_id, attack_suite, failure_mode, eu_ai_act, should_flag,
        text, note, expected_toxic ... expected_identity_hate, asserts_specific

    'asserts_specific' is True when the suite pins exact expected labels (so we
    can score per-category), False when it only asserts the binary should/should-
    not-flag outcome (e.g. obfuscation where any flag is acceptable).
    """
    suites = load_suites(path)["attack_suites"]
    rows = []
    cid = 0
    for suite_name, suite in suites.items():
        failure_mode = suite.get("failure_mode", "")
        eu = suite.get("eu_ai_act", "")
        should_flag = bool(suite.get("should_flag", False))
        stype = suite.get("type", "cases")

        if stype == "template":
            for tmpl in suite.get("templates", []):
                for ident in suite.get("identities", []):
                    cid += 1
                    text = tmpl.format(identity=ident)
                    exp = _expected_vector(suite.get("expected", {}), should_flag)
                    rows.append(_row(cid, suite_name, failure_mode, eu, should_flag,
                                     text, f"identity={ident}", exp, asserts=True))
        else:  # explicit cases
            for case in suite.get("cases", []):
                cid += 1
                exp = _expected_vector(case.get("expected", {}), should_flag)
                # a should_flag case with no explicit labels only asserts "flag something"
                asserts = (not should_flag) or bool(case.get("expected"))
                rows.append(_row(cid, suite_name, failure_mode, eu, should_flag,
                                 case["text"], case.get("note", ""), exp, asserts))
    return pd.DataFrame(rows)


def _row(cid, suite_name, failure_mode, eu, should_flag, text, note, exp, asserts):
    row = {
        "case_id": f"RT{cid:03d}",
        "attack_suite": suite_name,
        "failure_mode": failure_mode,
        "eu_ai_act": eu,
        "should_flag": should_flag,
        "text": text,
        "note": note,
        "asserts_specific": asserts,
    }
    for lab in LABELS:
        row[f"expected_{lab}"] = exp[lab]
    return row


def score_predictions(cases: pd.DataFrame, pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model predictions (DataFrame with the 6 LABEL columns, aligned by row
    order to `cases`) against red-team expectations.

    Adds:
        any_flagged       — did the model flag >=1 category?
        outcome_correct   — binary should/should-not-flag correctness
        exact_match       — all 6 labels match expectation (only where asserts_specific)
        result            — 'PASS' / 'FAIL' for the binary governance outcome
    """
    if len(cases) != len(pred_frame):
        raise ValueError("cases and pred_frame must align row-for-row.")
    pred = pred_frame.reset_index(drop=True)[LABELS].astype(int)
    cases = cases.reset_index(drop=True).copy()

    any_flagged = (pred.sum(axis=1) > 0)
    cases["any_flagged"] = any_flagged.values

    # binary outcome: should_flag matches whether anything was flagged
    cases["outcome_correct"] = (any_flagged.values == cases["should_flag"].values)

    # exact per-label match (meaningful only where the suite asserts specifics)
    exp_cols = [f"expected_{l}" for l in LABELS]
    exact = (pred.values == cases[exp_cols].values).all(axis=1)
    cases["exact_match"] = [bool(e) if a else None
                            for e, a in zip(exact, cases["asserts_specific"])]
    cases["result"] = cases["outcome_correct"].map({True: "PASS", False: "FAIL"})
    return cases


def suite_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pass-rate per attack suite — the red-team scorecard."""
    g = scored.groupby("attack_suite")
    out = g.agg(
        n_cases=("case_id", "count"),
        should_flag=("should_flag", "first"),
        passes=("outcome_correct", "sum"),
    )
    out["pass_rate_pct"] = (100 * out["passes"] / out["n_cases"]).round(1)
    return out.reset_index().sort_values("pass_rate_pct")
