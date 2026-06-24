"""
llm_client.py — Claude API toxicity classifier (the LLM "challenger" control).

Design goals for an LLM safety control
--------------------------------------
1. **Policy-native output.** The model judges content against *Anthropic Usage
   Policy clauses*, not abstract dataset labels — so its output speaks the
   language a safety team governs in. Structured output (a Pydantic schema)
   guarantees a parseable, schema-valid verdict every time.
2. **Deterministic & auditable.** Fixed model id, fixed prompt, structured schema
   — the same comment yields a reproducible, loggable verdict.
3. **Graceful degradation.** With no ANTHROPIC_API_KEY the notebooks must still
   run end to end. `is_available()` lets callers branch, and a deterministic
   offline heuristic stands in so NB3–5 produce data without a key or spend.

Model: claude-opus-4-8 (Anthropic's most capable model) per the project's
"default to the latest, most capable Claude model" guidance.
"""
from __future__ import annotations

import os
import re
from typing import List

from pydantic import BaseModel, Field

from .data_loader import LABELS, ANTHROPIC_POLICY_MAP

MODEL_ID = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Structured output schema — the contract the model must return.
# ---------------------------------------------------------------------------
class ToxicityAssessment(BaseModel):
    """A single comment's policy assessment, one bool per Jigsaw category."""
    toxic: bool = Field(description="Rude/disrespectful/unreasonable language.")
    severe_toxic: bool = Field(description="Extreme, aggressive toxicity.")
    obscene: bool = Field(description="Profanity / sexually explicit / vulgar.")
    threat: bool = Field(description="Intent to inflict harm or violence.")
    insult: bool = Field(description="Insulting/inflammatory attack on a person.")
    identity_hate: bool = Field(
        description="Hate/discrimination targeting a protected attribute.")
    policy_clauses: List[str] = Field(
        default_factory=list,
        description="Anthropic Usage Policy clauses the comment plausibly violates.")
    rationale: str = Field(
        default="", description="One-sentence, non-graphic justification.")

    def label_vector(self) -> list[int]:
        return [int(getattr(self, lab)) for lab in LABELS]


# ---------------------------------------------------------------------------
# System prompt — the policy-native instruction. Built from the canonical map so
# the prompt and the mapping table can never drift apart.
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    lines = [
        "You are a Trust & Safety content-moderation classifier operating under the "
        "Anthropic Usage Policy. You assess a single user comment and decide which "
        "of six harm categories apply. You are precise, calibrated, and you DO NOT "
        "reproduce or amplify harmful content — your rationale must stay non-graphic.",
        "",
        "Harm categories (Jigsaw label → Anthropic Usage Policy clause):",
    ]
    for lab in LABELS:
        m = ANTHROPIC_POLICY_MAP[lab]
        lines.append(f"- {lab} [{m['severity_tier']}]: {m['definition']} "
                     f"→ Policy: {m['anthropic_usage_policy_clause']}")
    lines += [
        "",
        "Critical fairness rule: a NEUTRAL mention of an identity group "
        "(e.g. 'I am a proud Muslim woman') is NOT identity_hate and usually not "
        "toxic. Only flag identity_hate for hostile/discriminatory content toward "
        "a protected group. Do not penalise reclaimed language or self-description.",
        "",
        "Return a structured assessment: one boolean per category, the list of "
        "Anthropic Usage Policy clauses the comment plausibly violates, and a single "
        "non-graphic sentence of rationale.",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------
class ToxicityClassifier:
    """
    Thin wrapper around the Anthropic Messages API for policy-native toxicity
    classification, with a deterministic offline fallback.

    Usage:
        clf = ToxicityClassifier()           # reads ANTHROPIC_API_KEY from env
        if clf.is_available():
            result = clf.classify("some comment")     # live Claude call
        else:
            result = clf.classify_offline("some comment")  # heuristic stand-in
    """

    def __init__(self, model: str = MODEL_ID, api_key: str | None = None):
        self.model = model
        self._client = None
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self._key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._key)
            except Exception:
                self._client = None

    def is_available(self) -> bool:
        """True only if an API key is present AND the SDK initialised."""
        return self._client is not None

    # -- live path ---------------------------------------------------------
    def classify(self, comment: str, max_tokens: int = 1024) -> ToxicityAssessment:
        """
        Live Claude classification with structured output. Raises if no client.
        Uses client.messages.parse() so the response is validated against the
        Pydantic schema automatically (model retries on schema mismatch).
        """
        if not self.is_available():
            raise RuntimeError(
                "No Claude client. Set ANTHROPIC_API_KEY or use classify_offline().")
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f"Assess this comment:\n\n<comment>\n{comment}\n</comment>"}],
            output_format=ToxicityAssessment,
        )
        return resp.parsed_output

    def classify_batch(self, comments, max_tokens: int = 1024, on_error="offline"):
        """
        Classify an iterable of comments. Per-comment errors are handled by
        on_error: 'offline' (fall back to the heuristic) or 'raise'.
        Returns list[ToxicityAssessment].
        """
        out = []
        for c in comments:
            try:
                out.append(self.classify(c, max_tokens=max_tokens))
            except Exception:
                if on_error == "raise":
                    raise
                out.append(self.classify_offline(c))
        return out

    # -- offline path ------------------------------------------------------
    _LEX = {
        "toxic": [r"idiot", r"stupid", r"moron", r"shut up", r"loser", r"pathetic",
                  r"dumb", r"garbage", r"trash", r"suck"],
        "severe_toxic": [r"kill yourself", r"kys", r"die in", r"worthless piece"],
        "obscene": [r"f\W*u\W*c\W*k", r"sh\W*it", r"bitch", r"asshole", r"bastard",
                    r"crap", r"damn"],
        "threat": [r"kill you", r"hurt you", r"hunt you", r"find you", r"beat you",
                   r"i will end", r"destroy you", r"come for you"],
        "insult": [r"idiot", r"moron", r"ugly", r"loser", r"pathetic", r"clown",
                   r"fool", r"incompetent"],
        "identity_hate": [r"\bgo back to\b", r"your kind", r"those people",
                          r"subhuman", r"vermin"],
    }
    _NEUTRAL_IDENTITY = re.compile(
        r"\b(i am|i'm|as a|proud)\b.{0,30}\b(muslim|jewish|gay|trans|black|woman|"
        r"asian|christian|disabled|immigrant)\b", re.I)

    def classify_offline(self, comment: str) -> ToxicityAssessment:
        """
        Deterministic lexical heuristic standing in for the live model when no API
        key is available. Intentionally simple and clearly labelled — its purpose
        is to keep NB3–5 runnable, NOT to be a credible classifier. The notebooks
        state this limitation explicitly wherever offline results are used.
        """
        t = str(comment).lower()
        flags = {lab: False for lab in LABELS}
        for lab, pats in self._LEX.items():
            if any(re.search(p, t) for p in pats):
                flags[lab] = True
        # severe implies toxic+obscene; threat/insult/identity imply toxic
        if flags["severe_toxic"]:
            flags["toxic"] = True
        if flags["threat"] or flags["insult"] or flags["identity_hate"] or flags["obscene"]:
            flags["toxic"] = True
        # fairness guard: a neutral self-identification is not identity_hate/toxic
        if self._NEUTRAL_IDENTITY.search(comment) and not any(
                re.search(p, t) for p in self._LEX["identity_hate"] + self._LEX["threat"]):
            flags["identity_hate"] = False
        clauses = sorted({ANTHROPIC_POLICY_MAP[l]["anthropic_usage_policy_clause"]
                          for l, v in flags.items() if v})
        return ToxicityAssessment(
            **flags, policy_clauses=clauses,
            rationale="[offline heuristic] lexical match; not a live model verdict.")


def assessments_to_frame(assessments) -> "object":
    """Convert list[ToxicityAssessment] → DataFrame of 0/1 label columns."""
    import pandas as pd
    return pd.DataFrame([a.label_vector() for a in assessments], columns=LABELS)
