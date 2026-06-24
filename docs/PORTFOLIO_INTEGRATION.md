# Portfolio Integration — How the Three Projects Connect

**Target role.** Quantitative AI Risk Analyst — Trust & Safety / Responsible AI. The
portfolio is deliberately a *progression*, not three disconnected demos: the same
governance discipline applied to a widening risk surface, ending at the LLM frontier.

---

## The arc

| # | Project | Model(s) | Risk surface | Governance contribution |
|---|---------|----------|--------------|--------------------------|
| 1 | **Fraud Detection + AI Risk Governance** | Logistic Regression, Random Forest, LightGBM | Financial fraud (classical ML) | Risk-governance scaffolding: thresholds, model-risk documentation, monitoring |
| 2 | **Home Credit Default + Fair Lending** | LightGBM + fairness analysis | Credit risk + protected-group fairness | Disparate-impact / fairness measurement under regulatory (fair-lending) constraint |
| **3** | **LLM Safety Evaluator + Usage Policy** | TF-IDF+LogReg **and** Claude (`claude-opus-4-8`) | **GenAI / LLM safety** | The *same* discipline (thresholds, fairness, monitoring) **plus** LLM-specific governance (policy mapping, red-teaming, hallucination KRIs) |

---

## The threads that carry across all three

### 1. Thresholds as a governance lever
- **P1:** fraud score cut-offs traded off precision/recall against investigation cost.
- **P3:** severity-tiered decision thresholds (Critical 0.30 / High 0.40 / Medium 0.50)
  trade off recall against over-flagging — the *same idea*, now driven by an Anthropic
  severity tier instead of a fraud-loss curve.

### 2. Fairness / disparate-impact monitoring
- **P2:** measured outcome disparities across protected groups for fair-lending.
- **P3:** `group_false_positive_rates` measures per-identity-group false-positive rate —
  literally the same disparate-impact instinct, applied to *who gets wrongly flagged as
  toxic* instead of *who gets wrongly denied credit*.

### 3. Statistical-threshold KRIs & escalation
- **P1:** model-monitoring KRIs with alerting.
- **P3:** `drift_kri` with severity-tiered tolerances and RAG escalation, tied to the KB
  Bank executive-accountability model ([LLM_SAFETY_FRAMEWORK.md](LLM_SAFETY_FRAMEWORK.md)).

### 4. Champion / challenger discipline
- **P1/P2:** benchmarked successive models against a baseline.
- **P3:** the TF-IDF+LogReg baseline is the *champion floor* the Claude *challenger* must
  beat to justify its cost, latency, and opacity.

---

## What is genuinely NEW in Project 3 (the bridge)

These have **no analogue** in Projects 1–2 — they are why P3 demonstrates *LLM* governance,
not just more classical ML:

| New capability | Why it only appears with LLMs |
|---|---|
| **Usage-Policy mapping** | Classical models output scores; LLM safety is governed against *policy clauses* |
| **Prompt-as-control** | The system prompt *is* part of the control surface and must be version-controlled |
| **Structured-output reliability** | Free-text models can produce malformed/fabricated verdicts; needs schema enforcement |
| **Hallucination / instability KRIs** | LLM-specific failure modes ([HALLUCINATION_MONITORING.md](HALLUCINATION_MONITORING.md)) |
| **Adversarial red-teaming as standard** | LLMs are attacked via natural language; robustness testing is first-class |
| **GPAI shared-responsibility model** | EU AI Act Art. 51–55 split between foundation-model provider and deployer |

---

## The one-sentence portfolio narrative

> "Across three projects I take a single governance discipline — thresholds, fairness
> monitoring, statistically-thresholded KRIs, champion/challenger validation — and carry it
> from classical fraud and credit models all the way to an LLM safety control, adding the
> LLM-specific layers (Usage-Policy mapping, red-teaming, hallucination monitoring) that the
> frontier risk surface demands. I can govern both the models a bank runs today and the
> models a frontier-AI lab ships tomorrow."

---

## Suggested reading order for a reviewer

1. This file — the arc.
2. [README](../README.md) — Project 3 at a glance.
3. [ANTHROPIC_USAGE_POLICY_MAPPING.md](ANTHROPIC_USAGE_POLICY_MAPPING.md) — the centrepiece.
4. Notebooks 1 → 5.
5. [LLM_SAFETY_FRAMEWORK.md](LLM_SAFETY_FRAMEWORK.md) + [RED_TEAMING_PROTOCOL.md](RED_TEAMING_PROTOCOL.md) — the governance depth.
