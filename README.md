# LLM Safety Evaluator & AI Governance
### Mapping Jigsaw toxicity → Anthropic Usage Policy, with a systematic red-teaming framework

**Author:** Olivia Kim 

---

## TL;DR for a reviewer

This project builds and **governs** a content-safety classifier the way a frontier-AI
Trust & Safety team would. It takes 159,571 real Wikipedia comments (the Jigsaw Toxic
Comment corpus), and:

1. **Characterises the harm surface** across 6 toxicity categories (NB1).
2. Establishes a **classical-ML baseline** — TF-IDF + Logistic Regression (NB2).
3. Adds a **Claude API classifier** (`claude-opus-4-8`) that judges content against
   **Anthropic Usage Policy clauses**, not abstract labels (NB3).
4. ⭐⭐ Runs a **systematic red-teaming framework** — obfuscation evasion, implicit
   coded hate, veiled threats, and identity false-positive bias (NB4).
5. Wraps the whole thing in a **governance evaluation** — severity-tiered decision
   thresholds, per-group fairness KRIs, and statistical drift alerting (NB5).

The three differentiators are the **Anthropic Usage Policy mapping**, the **red-team
framework** (EU AI Act Art. 15), and the application of a real-world **bank executive-
accountability / KRI framework** (KB Bank / KB Life) to an LLM safety control.

---

## ⭐ The centrepiece: Jigsaw 6 → Anthropic Usage Policy mapping

A toxicity label like `identity_hate` means nothing to an executive or a regulator on
its own. What they need is the line from **observed harm → the policy clause it violates
→ the accountable control**. That line is this project's core artefact
([docs/ANTHROPIC_USAGE_POLICY_MAPPING.md](docs/ANTHROPIC_USAGE_POLICY_MAPPING.md),
also emitted as [outputs/jigsaw_to_anthropic_usage_policy.csv](outputs/jigsaw_to_anthropic_usage_policy.csv)):

| Jigsaw category | Anthropic Usage Policy clause (paraphrased) | Harm family | EU AI Act | Severity |
|---|---|---|---|---|
| `toxic` | Do Not Generate Psychologically/Emotionally Harmful Content | Harassment & Abuse | Art. 5 · 15 | Medium |
| `severe_toxic` | …severe abuse | Harassment & Abuse | Art. 5 · 15 | High |
| `obscene` | Do Not Generate Sexually Explicit / Obscene Content | Adult / Explicit | Art. 50 | Medium |
| `threat` | Do Not Incite, Threaten or Facilitate Violence | Violence & Physical Harm | Art. 5 · Annex III | **Critical** |
| `insult` | Do Not Generate Psychologically/Emotionally Harmful Content (bullying) | Harassment & Abuse | Art. 5 · 15 | Medium |
| `identity_hate` | Do Not Generate Hateful Content / Promote Discrimination | Hate & Discrimination | Art. 5 · 10 · 15 | **Critical** |

> *Anthropic Usage Policy wording is paraphrased from the public policy at
> `anthropic.com/legal/aup` — an analyst's mapping, not a verbatim quote.*

The `severity_tier` is not decorative: it **drives governance downstream** — Critical
harms get lower decision thresholds (higher recall), harder red-team effort, and tighter
drift tolerances. One table, propagated through five notebooks.

---

## Repository structure

```
llm-safety-evaluator-governance/
├── README.md                       ⭐ this file — policy-mapping narrative
├── requirements.txt
├── notebooks/
│   ├── 01_data_understanding.ipynb       6-category distribution, co-occurrence, bias scan
│   ├── 02_baseline_ml.ipynb              TF-IDF + Logistic Regression (the control floor)
│   ├── 03_llm_classification.ipynb       ⭐ Claude API, policy-native verdicts, LLM vs ML
│   ├── 04_red_teaming.ipynb              ⭐⭐ systematic adversarial testing
│   └── 05_evaluation_governance.ipynb    safety metrics + KRI governance framework
├── src/
│   ├── data_loader.py              load + redact + canonical policy map
│   ├── ml_baseline.py              TF-IDF + LogReg pipeline
│   ├── llm_client.py               ⭐ Claude API wrapper (graceful w/o key)
│   ├── safety_metrics.py           per-category metrics, tiered thresholds, drift KRI
│   └── red_team_generator.py       ⭐ adversarial test-case generation
├── docs/                           ⭐⭐ the governance value
│   ├── MODEL_CARD.md
│   ├── ANTHROPIC_USAGE_POLICY_MAPPING.md
│   ├── LLM_SAFETY_FRAMEWORK.md      KB Bank accountability applied to LLM
│   ├── RED_TEAMING_PROTOCOL.md
│   ├── EU_AI_ACT_GPAI_MAPPING.md    Articles 51–55
│   ├── HALLUCINATION_MONITORING.md
│   └── PORTFOLIO_INTEGRATION.md     how Projects 1–3 connect
├── prompts/
│   ├── red_team_prompts.json       version-controlled attack suites
│   └── evaluation_prompts.json     classifier system prompt + calibration set
├── results/                        committed figures
├── outputs/                        committed CSV artefacts (policy map, scorecards)
├── data/                           (gitignored)
└── models/                         (gitignored)
```

Logic lives in `src/` and the notebooks import it (`from src import ...`), so each
notebook reads as a governance narrative rather than a wall of utility code — the
"engineer who governs" signal, not "notebook tinkerer".

---

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data — place the Kaggle Jigsaw zips here (already present in this repo):
#    jigsaw-toxic-comment-classification-challenge/train.csv.zip
#    (download: kaggle competitions download -c jigsaw-toxic-comment-classification-challenge)

# 3. (Optional) Claude API for Notebook 3 — without a key, NB3–5 still run via a
#    clearly-labelled offline heuristic so the pipeline is fully reproducible.
export ANTHROPIC_API_KEY=sk-ant-...      # or put it in a .env file

# 4. Run the notebooks in order
jupyter lab    # open notebooks/01 ... 05
```

> **No API key?** Every notebook is built to degrade gracefully:
> `ToxicityClassifier.is_available()` returns `False`, and NB3–5 fall back to a
> deterministic lexical heuristic that is **explicitly labelled as not a live-model
> verdict**. This keeps the governance pipeline runnable and auditable end-to-end.


---

## Portfolio arc

| # | Project | Risk domain | Shared thread |
|---|---------|-------------|---------------|
| 1 | Fraud Detection + AI Risk Governance | Classical ML | Governance frameworks, thresholds |
| 2 | Home Credit Default + Fair Lending | Classical ML + fairness | Disparate-impact / FPR monitoring |
| **3** | **LLM Safety Evaluator + Usage Policy** | **GenAI / LLM** | **Same governance discipline, new risk surface** |

See [docs/PORTFOLIO_INTEGRATION.md](docs/PORTFOLIO_INTEGRATION.md) for the full connective tissue.

---

## Honest scope & limitations

- The identity-bias scan uses **term-matching** — a screening proxy to prioritise
  red-teaming, **not** a bias verdict.
- The offline fallback classifier is a **lexical heuristic**, present only so the
  pipeline runs without an API key; live results require `claude-opus-4-8`.
- Red-team suites are **illustrative and version-controlled**, not exhaustive; the
  protocol ([docs/RED_TEAMING_PROTOCOL.md](docs/RED_TEAMING_PROTOCOL.md)) is the
  reusable contribution, and is designed to be extended.
- Jigsaw labels are crowd-sourced and carry their own annotation bias; this is treated
  as a known data-quality limitation, not ground truth about the world.

---