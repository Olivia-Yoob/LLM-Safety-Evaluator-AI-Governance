# Model Card — LLM Safety Evaluator

A model card in the style of Mitchell et al. (2019), adapted for a **two-control safety
system** (classical-ML baseline + Claude API challenger) governed as one.

---

## Model details

| Field | Value |
|---|---|
| Project | LLM Safety Evaluator & Governance (Portfolio Project 3) |
| Owner | Olivia Kim |
| Date | 2026 |
| Version | 1.0 |
| Control A (baseline) | TF-IDF (1–2 gram, 30k features) + One-vs-Rest Logistic Regression, `class_weight="balanced"` |
| Control B (challenger) | Claude API (`claude-opus-4-8`), policy-native structured-output classifier |
| Output | 6 binary labels: `toxic, severe_toxic, obscene, threat, insult, identity_hate` |
| Mapped to | Anthropic Usage Policy clauses + EU AI Act articles (see [ANTHROPIC_USAGE_POLICY_MAPPING.md](ANTHROPIC_USAGE_POLICY_MAPPING.md)) |

---

## Intended use

- **Primary:** demonstrate an end-to-end, *governed* content-safety classification
  pipeline for portfolio / educational purposes.
- **In-scope:** English-language short-form user comments (Wikipedia-talk-page style).
- **Out of scope:** production moderation decisions without human review; non-English
  text; long-form documents; any high-stakes individual determination. This system is a
  **decision-support control**, not an autonomous arbiter.

---

## Training & evaluation data

- **Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle), 159,571 labelled
  Wikipedia comments.
- **Class balance:** ~90% clean; high-severity categories (`threat`, `identity_hate`,
  `severe_toxic`) each < 1% — a severe imbalance that is itself a governed risk.
- **Split:** 80/20 train/test, fixed seed 42.
- **Known data limitations:** crowd-sourced labels (annotation bias); identity terms
  correlate with toxicity in the data, creating identity-bias risk; single domain.

---

## Performance (illustrative, baseline control)

Per-category metrics are produced by `src.safety_metrics.per_category_metrics` in
Notebook 2. Headline pattern (held-out test): high ROC-AUC across categories, but
**recall on Critical-severity rare classes is the binding constraint** — exactly the
categories the data is thinnest on. Accuracy is deliberately *not* reported as a headline
metric (a predict-all-clean model scores ~90%).

The Claude challenger (Notebook 3) is benchmarked against this baseline on a sampled
subset; full-corpus LLM evaluation is out of scope on cost grounds and is noted as such.

---

## Fairness & bias

- **Measured:** per-identity-group false-positive rate
  (`src.safety_metrics.group_false_positive_rates`) — the LLM analogue of fair-lending
  disparate-impact monitoring (Portfolio Project 2).
- **Known risk:** identity-term mentions carry an above-baseline toxicity rate in the
  data, which can teach a model the spurious association *identity-term ⇒ toxic*. This is
  red-teamed directly (NB4 `benign_identity_false_positive` suite) and monitored as a KRI
  (NB5).
- **Caveat:** the bias scan is a **term-matching screening proxy**, not a bias verdict.

---

## Governance & monitoring

- **Severity-tiered thresholds** (Critical 0.30 / High 0.40 / Medium 0.50).
- **Red-team battery** of named failure modes ([RED_TEAMING_PROTOCOL.md](RED_TEAMING_PROTOCOL.md)).
- **Drift KRIs** with RAG escalation tied to executive accountability
  ([LLM_SAFETY_FRAMEWORK.md](LLM_SAFETY_FRAMEWORK.md)).
- **Regulatory mapping:** EU AI Act Articles 5, 10, 15, 50–55
  ([EU_AI_ACT_GPAI_MAPPING.md](EU_AI_ACT_GPAI_MAPPING.md)).

---

## Ethical considerations & limitations

- Harmful sample text is **redacted** before display (`src.data_loader.redact`); the
  system is built to *not reproduce* the content it detects.
- The offline fallback classifier is a lexical heuristic for reproducibility only and is
  explicitly **not** a credible model.
- Crowd-sourced labels mean the system optimises toward *annotators' judgements*, which
  may not reflect any single community's norms.
- A safety classifier is a sociotechnical control: thresholds and tiers encode value
  judgements that should be owned by a policy team, not buried in code.
