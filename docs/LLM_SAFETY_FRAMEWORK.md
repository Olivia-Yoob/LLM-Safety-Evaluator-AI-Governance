# LLM Safety Governance Framework
### Applying KB Bank executive-accountability & KB Life KRI discipline to an LLM control

**Thesis.** A bank does not run a credit model and "hope it stays good." It assigns an
accountable executive, defines Key Risk Indicators (KRIs) with statistical thresholds,
and escalates on a RAG (Red/Amber/Green) basis when a metric drifts. An LLM safety
classifier is *also* a high-impact control — so it deserves the same discipline. This
framework ports the KB Bank / KB Life governance pattern onto the LLM safety evaluator.

---

## 1. Three lines of defence (adapted)

| Line | Banking role | LLM-safety analogue |
|---|---|---|
| 1st — own & operate | Business unit running the model | T&S/ML team operating the classifier; sets thresholds, triages flags |
| 2nd — oversee & challenge | Independent risk function | Responsible-AI / model-risk function reviewing this framework, red-team results, KRIs |
| 3rd — assure | Internal audit | Periodic audit that the control runs, KRIs fire, and escalations are actioned |

The notebooks, model card, and red-team protocol are the **evidence pack** the 2nd and
3rd lines would review.

---

## 2. Executive accountability

Borrowed from the KB Bank executive-accountability model: every material control has a
**named accountable owner** and a defined escalation path. For this control:

| Role | Responsibility |
|---|---|
| Control Owner (1st line) | Operates the classifier; owns threshold settings and flag triage |
| Model Risk Lead (2nd line) | Approves severity tiers, thresholds, and drift tolerances; signs off red-team scope |
| Accountable Executive | Ultimately answerable for safety-control failures; receives RED escalations |

> In a portfolio context these are illustrative roles, but the *principle* — no control
> without a named owner and an escalation path — is the transferable governance idea.

---

## 3. Key Risk Indicators (KRIs)

Each KRI has a baseline, a severity-tiered tolerance, and a RAG rule. Implemented in
`src.safety_metrics.drift_kri`.

| KRI | Definition | Baseline source | Tolerance (by tier) | RAG rule |
|---|---|---|---|---|
| **K1 — Detection recall** | Recall per category on a fixed eval set | NB2/NB3 scorecard | Critical 0.03 / High 0.05 / Medium 0.08 | drop > tol → RED; > tol/2 → AMBER |
| **K2 — Identity FPR gap** | Max per-group false-positive rate − overall FPR | NB5 fairness scan | 3 percentage points | gap > tol → RED |
| **K3 — Red-team pass rate** | % of adversarial suite passing | NB4 red-team run | 90% overall; 100% on benign-identity | below → RED |
| **K4 — Abstention / hallucination** | Rate of malformed or low-confidence verdicts | NB3 LLM run | see [HALLUCINATION_MONITORING.md](HALLUCINATION_MONITORING.md) | spike → AMBER |

The **severity tier sets the tolerance**: a 4-point recall drop on `threat` (Critical,
tol 0.03) trips RED, while the same drop on `obscene` (Medium, tol 0.08) is GREEN. This
is the KB-Life-KRI idea — *the threshold scales with the consequence of the risk.*

---

## 4. Escalation (RAG)

```
GREEN  — within tolerance               → log, no action
AMBER  — degradation > half-tolerance   → Control Owner investigates; watch next cycle
RED    — degradation > tolerance OR any  → escalate to Accountable Executive;
         benign-identity red-team fail     freeze/rollback the affected category head
```

A RED on a **Critical** category (`threat`, `identity_hate`) is treated as a potential
safety/discrimination incident, not a routine metric miss — mirroring how a bank treats a
breach on a safety-critical KRI.

---

## 5. Control lifecycle

| Stage | Gate | Evidence |
|---|---|---|
| Data understanding | Integrity checks pass; harm surface characterised | NB1 |
| Model build | Baseline + challenger trained; per-category metrics reviewed | NB2, NB3 |
| Pre-deployment validation | Red-team battery passes; fairness within tolerance | NB4, NB5 |
| Sign-off | Model card + this framework reviewed by 2nd line | docs/ |
| Monitoring | KRIs computed each cycle; drift KRI run vs baseline | NB5, `drift_kri` |
| Re-validation | On RED escalation or material data/policy change | trigger-based |

---

## 6. Why this is the differentiator

Most ML portfolios stop at "here is my F1 score." This framework demonstrates that the
candidate can take a **real financial-services governance pattern** (executive
accountability, three lines of defence, statistically-thresholded KRIs, RAG escalation)
and **apply it to a novel risk surface (LLM safety)** — which is precisely the bridge a
Quantitative AI Risk Analyst role requires.
