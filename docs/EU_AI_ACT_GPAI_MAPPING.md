# EU AI Act Mapping — Content-Safety Classifier & GPAI Articles 51–55

**Purpose.** Place the LLM safety evaluator inside the binding EU AI Act regime, on two
levels: (1) the **classifier as a high-risk-adjacent control** (Articles 5, 10, 15, 50),
and (2) the **General-Purpose AI (GPAI) model obligations** (Articles 51–55) that apply to
the foundation model (`claude-opus-4-8`) the challenger control is built on.

> Educational mapping for portfolio purposes; article interpretations are the author's and
> are not legal advice. Article numbering follows Regulation (EU) 2024/1689.

---

## Part A — Risk-management & data articles (the control itself)

| Article | Requirement (summary) | How this project addresses it |
|---|---|---|
| **Art. 5** — Prohibited practices | No harmful manipulation; protect vulnerable groups | Threat/identity-hate detection targets the very content Art. 5 is concerned with; severity-tiering treats violence/discrimination as Critical |
| **Art. 9** — Risk management system | Continuous, iterative risk management | The governance framework + KRIs + drift monitoring ([LLM_SAFETY_FRAMEWORK.md](LLM_SAFETY_FRAMEWORK.md)) is a risk-management system in miniature |
| **Art. 10** — Data governance | Training data examined for bias; representativeness | NB1 bias scan + per-group FPR KRI; documented data limitations in the [Model Card](MODEL_CARD.md) |
| **Art. 13** — Transparency to users | Clear, interpretable operation | Interpretable baseline (top-token weights), model card, policy mapping |
| **Art. 14** — Human oversight | Humans can oversee and intervene | System scoped as **decision-support**, not autonomous; redacted review surface mirrors human-in-the-loop T&S workflow |
| **Art. 15** — Accuracy, robustness, cybersecurity | Resilient to errors and adversarial manipulation | The entire **red-team protocol** ([RED_TEAMING_PROTOCOL.md](RED_TEAMING_PROTOCOL.md)) is an Art. 15 robustness exercise |
| **Art. 50** — Transparency obligations | Disclose AI interaction / synthetic content | Noted for deployment context (users informed an automated control is in use) |

---

## Part B — GPAI model obligations (Articles 51–55)

These articles govern the **foundation model** the challenger is built on. As a *deployer*
building on a GPAI model, this project documents the boundary of responsibility.

| Article | GPAI requirement (summary) | Relevance to this project |
|---|---|---|
| **Art. 51** — Classification as GPAI / systemic risk | Defines GPAI models and the systemic-risk threshold (e.g. training-compute criterion) | `claude-opus-4-8` is a GPAI model; the provider (Anthropic) carries the Art. 51 classification duties, not the deployer |
| **Art. 52** — Procedure for systemic-risk designation | Notification / designation process for systemic-risk GPAI | Provider obligation; deployer relies on provider's compliance posture |
| **Art. 53** — Obligations for GPAI providers | Technical documentation, training-data summary, copyright policy, downstream info | The deployer *consumes* provider documentation (model cards, usage policy) — this project's Usage-Policy mapping is the deployer-side use of that information |
| **Art. 54** — Authorised representative | Non-EU providers appoint an EU representative | Provider obligation |
| **Art. 55** — Obligations for systemic-risk GPAI | Model evaluation, adversarial testing (red-teaming), incident reporting, cybersecurity | **Mirrored at the deployer level here:** this project performs its own model evaluation (NB2/3), adversarial testing (NB4), and incident-style escalation (NB5 KRIs). The deployer's red-teaming *complements* the provider's Art. 55 obligations for the specific deployment context |

---

## Part C — Division of responsibility

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│ GPAI PROVIDER (Anthropic)   │        │ DEPLOYER (this safety control)    │
│ Art. 51–55:                 │        │ Art. 9,10,13–15:                  │
│ • model classification      │  uses  │ • deployment risk management      │
│ • provider-level red-team   │ ─────► │ • deployment-specific red-team    │
│ • technical documentation   │        │ • data governance for THIS use    │
│ • usage policy              │        │ • human oversight + monitoring    │
└─────────────────────────────┘        └──────────────────────────────────┘
```

The key governance insight: **building on a GPAI model does not transfer the deployer's
own obligations away.** Article 55-style adversarial testing still has to happen *for the
specific deployment* — which is exactly what Notebook 4 does. This project demonstrates the
deployer side of the shared-responsibility model.

---

## Interview talking point

> "The foundation model carries the GPAI obligations under Articles 51–55 — model
> classification, provider-level red-teaming, technical docs. But as a deployer I still own
> Articles 9, 10, 13–15 for my specific use: data governance, human oversight, and
> deployment-specific robustness testing. So I built my own Article-15 red-team battery on
> top of the provider's. The two don't substitute for each other — they compose."
