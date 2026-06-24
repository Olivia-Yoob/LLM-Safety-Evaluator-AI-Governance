# Hallucination & Reliability Monitoring (LLM-specific risk)

**Why this exists.** A classical-ML classifier returns a number; it cannot "make things
up." An **LLM** classifier can: it can return a malformed verdict, invent a policy clause
that doesn't exist, rationalise inconsistently, or flip its answer on a trivially-reworded
input. These are *LLM-specific* failure modes that have no analogue in Projects 1–2, and a
governed LLM control has to monitor for them explicitly. This is part of what makes Project
3 a genuine bridge into LLM governance rather than "classical ML with an API call."

---

## 1. LLM-specific failure modes monitored

| Failure mode | What it looks like | Mitigation in this project |
|---|---|---|
| **Schema violation** | Model returns text that doesn't fit the verdict schema | **Structured output** (`ToxicityAssessment` Pydantic schema via `messages.parse`) forces schema-valid output; violations are caught at parse time |
| **Clause fabrication** | Model cites an Anthropic Usage Policy clause that isn't in the policy | `policy_clauses` validated against the known clause set from `ANTHROPIC_POLICY_MAP`; unknown clauses flagged |
| **Inconsistency / instability** | Same comment, reworded trivially, flips the verdict | Red-team `obfuscation` + paraphrase probes measure prediction stability |
| **Rationale–label mismatch** | Booleans say "clean" but rationale describes harm (or vice-versa) | Cross-check: does the rationale's sentiment agree with the flags? |
| **Over-confidence on abstention cases** | Confidently flags ambiguous/benign content | Red-team `benign_identity` + `clean_hard_negatives` suites; FPR KRI |

---

## 2. Reliability KRIs (feed K4 in the governance framework)

| KRI | Definition | AMBER/RED rule |
|---|---|---|
| **Schema-violation rate** | % of API calls whose output fails schema validation (before retry) | > 1% → AMBER; > 5% → RED |
| **Clause-fabrication rate** | % of verdicts citing an unrecognised policy clause | any sustained > 0 → AMBER |
| **Paraphrase-instability rate** | % of cases where a benign paraphrase flips the flag | > 5% → AMBER |
| **Rationale-consistency rate** | % of verdicts where rationale sentiment agrees with the flags | < 95% → AMBER |

These are reliability analogues of the detection KRIs — they monitor whether the *control
itself* is behaving, independent of whether its classifications are correct.

---

## 3. Design choices that pre-empt hallucination

1. **Structured output, not free text.** The single most effective mitigation: the model
   cannot return a malformed verdict because the SDK validates against the Pydantic schema
   and re-prompts on mismatch.
2. **Closed clause vocabulary.** The system prompt is built from the canonical policy map,
   so the model is steered toward the *finite, real* set of policy clauses rather than
   inventing them.
3. **Non-graphic rationale instruction.** The model is instructed to justify without
   reproducing harmful content — reducing the chance the "explanation" itself becomes a
   harm vector.
4. **Deterministic offline fallback.** When the API is unavailable, the system falls back
   to a heuristic that is *explicitly labelled as not a model verdict* — so a degraded mode
   can never be silently mistaken for a confident model output.

---

## 4. What this is NOT

- Not a claim that the LLM never hallucinates — it can. The claim is that the control is
  **instrumented to detect** these failures and escalate them, rather than trusting the
  model blindly.
- Not a substitute for human review on contested cases; reliability monitoring tells you
  *when* to pull a human in.

---

## Interview talking point

> "Moving from a logistic-regression control to an LLM control introduced failure modes I'd
> never had to govern before — schema violations, fabricated policy clauses, answer
> instability under paraphrase. So I added reliability KRIs that monitor the *control's own
> behaviour*, used structured output to make malformed verdicts impossible, and constrained
> the model to a closed clause vocabulary. That's the part of LLM governance that classical
> model risk management doesn't prepare you for."
