# Evaluation

This page reports how well the investigation agent works, how it was tested, and where it is weak.
All numbers below come from the scripts in `backend/evaluation/`; nothing here was estimated by hand.

## Summary

- The agent is **not more accurate than a simple rules-plus-classifier baseline** on the cases that the documented
  rules cover. On those it matches the baseline. Its measured advantage is limited to Tool Wear, and that advantage comes
  from a documented rule, not from the language model.
- What the language model adds is an **evidence-backed written explanation**, an explicit **"no fault" / "unexplained"
  outcome** instead of a forced guess, and a record of why each confidence level was set.
- The test run is **partial: 37 of 106 planned cases** (26 incidents and 11 healthy readings). It was stopped because
  of the free-tier token limit of the language-model provider.

## Data and what is real

| Item | Status |
|---|---|
| Sensor values and failure labels | Real (AI4I 2020 dataset) |
| Machine IDs, timestamps | Synthetic (assigned in Phase 1) |
| Maintenance records | Synthetic and **random** (templates picked at random); hidden from the model by default |
| Technical manuals | Synthetic (written for this project) |

Because timestamps are synthetic, the readings of one machine are independent samples, not a time series. The agent
therefore never reasons about trends, and decisions use the **latest reading only**.

## How the test avoids leaking the answer

1. **Hidden descriptions.** Every stored incident description contains its own failure type (339 of 339). The agent is
   never shown it; it receives a neutral sentence, the machine, the incident time (`as_of`) and the incident id, so
   its own row is excluded from its evidence search.
2. **Time travel.** With `as_of`, readings, maintenance records and the machine's earlier incidents after that moment
   are hidden, and the classifier predicts from the reading at that moment.
3. **Held-out incidents only.** The classifier was trained on a fixed 80/20 split (`random_state=42`). The split was
   reconstructed exactly and only incidents the classifier **never saw** are scored. This matters: Tool Wear recall is
   1.00 on training rows and 0.12 on unseen rows.
4. **Development set kept apart.** 12 cases were used to find bugs and tune the agent. They are excluded from the test
   results below.

## Baseline (no language model)

Scored on the 68 held-out incidents and on 1,932 healthy held-out readings.

| Method | Incidents named exactly | False alarms on healthy readings |
|---|---|---|
| Documented hard limits only | 58 / 68 | 3 / 1,932 (0.16%) |
| Classifier only | 59 / 68 | 28 / 1,932 (1.4%) |
| Hard limits + classifier | 60 / 68 (88%, 95% interval 78 to 94%) | 29 / 1,932 |
| Hard limits + "tool wear at 200+ min" hint | 50 / 68 (+18 partial) | 151 / 1,932 (7.8%) |

Heat dissipation, power and overstrain are almost lookups for the baseline (the manual's limits are the rules that
generate those labels). Tool Wear is the only hard class: about 95% of readings at 200+ minutes of wear are healthy.

## Agent results (test run, partial)

37 cases, none from the development set. The order was round-robin over failure types, so this sample is
**stratified, not random: do not read it as an overall accuracy.**

| Group | n | Agent exact | Baseline exact (hard limits + classifier) |
|---|---|---|---|
| Heat dissipation | 6 | 6 | 6 |
| Overstrain | 6 | 6 | 6 |
| Power | 6 | 6 | 6 |
| Power + Overstrain | 1 | 1 | 1 |
| Unknown | 1 | 1 | 1 |
| **Tool Wear** | 6 | **4** | 1 |
| Healthy readings | 6 | 6 | 6 |
| Healthy, tool wear 200+ min | 5 | 5 | 5 |

- Incidents named exactly: **24 of 26 (92%, 95% interval 76% to 98%)**. Baseline on the same cases: 21 of 26.
- Regressions against the baseline: **0**. Gains: 3, all Tool Wear.
- False alarms on healthy readings: **0 of 11** (the 95% upper bound is still 26%).
- One **confident error** (rated "high"): `inc-72`. Mean cost: 5,096 tokens and 37 seconds per case.
- Numbers in the report that could not be found in the evidence: **0**.
- The model's own confidence rating was overruled by a documented rule in **22 of 27** hypotheses it evaluated.

### The two misses (both true Tool Wear)

- `inc-72`: wear x torque was 11,710, above the lowest overstrain limit (11,000) but below the next (12,000). The rule
  tests against the lowest quality tier because per-reading product quality is not trustworthy in this data, so it
  flagged "Overstrain, high". The classifier said "No Failure".
- `inc-199`: wear x torque was 10,764, below every limit, but the classifier predicted Overstrain, which the agent rated
  "medium" next to Tool Wear "medium". A stricter design would not let a classifier claim outrank a documented limit
  that is not met.

## Why the design is what it is

The first pilot (12 cases) was run with the model rating its own confidence. It named only **4 of 10 incidents
exactly against 9 of 10 for the baseline**: it rated Tool Wear "high" on readings that broke other limits, and once
rated Heat Dissipation "low" although the documented limit was met. The fix was to let **documented rules set the
confidence** wherever a rule exists, to judge limits on the latest reading only, and to use the model only to write
the explanation. Its own ratings are kept as a second opinion.

## Known limitations

- Partial test run (37 of 106). The remaining cases are mostly the easy rule-lookup classes and healthy readings.
- Per-class samples are tiny: 6 Tool Wear cases, 1 Unknown, 1 compound. Compound failures (several causes at once) are
  barely tested because the development set used most of them.
- Tool Wear and Random failures cannot be detected reliably from sensors. The agent names Tool Wear at **medium at
  most**, and only when a failure was reported; in a routine check it issues a maintenance advisory instead.
- The overstrain rule uses the lowest quality limit and can over-flag (see `inc-72`).
- The synthetic maintenance records are not used as evidence. Whether they would help or hurt was **not measured**
  (the `--maintenance-evidence` ablation was not run).
- The agent runs on a free-tier model API (about 5,100 tokens per case); the evaluation was limited by its daily quota.

## Reproduce

```
python evaluation/08_make_eval_sets.py     # held-out split, leak check, case list
python evaluation/08_baseline.py           # no-LLM baseline
python evaluation/08_run_agent.py          # calls the model; resumable; stops at the daily limit
python evaluation/08_report.py             # agent vs baseline on the same cases
```
