# RQ4 result report — TEMPLATE (all values blank until produced by `run_rq4.py --analyze`)

**Run label:** `PILOT_ONLY` / `INCONCLUSIVE` / `BLOCKED` / complete — _____  **Config fingerprint:** _____  **Date(s):** _____
**Stage(s) completed:** _____  **Rows:** ___ / 1,280  **Non-ok rows (safety_excluded / control_degenerate / ineligible_pair / error):** ___

## 1. What was run
Model/lens, items, arms, layers, seeds as in `manifests/config_seed_manifest.json`. Deviations from the frozen protocol (dated, outcome-status stated): _____. Pre-outcome exclusions: _____ (manifest says none at Stage 0). Machine/memory notes: _____.

## 2. Validity checks (from `results/runtime_preflight.json`)
| Check | Result |
|---|---|
| Lens source layers = 0–22, `d_model` equal, block count | |
| Lens layer ↔ block-output alignment (formula, rel. error, neighbours) | |
| Final token `:` and Week 3 `seq_len` equal, all 80 prompts | |
| Tokenizer-eligible pairs (ineligible list) | |
| Hook fired exactly once per call; measured vs intended displacement; max ratio | |
| Safety-excluded (item, layer) cells | |

## 3. Generations and parsing
Share of stereotype / counter / unknown-hedged / unparseable by format × arm × layer (`tables/category_rates.csv`). Parser failures seen in raw text (examples, not tuned): _____.

## 4. Primary results (per layer; ten-item mean, 95 % base-item clustered bootstrap, n valid items)
| Layer | MC excess \|Δmargin\| | Open excess \|Δmargin\| | MC − open | Behavioural excess (MC) | (Open) | MC − open |
|---|---|---|---|---|---|---|
| L8 | | | | | | |
| L16 | | | | | | |
| L22 | | | | | | |

Sham (second control) and seed spread: _____. Individual base-item results: `tables/per_item.csv`; summarised: _____.

## 5. Secondary
Signed margin change, rank-gap change (Week 3 definition), evidence accuracy on disambiguated items, unknown/hedged rate, displacement summary: _____.

## 6. Interpretation (choose the supported statement; keep negatives)
- H1 (readout amplified more than leverage), H2 (leverage differs in the same direction), H3 (target ≤ controls), H4 (targeted effect without better stereotype-dependence or evidence following): _____ supported / not supported / inconclusive, because _____.
- Behavioural causality claimed only if generated-answer categories changed beyond controls: yes / no.
- Direction reversal vs neutralisation: _____.

## 7. Limitations
Ten base items, exploratory, previously seen in Week 3; intervals descriptive; single model; token-set contrast ≠ Week 3 rank statistic; bf16 rounding; parser coverage: _____.

## 8. Figures
`results/figures/fig1_excess_margin_change.png`, `fig2_excess_behavioural_change.png`, `fig3_category_composition.png` — rebuilt from `rq4_raw_results.jsonl` only.
