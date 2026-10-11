# Jev (BeatAPI) Evaluation — Methodology and Results

Ask: evaluate Jev (TypeSafe's "choice" decision model, accessed via BeatAPI)
on DiSC-AMC's modulation classification task, compare it fairly against the
paper's reported numbers for equivalent configurations, and check whether
the prompt and API payload — both originally written for free-text
generative LLMs (Gemini/GPT/DeepSeek) — were actually suited to Jev's native
structured-choice interface. Code: `src/evaluation/jev_eval.py`, runners:
`run_jev_eval*.sh`.

## TL;DR

- Jev (`jev-1.13`, paid tier) is a *choice* classifier, not a free-text
  generator: you send a `state` (context) plus a labeled option set via a
  `questions`/`criteria` schema, and it returns the highest-probability
  option with a probability per option — no chain-of-thought to parse.
- **First comparisons were not apples-to-apples.** The paper's 71.5%
  DeepSeek-R1-7B number and Jev's first `faiss_filled` run used different
  candidate-shortlisting sources: `faiss_filled` always force-pads to 5
  candidates, while plain `faiss` lets the set collapse to 1 candidate on
  ~31% of queries (which are then "free" at ~98% accuracy). Rebuilding the
  Jev run on the *same* `faiss` config the 71.5% number actually used
  changed the comparison from 61.5% → 65.0%.
- **The prompt text was never adapted for Jev.** It's the exact same
  generative-LLM prompt used for Gemini/GPT/DeepSeek, including `<think>`
  CoT instructions and "no extra words" output-format rules that are inert
  for Jev (it never free-text-generates or gets parsed for those tags).
- **Found and fixed a real bug**, not just a tailoring choice: BeatAPI's
  `criteria` field is documented as "an object mapping each option name to
  what it means" (a description string). Every Jev run before this fix sent
  `{candidate: None for candidate in candidates}` — bare class names with
  *zero* distinguishing information, never using the schema as designed.
- **Net effect of both fixes (stripped CoT + real criteria descriptions)**:
  a real but modest +3.0pp gain on the simplest (10-class) config, roughly
  flat-to-slightly-negative everywhere else. Jev still does not approach the
  paper's DeepSeek-R1-7B numbers on either SynthMod-10 (61.5–65.0% vs.
  71.5%) or RadioML (32.5–33.3% vs. 76.25%) — that gap looks like a model
  capability ceiling, not a prompt problem.

## Methodology, in order of how it actually unfolded

1. **Baseline run** (`run_jev_eval.sh`, Table VI config: centroid candidate
   retrieval, discretized stats, RAG=false) — the paper's existing
   apples-to-apples target for Jev, since it's the same config used to
   report Gemini-2.5-Flash (45.5%) / Gemini-2.5-Pro (51.0%) in Table VI.
   Result: **53.5%** (107/200) — already ahead of both Gemini variants.
2. **"Best" config run** (`run_jev_eval_best.sh`: FAISS-filled candidate
   retrieval + PCA-compressed DINOv2 embeddings, matching the paper's
   strongest training-free ablation direction). Result: **61.5%**
   (123/200), vs. the paper's cited best training-free cell of 71.5%
   (DeepSeek-R1-7B). This was *not* actually apples-to-apples — see next.
3. **Root-caused the 61.5% vs 71.5% gap**: inspected the actual experiment
   that produced 71.5%
   (`exp/comparisons/unlabeled_10k_faiss_emb10_disc_unsloth_DeepSeek-R1-Distill-Qwen-7B_v01/config.json`)
   and found it used `prediction_source="faiss"` (plain kNN voting), not
   `"faiss_filled"` (brute-force padded to a fixed candidate count). Plain
   `faiss` lets the candidate set collapse to as few as 1 class when kNN
   voting is unanimous; breaking down that run's own accuracy by candidate
   count showed 1-candidate queries score 98.4% (61/62, essentially free)
   while 5-candidate queries score only 14.3% (1/7, below random). 31% of
   that run's queries were "free" in a way Jev's always-5-candidate
   `faiss_filled` run never got the benefit of.
4. **True apples-to-apples run** (`run_jev_eval_faiss_apples.sh`: same
   `prediction_source="faiss"` as the 71.5% run). Verified the candidate-
   count distribution matched the reference run exactly
   (`{1:62, 2:70, 3:28, 4:33, 5:7}`) before spending API budget. Result:
   **65.0%** (130/200) — the correct, fair comparison point against 71.5%.
5. **Best-RadioML apples-to-apples run** (`run_jev_eval_radioml_best.sh`):
   located the paper's best RadioML cell (Table III, DINOv2 fine-tuned
   encoder, +20dB = 76.25%) and its exact underlying pkl/config
   (`prediction_source="faiss_filled_rml"`, `snr_20db`,
   `prompt_version="v2"`) by cross-referencing
   `exp/radioml_native_eval/radioml_native_eval_results.csv`. Result:
   **33.3%** (80/240) — again `faiss_filled_rml` is always-5-candidate, so
   this comparison was already apples-to-apples on candidate count.
6. **Prompt tailoring**: noticed the prompt text sent to Jev is identical to
   the one built for Gemini/GPT/DeepSeek — including `<think>` CoT mandates
   and format rules ("the output MUST be a single entry... no extra words").
   Per the module's own design, Jev never free-text-generates or gets
   parsed for those tags; the `choice`/`criteria` schema extracts the answer
   structurally. Added `tailor_prompt_for_jev()` to strip the
   `**RESPONSE RULES:**` block and the trailing "...following all rules
   above" sentence, verified on all three prompt styles (centroid,
   faiss_filled, RadioML pv2) before running.
7. **Found the criteria-description bug**: fetched BeatAPI's actual
   `/decisions` schema docs and found `criteria` is "an object mapping each
   option name to what it means" (string descriptions in every documented
   example), not `null`. `get_jev_response()` had been sending
   `{c: None for c in candidates}` since the module was first written —
   every prior Jev run gave the choice schema zero information beyond the
   bare class-name strings. Added `CLASS_DESCRIPTIONS` (short,
   technically-grounded envelope/phase/amplitude blurbs for all 10
   SynthMod-10 + 24 RadioML classes) and wired it through as
   `use_criteria_descriptions=True`, decoupled from `tailor_prompt` so
   either could be toggled independently.
8. **Reran all 4 configs with both fixes** (`run_jev_eval_tailored.sh`,
   `run_jev_eval_best_tailored.sh`, `run_jev_eval_faiss_apples_tailored.sh`,
   `run_jev_eval_radioml_best_tailored.sh`) — tagged `jev-1.13-tailored` in
   the output filename so they never clobber the default-prompt results;
   the actual BeatAPI `model` field still sends plain `jev-1.13`.

## Final comparison — default vs. tailored prompt, vs. paper

Full machine-readable version: `exp/comparisons/jev_prompt_tailoring_comparison.csv`
(not tracked in git — `exp/` is gitignored for this repo; this table is the
checked-in copy).

| Config | Paper location | Paper accuracy | Jev default | Jev tailored (CoT-stripped + criteria) | Δ |
|---|---|---|---|---|---|
| SynthMod-10, Table VI baseline (centroid, stats) | Table VI | 45.5% (Gemini-2.5-Flash) / 51.0% (Gemini-2.5-Pro) | 53.5% (107/200) | **56.5%** (113/200) | +3.0pp |
| SynthMod-10, best config (faiss_filled + embeddings) | Sec. VI-A / Fig. 4–5 (star annotation) | 71.5% (best training-free cell, DeepSeek-R1-7B) | 61.5% (123/200) | 60.0% (120/200) | −1.5pp |
| SynthMod-10, apples-to-apples w/ 71.5% run (plain `faiss`) | same `config.json` as the 71.5% run, cited in Sec. VI-A / Fig. 4–5 | 71.5% (143/200, DeepSeek-R1-7B) | 65.0% (130/200) | 65.0% (130/200) | 0.0pp |
| RadioML `snr_20db`, best config (`faiss_filled_rml` + embeddings) | Table III, DiSC-AMC column, +20dB row | 76.25% (183/240, DeepSeek-R1-7B) | 33.3% (80/240) | 32.5% (78/240) | −0.8pp |

## Why tailoring didn't help uniformly

Per-class accuracy on the RadioML run shows the worst scores cluster
entirely in same-family, different-order modulation groups: QAM orders
(128QAM 0.3, 256QAM 0.2–0.3, 32QAM 0.2, 16QAM 0.4, 64QAM 0.4) and APSK
orders (128APSK 0.0, 32APSK 0.1–0.2, 64APSK 0.2, 16APSK 0.2–0.3) — 13 of
RadioML's 24 classes are same-family-different-order. A per-class
`criteria` description for `16PSK` vs `32PSK` can only restate the symbol
count already present in the class name; true order discrimination needs
fine-grained detail from that specific query's discretized feature values
(already present in `state`), which a static per-class blurb cannot supply.
SynthMod-10 has no such repeats (all 10 classes are genuinely distinct
modulation families), which is consistent with it being the one config
where tailoring produced a clean, unambiguous gain.

## Caveats / things not yet resolved

- The `faiss_filled` vs. plain `faiss` candidate-count gap (free 1-candidate
  queries vs. always genuine 5-way choices) is itself a methodological
  wrinkle in the *paper's own* training-free ablations, not something
  introduced by this Jev evaluation — anyone citing the 71.5%/76.25% numbers
  should be aware a meaningful fraction of those queries were not real
  5-way classification decisions.
- All Jev runs here use `num_tries=1` (no self-consistency voting), matching
  the paper's own evaluation protocol for its training-free LLM rows.
- Jev's gap to DeepSeek-R1-7B is large and essentially prompt-invariant
  (default vs. tailored moves it by at most a few points either way),
  suggesting it reflects the underlying model's capability on this specific
  discretized-symbolic-feature task rather than a prompt-engineering
  problem — this evaluation does not have a path to close that gap further
  without changing which Jev model/tier is used.
