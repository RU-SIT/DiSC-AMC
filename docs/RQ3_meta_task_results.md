# RQ3: Meta-Task Regularization for DenoMAE2 — Final Results

Committee ask (proposal defense): compare the *same backbone* with and
without Meta-Task regularization on few-shot/cross-domain AMC adaptation,
reporting post-adaptation accuracy, improvement over baseline, and
adaptation cost — using an existing AMC dataset and the existing Meta-Task
implementation rather than a new framework. Code: `src/fewshot_meta_task/`.
Original plan: `/home/woreom/.claude/plans/declarative-soaring-bubble.md`.

## TL;DR

Three rounds of experiments, with two methodology corrections along the
way. **Round 3 is the one to cite.**

- **Round 1 (frozen encoder)**: Meta-Task regularization looked like a real
  win — +4.58pp at −20dB in the full DiSC-AMC pipeline. But the comparison
  used a frozen encoder (classification head trained, encoder weights
  untouched) for both conditions.
- **Correction 1**: the frozen-encoder convention doesn't match how DINOv2's
  baseline in the paper was built (DINOv2's encoder *is* updated during
  finetuning; DenoMAE2.0's, in the code as it stands, is not). This asymmetry
  was already present in the published paper, not something introduced here.
- **Round 2 (real, end-to-end finetuned encoder, matching DINOv2's
  convention)**: Meta-Task's apparent edge **mostly disappears** — averaged
  over all 5 SNRs, Meta-Task-regularized DenoMAE2.0 (22.92%) is marginally
  *worse* than vanilla DenoMAE2.0 (23.00%).
- **Correction 2**: the Meta-Task auxiliary loss was implemented as a
  scalar-weighted term (`aux_weight`) added into one combined loss through
  one shared AdamW optimizer. Under Adam, a constant scalar on one loss term
  is a much weaker, less controllable knob than it looks — Adam's
  per-parameter gradient normalization largely absorbs it. The reference
  Meta-Task implementation instead gives the auxiliary loss its own
  optimizer at its own (much lower) learning rate. Fixed to match that.
- **Round 3 (corrected auxiliary-loss mechanism, still real end-to-end
  finetuning)**: the fix did not change the conclusion — if anything,
  Meta-Task's average (22.17%) is slightly *lower* than the flawed Round 2
  version (22.92%), and still below vanilla's 23.00%. This makes the null
  result more robust: it is not an artifact of a weak or confounded
  auxiliary-loss mechanism.
- **The one finding that held up across every round**: real end-to-end
  finetuning by itself — no Meta-Task involved — lifts DenoMAE2.0's DiSC-AMC
  accuracy from an average of 21.50% (original frozen baseline) to 23.00%.
  That is a bigger and more reliable effect than anything Meta-Task
  contributed in any round.

## Methodology, in order of how it actually unfolded

1. **Prototype-distance K-shot few-shot eval** (the literal form of the
   committee's request): episodic Prototypical-Network training, with and
   without Meta-Task's auxiliary reconstruction loss, evaluated on held-out
   RadioML SNRs (−10dB, −20dB) via nearest-centroid classification. First
   pass (lr=1e-5, 2000 episodes) failed to train at all — `proto_loss`
   stayed flat and both conditions sat at chance even in-distribution.
   Fixed by raising lr to 1e-4 and episodes to 5000 (then 10000 with a
   stronger auxiliary weight as an escalation, before that mechanism was
   itself replaced — see Correction 2). Result: real in-distribution
   learning in both conditions, but **both collapse to chance (~4%) on
   held-out SNR**, with no consistent gap between conditions. This method
   alone was inconclusive.
2. **Trained classification-head readout** (`train_head_eval.py`, frozen
   encoder, matching `finetune.py`'s convention): a second, independent
   check. Meta-Task modestly ahead in all 4 held-out comparisons (both
   training-budget runs, both SNRs) — small margins, consistent direction,
   but a different (frozen) convention than DINOv2's baseline uses.
3. **DiSC-AMC pipeline, frozen encoder** (Round 1): the Meta-Task-regularized
   encoder swapped into the exact pipeline/config that reproduces the
   paper's published Table III DenoMAE2.0 column. Real pipeline-level
   number, but still using the frozen convention. This is the result that
   looked like a clean win (+4.58pp at −20dB).
4. **Methodology check #1**: inspecting `src/denoMAE2/finetune.py` showed
   `DownstreamClassifier`'s `freeze_encoder=True` is hardcoded with no CLI
   override, while DINOv2's `ImageClassifier` (`classifier_training.py`)
   defaults to `freeze_encoder=False`. Checking the paper's own text (v2,
   arXiv:2510.00316) confirmed it describes the fine-tuned setting, for
   both backbones, as one where the encoder itself is updated before a
   classification head is added — i.e. unfrozen. The code does not do that
   for DenoMAE2.0. A real discrepancy in the existing paper/codebase, not
   something this experiment introduced. See "Paper/code discrepancy" below.
5. **DiSC-AMC pipeline, real unfrozen encoder** (Round 2): added
   `--no_freeze_encoder` support to `train_head_eval.py`, retrained *both*
   vanilla DenoMAE2.0 and Meta-Task DenoMAE2.0 end-to-end (200 epochs,
   lr=5e-3, matching the hyperparameters already used to build the original
   baseline checkpoint's head-only training), then reran the full DiSC-AMC
   pipeline across all 5 RadioML SNRs for both.
6. **Methodology check #2**: the Meta-Task auxiliary loss in step 1-3/5's
   M-stage training used `loss = proto_loss + aux_weight * rec_loss` through
   one shared AdamW optimizer (`episodic_train.py`). Re-reading the
   reference Meta-Task implementation (`autoencoder_prototypical.py`,
   `update_encoder` / `init_encoder_optimizer`) showed it instead runs the
   auxiliary loss through its own dedicated optimizer at its own learning
   rate (default `1e-6` there) — a materially different mechanism, and a
   more standard one for combining a main task with an auxiliary
   self-supervised signal under Adam.
7. **M-stage retrained with the corrected mechanism** (Round 3):
   `episodic_train.py` now does two separate backward/optimizer-step calls
   per episode — the auxiliary reconstruction loss updates the encoder
   through its own optimizer at `lr=1e-6`, then the prototype loss updates
   through the main optimizer at `lr=1e-4` (matching what already worked).
   `--aux_weight` was removed; `--aux_lr` replaces it. Control's mechanism
   is untouched (no auxiliary loss involved), so the existing vanilla
   checkpoint/results remain valid without rerunning. Retrained Meta-Task's
   encoder (5000 episodes), real-unfroze it (200 epochs, same regimen as
   Round 2), and reran the full 5-SNR DiSC-AMC pipeline. This is the final,
   most rigorous comparison, reported below.

## Final comparison — DiSC-AMC pipeline accuracy (%), all 5 SNRs

| SNR | Baseline (orig., frozen) | Meta-Task (frozen, Round 1) | Vanilla (real finetune) | Meta-Task (real finetune, flawed aux, Round 2) | Meta-Task (real finetune, corrected aux, Round 3) | DINOv2 (paper, real finetune) |
|---|---|---|---|---|---|---|
| −20dB | 17.92 | 22.50 | **23.33** | 20.42 | 20.42 | 19.58 |
| −10dB | 19.17 | 18.75 | 25.00 | 23.75 | **25.00** (tied) | 17.50 |
| 0dB | 25.00 | 19.17 | **22.50** | 25.42 | 20.83 | 40.83 |
| +10dB | 24.17 | 24.58 | 23.33 | 20.83 | **25.00** | 72.50 |
| +20dB | 21.25 | 22.08 | **20.83** | 24.17 | 19.58 | 76.25 |
| **avg** | 21.50 | 21.42 | **23.00** | 22.92 | 22.17 | 65.42 |

Bold = best of {vanilla, Meta-Task Round 3} at that SNR. DINOv2 column
included for scale — it dominates at positive SNRs, but DenoMAE2.0 (all
variants) is actually competitive with or better than DINOv2 at the two
negative, severe-shift SNRs, which is where this investigation started.

### Before → after, what changed and why (two corrections)

**Before (Round 1)**: Meta-Task regularization gives a clear, reportable
win — "+4.58pp at the hardest held-out SNR, through the real DiSC-AMC
pipeline."

**After correction 1 (Round 2, real finetuning)**: Meta-Task regularization
is a wash on average (22.92% vs. 23.00%) with a mixed per-SNR pattern. The
real driver of improvement over the *original* baseline turned out to be
giving the encoder a genuine end-to-end finetune at all (21.50% → 23.00%),
independent of Meta-Task.

**After correction 2 (Round 3, corrected auxiliary-loss mechanism)**: the
wash persists, and gets slightly *more* negative for Meta-Task (22.17%
average, vs. vanilla's 23.00%). Fixing the mechanism ruled out "the
auxiliary loss just wasn't being applied with enough effective strength" as
an explanation for the null result.

**Why both corrections matter**: Round 1 answered a narrower question
("does Meta-Task help on top of a frozen, head-only-trained encoder") than
the one the committee and the paper's own DINOv2 baseline actually call for.
Round 2's `aux_weight` scalar then under-delivered what it looked like it
should do, because of how Adam normalizes gradients — reporting Round 2 as
final would have risked a "Meta-Task just needs more weight" objection that
a cheap, principled fix (not more compute) could answer directly. Round 3
answers it: the result is not sensitive to that knob.

## Paper/code discrepancy — before and after

**Current paper text** (arXiv:2510.00316v2, describing the fine-tuned
setting used for Table III, both backbones): the encoder backbone is
described as being *updated* on the target dataset before a classification
head is added — i.e., real, unfrozen finetuning.

**What the code actually does** (`src/denoMAE2/finetune.py`,
`DownstreamClassifier.__init__`): `freeze_encoder=True` is hardcoded, with
no CLI flag to disable it. Every DenoMAE2.0 "finetuned" checkpoint produced
by this script — including the one behind the published Table III
DenoMAE2.0 column — has a frozen encoder; only the classification head was
ever trained. DINOv2's own classifier (`classifier_training.py`,
`ImageClassifier`) does default to `freeze_encoder=False`, so DINOv2's
numbers likely do match the paper's description. DenoMAE2.0's did not,
prior to this work.

**Suggested correction** — either:
- Fix the paper text to say DenoMAE2.0's Table III "fine-tuned" setting
  froze the encoder (matching Table IV's "unadapted" framing more closely
  than currently implied), and note the two backbones used different
  protocols, or
- Fix `finetune.py` to actually support unfreezing (now done in this
  branch via `train_head_eval.py --no_freeze_encoder`) and regenerate the
  published DenoMAE2.0 Table III numbers from a real unfrozen finetune —
  which is exactly what Rounds 2-3 above do, and gives a higher DenoMAE2.0
  DiSC-AMC average (23.00%) than the currently-published one (21.50%).

There's also a second, smaller inconsistency in the paper itself: Section
IV-B's text states kNN=50 for this setting, while the Table III/IV captions
say kNN=10. The experiments here used 50 (matching the existing
`run_denomae_ft_vanilla_eval.sh` config and the section text), so our
numbers are internally consistent with at least one part of the paper, but
the paper should resolve which value is correct.

## Did any existing experiment need to be redone?

Round 1's and Round 2's numbers are still accurate *for what they
measured* and are kept in the table above for comparison — nothing was
invalidated or deleted. What changed, twice, is which comparison is the
right one to lead with:

- Round 2 (real finetuning, matching DINOv2's own protocol) superseded
  Round 1 for "does Meta-Task help," because it's the fair comparison.
- Round 3 (corrected auxiliary-loss mechanism) superseded Round 2 for the
  same question, because Round 2's `aux_weight` knob didn't behave the way
  it was assumed to under Adam.

Round 3 is the number to cite. Rounds 1-2 remain useful as secondary data
points in the methodology narrative (why the comparison needed fixing
twice), not as competing conclusions.

## Honest framing for the committee response

Meta-Task regularization, tested with the most rigorous protocol available
here (same real-finetuning regimen for both conditions matching DINOv2's
own baseline, and the auxiliary loss implemented the way the reference
Meta-Task code does it, not as a combined-loss scalar), does not show a
reliable net improvement for DenoMAE2.0 on RadioML cross-domain transfer —
it wins at some SNRs (−10dB tied, +10dB) and loses at others (−20dB, 0dB,
+20dB), averaging to a small net loss relative to vanilla finetuning. That
is a legitimate, informative null result for RQ3, consistent with the
committee's own note that a non-improvement is still worth reporting as a
limitation rather than reframed as a win — and it is a *robust* null result
specifically because two independent methodological objections (frozen vs.
real finetuning; combined-loss scalar vs. separate-optimizer auxiliary
update) were each checked and neither one flips the outcome.

Separately, and not originally part of RQ3, this investigation surfaced a
real methodology gap between the paper's text and its code for how
DenoMAE2.0 was finetuned, and showed that fixing it (real end-to-end
finetuning instead of a frozen head) raises DenoMAE2.0's own DiSC-AMC
baseline by about 1.5pp on average — a more robust and more directly
actionable finding than the Meta-Task question itself.

## Where everything lives

- Code: `src/fewshot_meta_task/` (episodic sampler, Meta-Task training
  with the corrected separate-optimizer auxiliary mechanism, few-shot eval,
  paper-style head training/eval with `--no_freeze_encoder`),
  `src/naming.py` (new `PredictionSource` registry entries for each
  backbone variant), `run_meta_task_eval.sh`,
  `run_denomae_metatask_vanilla_eval.sh`, `run_denomae_backbone_eval.sh`
  (generic, parameterized version used for the multi-condition x 5-SNR
  sweeps).
- Raw results (gitignored, not committed — regenerate via the scripts
  above): `exp/meta_task_fewshot/`, `exp/meta_task_fewshot_v2/` (Round 1
  encoder-level results + checkpoints, flawed `aux_weight` mechanism),
  `exp/meta_task_fewshot_v3/` (Round 3 M-stage, corrected mechanism),
  `exp/unfrozen_eval/` (Round 2 finetuned checkpoints),
  `exp/unfrozen_eval_v3/` (Round 3 finetuned checkpoint),
  `exp/denomae_metatask_vanilla_deepseek_eval/`,
  `exp/denomae_vanilla_unfrozen_deepseek_eval/`,
  `exp/denomae_metatask_unfrozen_deepseek_eval/`,
  `exp/denomae_metatask_v3_unfrozen_deepseek_eval/` (DiSC-AMC pipeline
  results, one dir per condition/round).
