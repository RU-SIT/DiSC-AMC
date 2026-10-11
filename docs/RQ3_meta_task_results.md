# RQ3: Meta-Task Regularization for DenoMAE2 — Final Results

Committee ask (proposal defense): compare the *same backbone* with and
without Meta-Task regularization on few-shot/cross-domain AMC adaptation,
reporting post-adaptation accuracy, improvement over baseline, and
adaptation cost — using an existing AMC dataset and the existing Meta-Task
implementation rather than a new framework. Code: `src/fewshot_meta_task/`.
Original plan: `/home/woreom/.claude/plans/declarative-soaring-bubble.md`.

## TL;DR

Two rounds of experiments, with a significant methodology correction in
between. **The second round is the one to cite.**

- **Round 1 (frozen encoder)**: Meta-Task regularization looked like a real
  win — +4.58pp at −20dB in the full DiSC-AMC pipeline. But the comparison
  used a frozen encoder (classification head trained, encoder weights
  untouched) for both conditions.
- **Correction**: the frozen-encoder convention doesn't match how DINOv2's
  baseline in the paper was built (DINOv2's encoder *is* updated during
  finetuning; DenoMAE2.0's, in the code as it stands, is not). This asymmetry
  was already present in the published paper, not something introduced here.
- **Round 2 (real, end-to-end finetuned encoder, matching DINOv2's
  convention)**: Meta-Task's apparent edge **mostly disappears** — averaged
  over all 5 SNRs, Meta-Task-regularized DenoMAE2.0 (22.92%) is marginally
  *worse* than vanilla DenoMAE2.0 (23.00%) once both get the same real
  finetuning treatment. The per-SNR picture is a mixed bag, not a clean win.
- **The more robust, more useful finding**: real end-to-end finetuning by
  itself — no Meta-Task involved — lifts DenoMAE2.0's DiSC-AMC accuracy from
  an average of 21.50% (original frozen baseline) to 23.00%. That's a bigger
  and more reliable effect than anything Meta-Task contributed.

## Methodology, in order of how it actually unfolded

1. **Prototype-distance K-shot few-shot eval** (the literal form of the
   committee's request): episodic Prototypical-Network training, with and
   without Meta-Task's auxiliary reconstruction loss, evaluated on held-out
   RadioML SNRs (−10dB, −20dB) via nearest-centroid classification. First
   pass (lr=1e-5, 2000 episodes) failed to train at all — `proto_loss`
   stayed flat and both conditions sat at chance even in-distribution.
   Fixed by raising lr to 1e-4 and episodes to 5000 (then 10000 with
   `aux_weight=5.0` as a stronger escalation). Result: real in-distribution
   learning in both conditions, but **both collapse to chance (~4%) on
   held-out SNR**, with no consistent gap between conditions, even after the
   escalation. This method alone was inconclusive.
2. **Trained classification-head readout** (`train_head_eval.py`, frozen
   encoder, matching `finetune.py`'s convention): a second, independent
   check. Meta-Task modestly ahead in all 4 held-out comparisons (both
   training-budget runs, both SNRs) — small margins, consistent direction,
   but a different (frozen) convention than DINOv2's baseline uses.
3. **DiSC-AMC pipeline, frozen encoder** (Round 1): the Meta-Task-regularized
   encoder swapped into the exact pipeline/config that reproduces the
   paper's published Table III DenoMAE2.0 column. Real pipeline-level
   number, but still using the frozen convention. This is the result that
   looked like a clean win (+4.58pp at −20dB) — see "Round 1 table" below.
4. **Methodology check**: inspecting `src/denoMAE2/finetune.py` showed
   `DownstreamClassifier`'s `freeze_encoder=True` is hardcoded with no CLI
   override, while DINOv2's `ImageClassifier` (`classifier_training.py`)
   defaults to `freeze_encoder=False`. Checking the paper's own text (v2,
   arXiv:2510.00316) confirmed it describes the fine-tuned setting as one
   where "the encoder backbone is updated on the target dataset before
   adding a classification head" — i.e., unfrozen — for *both* backbones.
   The code does not do that for DenoMAE2.0. This is a real discrepancy in
   the existing paper/codebase, not something this experiment introduced.
   See "Paper/code discrepancy" below.
5. **DiSC-AMC pipeline, real unfrozen encoder** (Round 2): added
   `--no_freeze_encoder` support to `train_head_eval.py`, retrained *both*
   vanilla DenoMAE2.0 and Meta-Task DenoMAE2.0 end-to-end (200 epochs,
   lr=5e-3, matching the hyperparameters already used to build the original
   baseline checkpoint's head-only training), then reran the full DiSC-AMC
   pipeline across all 5 RadioML SNRs for both. This is the apples-to-apples
   comparison, and it's the one reported as final below.

## Final comparison — DiSC-AMC pipeline accuracy (%), all 5 SNRs

| SNR | Baseline (orig., frozen) | Meta-Task (frozen, Round 1) | Vanilla (real finetune, Round 2) | Meta-Task (real finetune, Round 2) | DINOv2 (paper, real finetune) |
|---|---|---|---|---|---|
| −20dB | 17.92 | 22.50 | **23.33** | 20.42 | 19.58 |
| −10dB | 19.17 | 18.75 | **25.00** | 23.75 | 17.50 |
| 0dB | 25.00 | 19.17 | 22.50 | **25.42** | 40.83 |
| +10dB | 24.17 | 24.58 | **23.33** | 20.83 | 72.50 |
| +20dB | 21.25 | 22.08 | 20.83 | **24.17** | 76.25 |
| **avg** | 21.50 | 21.42 | **23.00** | 22.92 | 65.42 |

Bold = better of {vanilla, Meta-Task} *under the real-finetune, apples-to-apples
condition* at that SNR. DINOv2 column included for scale — it dominates at
positive SNRs, but DenoMAE2.0 (both variants) is actually competitive or
better than DINOv2 at the two negative, severe-shift SNRs, which is where
this whole investigation started.

### Before → after, what changed and why

**Before** (what RQ3 looked like after Round 1): Meta-Task regularization
gives a clear, reportable win — "+4.58pp at the hardest held-out SNR,
through the real DiSC-AMC pipeline."

**After** (Round 2, apples-to-apples): Meta-Task regularization is a wash on
average (22.92% vs. 23.00%) with a mixed per-SNR pattern — it wins at 0dB
and +20dB, loses at −20dB, −10dB, and +10dB. The real driver of improvement
over the *original* baseline turned out to be giving the encoder a genuine
end-to-end finetune at all (21.50% → 23.00%), independent of Meta-Task.

**Why the correction matters**: Round 1's comparison wasn't wrong, it was
answering a narrower question ("does Meta-Task help on top of a frozen,
head-only-trained encoder") than the one the committee and the paper's own
DINOv2 baseline actually call for ("does Meta-Task help a properly
finetuned encoder"). Reporting Round 1 alone would have overstated Meta-Task's
contribution.

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
  which is exactly what Round 2 above does, and gives a higher DenoMAE2.0
  DiSC-AMC average (23.00%) than the currently-published one (21.50%).

There's also a second, smaller inconsistency in the paper itself: Section
IV-B's text states kNN=50 for this setting, while the Table III/IV captions
say kNN=10. The experiments here used 50 (matching the existing
`run_denomae_ft_vanilla_eval.sh` config and the section text), so our
numbers are internally consistent with at least one part of the paper, but
the paper should resolve which value is correct.

## Did any existing experiment need to be redone?

No experiment was invalidated or needed to be deleted — Round 1's frozen-
encoder numbers are still accurate *for what they measured* and are kept
in the tables above for comparison. What changed is which comparison is the
right one to lead with: Round 2 (real finetuning, matching DINOv2's own
protocol) supersedes Round 1 as the number to cite for "does Meta-Task
help," because it's the fair comparison. Round 1 remains useful as a
secondary data point showing Meta-Task's effect is sensitive to whether the
encoder gets to adapt further downstream.

## Honest framing for the committee response

Meta-Task regularization, tested properly (same real-finetuning protocol
for both conditions, matching how the paper's own DINOv2 baseline was
built), does not show a reliable net improvement for DenoMAE2.0 on RadioML
cross-domain transfer — it wins at some SNRs and loses at others, averaging
to a wash. That is a legitimate, informative null result for RQ3, consistent
with the committee's own note that a non-improvement is still worth
reporting as a limitation rather than reframed as a win.

Separately, and not originally part of RQ3, this investigation surfaced a
real methodology gap between the paper's text and its code for how
DenoMAE2.0 was finetuned, and showed that fixing it (real end-to-end
finetuning instead of a frozen head) raises DenoMAE2.0's own DiSC-AMC
baseline by about 1.5pp on average — a more robust and more directly
actionable finding than the Meta-Task question itself.

## Where everything lives

- Code: `src/fewshot_meta_task/` (episodic sampler, Meta-Task training,
  few-shot eval, paper-style head training/eval), `src/naming.py` (new
  `PredictionSource` registry entries for each backbone variant),
  `run_meta_task_eval.sh`, `run_denomae_metatask_vanilla_eval.sh`,
  `run_denomae_backbone_eval.sh` (generic, parameterized version used for
  Round 2's 3-condition x 5-SNR sweep).
- Raw results (gitignored, not committed — regenerate via the scripts
  above): `exp/meta_task_fewshot/`, `exp/meta_task_fewshot_v2/` (Round 1
  encoder-level results + checkpoints), `exp/unfrozen_eval/` (Round 2
  finetuned checkpoints), `exp/denomae_metatask_vanilla_deepseek_eval/`,
  `exp/denomae_vanilla_unfrozen_deepseek_eval/`,
  `exp/denomae_metatask_unfrozen_deepseek_eval/` (Round 1 and Round 2
  DiSC-AMC pipeline results, one dir per condition).
