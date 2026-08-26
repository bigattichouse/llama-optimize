# Handoff — open work as of 2026-08-26

Working state at the end of the 2026-08-25/26 sessions. Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything below is committed and pushed to `main` (through `c6881b7`), the
working tree is clean, and `--selftest` passes.

## Do these first

### 1. Confirm CI is actually green

The last several runs failed with *"The job was not acquired by Runner of type
hosted even after multiple attempts"* — empty step list, conclusion `cancelled`.
That is GitHub runner starvation, **not** a test failure; `--selftest` passes
locally on every one of those commits. A re-run was queued and had not been
picked up either. Check `gh run list` and re-run if still red before assuming any
of the recent work broke something.

### 2. Cut 0.2.0

Everything in [`CHANGELOG.md`](../CHANGELOG.md) `[Unreleased]` is done and
verified. To release: change `__version__` in `llama-optimize.py` from
`0.2.0-dev`, move the `[Unreleased]` heading to `[0.2.0]` with a date, and tag.
`v0.1.0` is already tagged at `bbb206a` as the pre-provenance baseline.

The changelog's **⚠️ Affects existing results** section is the point of the
release: ngram numbers are upper bounds, and concurrency sweeps never measured
`kv_unified`. Both need to reach users who have already run sweeps.

### 3. Measure whether MTP acceptance is inflated (needs GPU, ~10 min)

**This blocks trusting any MTP sweep**, including on the Qwen3.8-27B that
prompted the field reports.

We proved n-gram speculation is inflated by the identical-prompt harness
([`workload-shape-design.md`](workload-shape-design.md)): 100% acceptance and
2.3–3.4x throughput on a repeated prompt, no drafting at all on distinct ones.
MTP drafts from the model's own NextN head rather than from cross-request n-gram
state, so it is *probably* unaffected — but "probably" is exactly what was said
about ngram before measuring.

Method (the ngram test, rerun with MTP): load a NextN model with
`--spec-type draft-mtp`, send the same prompt three times, then three distinct
prompts, and compare `draft_n` / `draft_n_accepted` in the response `timings`.
`draft_acc` already records this. If MTP is flat across both, MTP results stand
and only ngram needs the caveat.

## Designed, not built

Each has a checklist and a no-GPU test plan in its own doc.

| what | doc | notes |
|---|---|---|
| **Workload shape / `--prefix-reuse`** | [`workload-shape-design.md`](workload-shape-design.md) | **Highest value.** Now a correctness fix, not a feature: it is the mechanism by which ngram gets measured honestly at all. Prerequisite: re-derive `TG_OVER_PP_LIMIT` under partial reuse *before* any cache factor lands, because a too-tight limit fails silently by deleting real configs |
| **Concurrency / `kv_unified`** | [`concurrency-kv-design.md`](concurrency-kv-design.md) | Design done this session. Blocked on settling the per-regime `n_ctx` rule (K3) first — a slot's context differs between shared and split KV, and getting it wrong recreates the batch-floor defect somewhere new |
| **Draft model / `-md`** | [`draft-model-design.md`](draft-model-design.md) | Unlocks ~12 `--spec-draft-*` knobs plus `draft-simple`/`draft-dflash`/`draft-dspark`. Prerequisite: `predict_fits` must account for two models resident |
| **Multi-GPU / `-ts`,`-sm`,`-mg`** | [`multi-gpu-design.md`](multi-gpu-design.md) | Predates these sessions. Needs two non-identical GPUs to validate the part that matters |

## Smaller open items

- **`-sps`/`--slot-prompt-similarity`** — the server-side prefix-reuse routing
  knob. Unmeasurable until `--prefix-reuse` exists; folded into the workload-shape
  design.
- **Auto-sweep decisions** for the newly registered knobs
  ([`remaining-factors-design.md`](remaining-factors-design.md)): can SWA be
  detected from GGUF metadata (would gate `swa_full` the way `n_nextn` gates
  `mtp`)? Should `load_mode` auto-sweep when the model does not fit in VRAM —
  which is detectable, and exactly where paging decides throughput?
- **RoPE/YaRN tail** — we sweep `rope_scaling` and `yarn_factor` and stop.
  Complete the family or state the scope; an incomplete family reads as a
  deliberate decision and is not one.
- **`--audit-flags`** — parse `--help`, diff against `FACTORS` plus an in-code
  exclusion allow-list, report anything in neither. Measured drift across one
  routine pull: 2 flags added, 0 removed, 4 of 12 cited line numbers moved. See
  [`flag-coverage.md`](flag-coverage.md).
- **Re-measure the ngram screen** under distinct prompts once `--prefix-reuse`
  lands, and say plainly in the report that earlier numbers were upper bounds.

## From the Bayesian autotuner (F6)

[`field-reports.md`](field-reports.md) F6 reviews
[SergioMorillas/vllm-bayesian-autotuner](https://github.com/SergioMorillas/vllm-bayesian-autotuner).
Three concrete things to pull from it:

1. **Their six-term memory model is our OOM pruner, already specified.** ROADMAP
   item 2 has been blocked on what to sum; they sum weights + KV + Mamba state +
   CUDA-graph overhead + activations + margin against two inequalities. The
   llama.cpp translation drops Mamba and CUDA graphs and gains the
   partial-offload split.
2. **Objectives we do not measure**: TTFT (ours is derived, not measured — ROADMAP
   5), TPOT and goodput (cheap now that per-request timings are recorded), and
   **tokens/joule**, which is a genuinely new axis and interesting on a card whose
   thermal ceiling is already the dominant noise source.
3. **Their prompt-battery design doc** is worth reading before building
   `--prefix-reuse`.

## Verified this session, so you do not need to re-check

- llama.cpp rebuilt at `4d19b2876` (ROCm 7.2.1, gfx906) and confirmed on real
  inference: `backend=ROCm`, 444 t/s tg on gemma-3-270m vs 115 t/s CPU-only.
- All 37 flags the registry emits are still accepted by that build; none removed.
- Every newly registered knob's spelling is accepted by `llama-server`.
- The emitted command carries exactly one `--load-mode`, and default-on booleans
  emit their off-spelling.
- `tool_version` / `llama_build` / `backend` land in real results CSVs.
- The GPU-visibility warning fires on a blinded GPU and stays silent otherwise.

## A note on the machine

The GPU is shared and frequently busy — ask before running anything that loads a
model. `--selftest` needs no GPU, and plan-only runs (omit `--run`) need no GPU
either, which covers most verification.
