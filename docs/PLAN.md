# Plan — sequencing the remaining work

[`ROADMAP.md`](../ROADMAP.md) is the idea backlog ordered by expected value.
[`NEXT-SESSION.md`](NEXT-SESSION.md) is the transient handoff: current state and
what was already verified. **This file is the third thing: what order to do it
in, and why that order.** It exists because several items are blocked on each
other in ways that are not obvious from either list, and doing them out of order
means measuring things twice.

## The dependency that shapes everything

Four separate pieces of work all wait on the same question:

```
  measure MTP acceptance  ─┬─►  flip --prefix-reuse default  ─►  re-measure ngram screen
      DONE: unaffected     │            DONE                          (needs GPU)
                           └─►  MTP advisory: NOT needed  ─────►  cut 0.2.0  <-- next
```

n-gram speculation turned out inflated 2.3x by the identical-prompt harness
([`workload-shape-design.md`](workload-shape-design.md)). MTP drafts from the
model's own NextN head rather than from cross-request n-gram state, so it is
*probably* unaffected — which is the exact word used about ngram before anyone
measured it.

Measured 2026-08-26: **MTP is unaffected**, so the advisory stays ngram-only and
steps 2 and 3 are clear to proceed. The reasoning below is kept because it is why
this went first.

Until it was measured:

- the 0.2.0 advisory cannot say whether MTP results are affected;
- the right `--prefix-reuse` default is unclear for MTP-capable models;
- and a full Qwen3.8-27B sweep (~3h21m) cannot be trusted to answer what it is
  usually run to answer.

Ten minutes of GPU resolves all three. **It goes first whenever the card is
free.**

## Order

### 1. ~~Measure MTP acceptance across reuse levels~~ — **done**

**MTP is unaffected.** Acceptance moves 0.78 → 0.70 across the whole reuse range
and reproduces to ±0.02, against n-gram's 1.00 → 0.31. MTP sweep results stand;
only ngram needs the advisory. Full table and the thermal control in
[`workload-shape-design.md`](workload-shape-design.md).

Unblocks steps 2 and 3.

### 2. ~~Flip the `--prefix-reuse` default~~ — **done**

The blast radius is already measured: non-speculative `tg` is flat across reuse
(371.6 / 372.0 / 368.9 / 368.6 t/s, a spread smaller than run-to-run noise), so
changing the default moves *only* speculative results — exactly the ones
currently wrong. Landed as default 0, with `agents` at 90 because that use case is itself a claim
about traffic. Validating it on the GPU found two further contamination layers in
the prompt generator and one defect in `draft_acc` itself — see
[`workload-shape-design.md`](workload-shape-design.md). Corrected figures: ngram
inflation is ~1.46x, not the 2.3x earlier drafts claimed.

### 3. Cut 0.2.0 — *mechanical*

Bump `__version__`, move `[Unreleased]` to `[0.2.0]`, tag. The ⚠️ **Affects
existing results** section is the reason to ship rather than sit on it: users who
have already run ngram sweeps need to know their numbers were upper bounds.

### 4. ~~The OOM pruner~~ — **already done, now validated**

This step was written on a stale reading of ROADMAP item 2. The pruner does not
need a memory model of ours and never did: `predict_fits` asks
`llama-fit-params`, llama.cpp's own estimator — the better design by this
project's own rule about asking llama.cpp rather than modelling it. F6's
six-term model is therefore *not* wanted here; it stays on file as a fallback
shape if a future backend has no equivalent tool.

What was genuinely open was item 2's last bullet, live-GPU validation, and that
is now done: the boundary is real and the pruner is not over-conservative. It
also surfaced the `SIGNAL` status. Details in [`../ROADMAP.md`](../ROADMAP.md).

### 5. ~~C2 / `kv_unified`~~ — **done**

K3 settled empirically rather than by argument: at `-c 2048`, four split slots
get `n_ctx_slot = 512` each and four unified slots get 2048 each, so
`n_ctx = per_slot x (slots if split else 1)`. The `concurrency` factor implements
that, `auto` is reachable for the first time, and `kv_unified` is recorded per
row. All four levels verified against llama.cpp's own log —
[`concurrency-kv-design.md`](concurrency-kv-design.md).

### 6. Draft model / `-md` — *designed, one prerequisite*

[`draft-model-design.md`](draft-model-design.md). Unlocks ~12 `--spec-draft-*`
knobs plus `draft-simple`/`draft-dflash`/`draft-dspark`. Prerequisite:
`predict_fits` must account for two resident models — which is step 4, so these
are naturally sequential.

### 7. Multi-GPU — *blocked on hardware we do not have*

[`multi-gpu-design.md`](multi-gpu-design.md). Tiers 1 and 2 of its test plan
(parser tests, single-NVIDIA validation) are doable; tier 3 — the split itself —
needs two non-identical GPUs and by construction cannot be tested here.

## Standing items, no ordering

- **`--audit-flags`**, so coverage stops rotting silently. Measured drift across
  one routine pull: 2 flags added, 0 removed, 4 of 12 cited line numbers moved
  ([`flag-coverage.md`](flag-coverage.md)).
- **The four hardware-poisoned constants** left as findings in
  [`constants-audit.md`](constants-audit.md). `FIXED_FA` is the sharpest and the
  least tractable: it needs a non-ROCm backend to validate.
- **RoPE/YaRN tail** — complete the family or state the scope.

## What this plan assumes

That the GPU is shared and frequently busy, so anything needing it is scheduled
opportunistically rather than depended on. Everything in steps 2–6 is
reachable with `--selftest` and plan-only runs; only steps 1 and 7 genuinely
require hardware.
