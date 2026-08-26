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
      (needs GPU, ~10m)    │            (small)                       (needs GPU)
                           └─►  MTP advisory: needed or not?  ─►  cut 0.2.0
```

n-gram speculation turned out inflated 2.3x by the identical-prompt harness
([`workload-shape-design.md`](workload-shape-design.md)). MTP drafts from the
model's own NextN head rather than from cross-request n-gram state, so it is
*probably* unaffected — which is the exact word used about ngram before anyone
measured it.

Until that is measured:

- the 0.2.0 advisory cannot say whether MTP results are affected;
- the right `--prefix-reuse` default is unclear for MTP-capable models;
- and a full Qwen3.8-27B sweep (~3h21m) cannot be trusted to answer what it is
  usually run to answer.

Ten minutes of GPU resolves all three. **It goes first whenever the card is
free.**

## Order

### 1. Measure MTP acceptance across reuse levels — *needs GPU, ~10 min*

Method already proven on ngram: load an MTP model with `--spec-type draft-mtp`,
send identical prompts then distinct ones, compare `draft_acc`. Flat across reuse
means MTP results stand and only ngram needs the advisory; not flat means the
advisory widens, and it is far better to know before a three-hour sweep than
after.

### 2. Flip the `--prefix-reuse` default — *small, no GPU*

The blast radius is already measured: non-speculative `tg` is flat across reuse
(371.6 / 372.0 / 368.9 / 368.6 t/s, a spread smaller than run-to-run noise), so
changing the default moves *only* speculative results — exactly the ones
currently wrong. Recommended shape: default 0, with `agents` at 90 because that
use case is itself a claim about traffic. Waits on step 1 only because the result
may change what the advisory says.

### 3. Cut 0.2.0 — *mechanical*

Bump `__version__`, move `[Unreleased]` to `[0.2.0]`, tag. The ⚠️ **Affects
existing results** section is the reason to ship rather than sit on it: users who
have already run ngram sweeps need to know their numbers were upper bounds.

### 4. The OOM pruner — *no longer blocked*

ROADMAP item 2 was stalled on *what to sum*. F6 supplied it: weights + KV +
state + graph overhead + activations + margin, against a usable pool and a hard
VRAM limit, first-order additive on purpose because conservative is the point
([`field-reports.md`](field-reports.md) F6). The llama.cpp translation drops
Mamba and CUDA graphs and gains the partial-offload split. Independent of
everything above — good work to interleave.

### 5. C2 / `kv_unified` — *designed, blocked on one decision*

[`concurrency-kv-design.md`](concurrency-kv-design.md). Blocked on settling the
per-regime `n_ctx` rule (K3): a slot's context differs between shared and split
KV, and guessing recreates the batch-floor defect somewhere new. That decision is
the work; the factor is easy after it.

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
