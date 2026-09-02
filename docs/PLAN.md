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
                           └─►  MTP advisory: NOT needed  ─────►  cut 0.2.0  DONE
```

That chain is finished. The sequencing question since then has been a different
one — see *After 0.2.0* below.

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

### 3. ~~Cut 0.2.0~~ — **done** (2026-08-26)

Shipped with the ⚠️ **Affects existing results** section, which was the reason to
ship rather than sit on it: users who had already run ngram sweeps needed to know
their numbers were upper bounds.

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

### 6. Draft model / `-md` — *mostly landed, one path unverified*

[`draft-model-design.md`](draft-model-design.md). The prerequisite is met —
`predict_fits` prices the second resident model (step 4) — and the speculative
types are reachable: `draft-simple`, `draft-dflash`, `draft-dspark`,
`draft-eagle3` and `draft-mtp` all emit correctly, chosen from the draft GGUF's
own architecture where llama.cpp can infer it and named where it cannot
(issue #19).

Three bugs on this path were found and fixed on the way, all of them silent: a
supplied draft was loaded and then ignored on a target with its own MTP head; a
*plain* draft model never speculated at all, because llama.cpp infers nothing
from an ordinary model and its default type is `none`; and the pasted command
omitted `-md` entirely, so it could not reproduce the row above it.

**What is left is verification, not construction.** `draft-eagle3` has never
actually run — see [`NEXT-SESSION.md`](NEXT-SESSION.md), including which head to
use and which one not to waste an hour on.

### 7. ~~`ngl` levels that know about VRAM~~ — **done** (`aa63d19`)

[Issue #14](https://github.com/bigattichouse/llama-optimize/issues/14), closed.
When `predict_fits` says every layer fits at the deepest depth and largest KV in
the design, `ngl_levels` spans the top quarter instead of the whole range:
`[0, 30, 33, 37, 40]` rather than `[0, 10, 20, 30, 40]` on a 40-layer model.
`ngl=0` survives every verdict, because the verdict can be wrong and it is then
the only row that can still measure something (SC5); `--no-oom-prune` restores the
even span. Reasoning in [`sweep-cost-design.md`](sweep-cost-design.md).

The framing changed while doing it, and the note is worth keeping: this is not
"80% of a factor's levels wasted". In an orthogonal array every row informs every
factor's main effect, so those rows were never wasted for information. They cost
**wall clock** — `ngl=0` is CPU-only decode, a minority of the design and a
majority of its runtime — and **additivity**, since `kv_type` at `ngl=0` is not
the same phenomenon as at `ngl=40`.

Spun out: [issue #17](https://github.com/bigattichouse/llama-optimize/issues/17),
the `ngl`/`ncmoe` overlap on MoE models, which wants a real MoE model to settle.

**The ordering argument this entry carried is now moot** — it was placed ahead of
step 8 on cost, against the objection that it unblocks nobody. It is done, and
step 8 is next on its own merits.

### 8. Multi-GPU — *blocked on hardware we do not have*

[`multi-gpu-design.md`](multi-gpu-design.md). Tiers 1 and 2 of its test plan
(parser tests, single-NVIDIA validation) are doable; tier 3 — the split itself —
needs two non-identical GPUs and by construction cannot be tested here.

## Standing items, no ordering

- **`--audit-flags`**, so coverage stops rotting silently. Measured drift across
  one routine pull: 2 flags added, 0 removed, 4 of 12 cited line numbers moved
  ([`flag-coverage.md`](flag-coverage.md)).
- **The four hardware-poisoned constants** left as findings in
  [`constants-audit.md`](constants-audit.md). `FIXED_FA` is the sharpest and the
  least tractable: it needs a non-ROCm backend to validate, and see issue #20
  below for how far it got. `CHARS_PER_TOKEN` (C-B) is now measured rather than
  assumed.
- **RoPE/YaRN tail** — complete the family or state the scope.
- **Issue #20 — `fa` is pinned from one GPU.** Half done: `--factor fa=0,1` no
  longer builds unlaunchable rows (measured — `-fa off -ctk q8_0` cannot create a
  context, so the quantized `kv_type` levels are dropped automatically). What
  remains is whether the pin itself is right, and it needs a *constrained
  relation* between `fa` and `kv_type` rather than a level filter: the cells that
  must not exist are an interaction, and an orthogonal array cannot omit one cell.
  **Note the retraction** in [`constants-audit.md`](constants-audit.md) C-A — the
  "33% slower" measurement was a cold row against a hot one, and the question is
  still open rather than answered.
- **Issue #21 — shareable fingerprints. Done** (2026-09-02). Every sweep writes
  `<results>.fingerprint.json`; `--fingerprint` prints it standalone;
  `community/` holds the format and the match-distance table. What remains is
  **matching** — ranking how close another box is — which is the part with real
  value and which needs a corpus before its rules can be anything but a guess.
- **Issue #17 — `ngl` and `ncmoe` on MoE models.** Partly dissolved: on a *hybrid*
  MoE, `ngl` is now `[0, 99]` (#18) so the two no longer compete for the offload
  axis. Still open for a dense-attention MoE, which this box does not have.
- **Verify `draft-eagle3`.** Built and unit-tested, never run. Needs the PRISM
  GGUF head; the base-matched specdrift heads are a different EAGLE3 flavour and
  do not convert — see [`NEXT-SESSION.md`](NEXT-SESSION.md) for why, so nobody
  spends an hour rediscovering it.
- **`tg=0.00` with status `OK` at deep context.** Two rows measured `pp` fine and
  never produced a decode number. `measured_ok` keeps it out of the picks, so
  nothing is wrong in the report — but something did not measure and did not say
  so, which is the defect class the rest of this list exists for. Free to
  investigate.

## What this plan assumes

That the GPU is shared and frequently busy, so anything needing it is scheduled
opportunistically rather than depended on. Everything in steps 2–7 is
reachable with `--selftest` and plan-only runs; only steps 1 and 8 genuinely
require hardware.

## After 0.2.0 — what actually shapes the order now

The 2026-08-30 session cleared the sweep-cost work and the first draft-model
increment, so the dependency that shapes the *next* stretch is hardware, not
sequencing:

```
  GPU is free  ─┬─►  re-measure ngram screen (blocked since 0.2.0)
                ├─►  route 1 vs route 2 for MTP (issue #12, needs -md + a NextN model)
                └─►  live-validate the throughput floors actually save what the
                     arithmetic says
```

None of these can be faked in-process, and none of them ran this session because
the card belongs to another project. Everything that *could* be done without a
GPU has been; see [`NEXT-SESSION.md`](NEXT-SESSION.md) for the current state.

The one non-GPU item worth doing next is not a feature: three separate defects
this session shared the shape *an estimate answering a different question than
the one asked is worse than no estimate* (`-ncffn` dropped from the fit argv,
total-vs-free VRAM, `-md` unparseable by the estimator). Two are handled by one
mechanism, `fit_blind_flags`; whether the third belongs in it is a design
question worth settling before a fourth arrives.
