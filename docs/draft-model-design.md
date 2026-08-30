# Draft-model tuning — design & work log

Concrete plan for accepting a draft model and sweeping the draft-side knobs it
unlocks. The *principles* — why gated parameters cannot ride a flat orthogonal
array, and the staged mechanism that fixes it — live in
[`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md); this file is the
draft-specific factor model and checklist, the same split
[`ngram-design.md`](ngram-design.md) and
[`multi-gpu-design.md`](multi-gpu-design.md) use.

## Origin

[`field-reports.md`](field-reports.md) F2. Two independently tuned vLLM setups
spend most of their effort on the drafter — where its weights live, how they are
quantized, how much VRAM its KV cache gets — because on a 24 GB card the draft
model competes directly with the target model for exactly the resource `-ngl` is
already the biggest lever over.

The first pass at that finding proposed adding `--spec-draft-ngl` and stopped
when it turned out to be inert. That was the wrong conclusion from a correct
observation, and the reason is worth recording: it was scoped to the development
machine, which has no draft model. This tool is not run only here.

## Defect

`llama-optimize` never emits `-md`, and there is no input by which a user could
ask it to. So for anyone who owns a draft model, the entire draft-side surface is
unaskable — not badly tuned, not defaulted, simply outside the space the tool can
describe. `--factor` is no workaround: it rejects names absent from the registry.

The blast radius is larger than one flag. `has_dft()` is just "was a `-md` path
given" (`common/common.h:382`), and it gates:

- **`--spec-draft-ngl`** and the rest of the placement mirror (table in F2) —
  the whole `--spec-draft-*` family, `common/arg.cpp` ~3927-4165
- **`--spec-type draft-simple | draft-eagle3 | draft-dflash | draft-dspark`**
  (`common/speculative.cpp:34-38`) — every draft-*model* speculation strategy.
  Our `mtp` factor covers `draft-mtp`, which is the one variant that needs no
  draft model, so the registry currently describes the exception and omits the rule.

## Two models in VRAM is the thing that is actually new

Every existing factor tunes one model. A draft model makes the box hold two, and
that is where the design work is — not in the flag plumbing.

**D1 — `-md` is an input, not a factor.** It takes a filesystem path supplied by
the user, exactly like `--model`. Nothing derives it and nothing sweeps it; a
sweep either has a draft model or does not. This is the same shape as `--ngram`
(a capability switch) rather than `ngl` (a level set).

**D2 — every draft factor is conditional on that input.** With no `-md`, every
draft-side column would be inert — each level producing an identical run, and a
null main effect reading as "draft placement doesn't matter" rather than "we
never tested it". This is `active_when` (`CONDITIONAL-FACTORS.md`) with the gate
being a config input rather than another factor's level, which the current
`is_active` signature does not express: it resolves gates against the assignment
only. Either the gate generalizes to "assignment or config", or `build_factors`
omits the draft factors entirely when no draft model is set. **The latter is
preferred** — it matches how `ncmoe`, `mtp` and `ngram` already appear only when
the model and flags make them meaningful, and it keeps `is_active` single-purpose.

**D3 — REVISED TWICE. The pruner cannot ask the estimator about a second model,
so it prices one itself.** The first revision concluded that pruning should
simply stand down on draft rows. That was too quick: it made the tool honest
without making it useful, and "we cannot get a perfect number" is not a reason to
use no number. Standing down admits *every* doomed config; a conservative
estimate admits only some.

What remains true is that `llama-fit-params` rejects `-md` outright:

```
$ llama-fit-params -m model.gguf --fit-print on -md draft.gguf
error: invalid argument: -md
```

It accepts `-ctkd`/`-ctvd`, so it knows about draft *cache types*, but there is
no way to tell it a second model exists. Estimating anyway would report the
target's footprint alone while the machine holds two — confidently wrong, in the
direction that approves configurations which then abort.

So `resident_extra_mib` adds what we can measure without it: **weights on disk
are a hard lower bound on weights in VRAM.** The draft model's GGUF size, scaled
by `-ngld`'s share of its layers, plus the projector's size when
`--mmproj-offload` leaves it on the GPU. Both are per-row, since placement is a
factor, so the figure reaches the fit cache key as well.

A lower bound is the only safe direction, and the reason is asymmetric. The
estimate is compared against a ceiling, so **understating** it prunes fewer rows:
some configs that will not fit get run anyway, costing time. **Overstating** it
deletes configs that would have fit, costing information you cannot recover — the
failure this project keeps rediscovering (issues #5, #7, and the `-ncffn` gate).
So compute buffers and the draft model's own KV cache are deliberately not
modelled, and an unreadable draft geometry prices as zero rather than as "all of
it".

This does not make the estimate correct, and it is not meant to. It makes it
*directionally useful and never harmful*, which is the most that can be had while
the estimator cannot be told these artifacts exist. If llama.cpp ever teaches
`llama-fit-params` about `-md` and `--mmproj`, this term should be deleted rather
than added to.

The original reasoning, still right about the stakes:

> **D3 — the OOM pruner must account for both models.** `predict_fits` fits a VRAM
footprint from `ngl`/`kv_type`/context over one model. With a draft model
resident the free VRAM for the target shrinks by the draft's own offloaded
layers and KV cache, and `-ngld` moves that number *within the sweep*. A pruner
that ignores it will predict fits that OOM — the failure mode the pruner exists
to prevent, and the one it must stay conservative about (ROADMAP item 2). This is
the direct analogue of `multi-gpu-design.md`'s N2.

**D4 — the mirror is close to exact, but not exact.** Most draft flags take the
same values as their target twin, which is what makes a generated mirror
tempting. Two do not: `--spec-draft-poll` is `<0|1>` where `--poll` is `N`
(0-100), and `-ngld` accepts `auto`/`all` alongside an integer. A mirror
mechanism that assumes shared level sets will silently emit values llama.cpp
rejects, so level sets stay per-factor even where the flag mapping is mechanical.

**D5 — the draft model's size decides whether any of this is a question.** A
0.5B draft on a 32 GB card fits whole and `-ngld` has one sensible level; the
same draft against a target that already spills to RAM is a genuine trade. Level
generation must read the draft GGUF's layer count the way `ngl_levels` reads the
target's, not assume a range.

## Invariants

- **DM1 — No draft column without a draft model.** If `cfg.draft_model` is unset,
  no draft factor appears in the design, the CSV, or the emitted command.
- **DM2 — A draft-model spec type is never emitted without `-md`.**
  `--spec-type draft-simple` with no draft model is a run that cannot speculate;
  it must be unconstructible, not merely unlikely — the `spec_off` telemetry
  (F1) would flag it after the fact, but by then a row has been wasted.
- **DM3 — VRAM predictions account for every resident artifact, at a lower
  bound.** Any consumer of the fitted footprint sees the draft model and the
  projector, priced conservatively from their file sizes and per-row placement,
  or the prediction is not made at all. Never a footprint that silently omits
  one.
- **DM4 — Draft factors are staged, not flat.** A dozen new columns in one array
  is the inflation `CONDITIONAL-FACTORS.md` was written to prevent: screen the
  spec type at default knobs, then tune the winner's placement.

## Cost, and why staging is not optional

The mirror is ~10 factors. Added flat to an L25 design they do not fit; added as
a second block they roughly double the run count, and every one of those runs
loads two models. The ngram precedent applies directly — that surface went from 9
knobs to 3 collapsed ones and L125 back to L25 by screening the gate first.

Sensible first cut, cheapest first:

1. **Screen** `--spec-type` over `{draft-simple, draft-mtp (if available), off}`
   at default knobs. Many users' answer stops here.
2. **Tune placement** for the winner: `-ngld`, then `-ctkd`/`-ctvd`.
3. **Everything else** (`-otd`, `-ncmoed`, draft CPU/affinity knobs) only on
   request — they are the long tail and each one costs a column.

## Open questions

0. **Does `-md` with an already-embedded NextN head differ from omitting it?**
   The sharpest question from issue #12, and unanswerable without hardware. A
   model can carry its MTP head *and* be given a separate head file: verified on
   `Qwen3.8-27B-UD-IQ4_XS`, which reports `nextn_predict_layers = 1` and contains
   all four `blk.64.nextn.*` tensors, while Unsloth also publishes a standalone
   18-tensor `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. `has_dft()` is just "was a `-md`
   path given", so the two take different branches of
   `common_speculative_init_result`:

   - **route 1** — embedded head, no `-md`: drafts from the already-loaded target
   - **route 2** — `-md`: loads a second model, at its own independent quant
     (a user in issue #12 pairs a `Q4_0` head with an `IQ4_XS` target)

   Different VRAM footprints, plausibly different acceptance and throughput.
   Both are now expressible, so this is a measurement someone can simply run:
   the same sweep with and without `--draft-model`, on a model whose head is
   embedded. Needs a GPU and an MTP-capable model; asked on issue #12.

1. Should `--draft-model` accept `auto`, resolving a conventional sibling of the
   target GGUF? Convenient, but it would make a sweep's factor set depend on a
   filesystem guess — probably not worth the surprise.
2. Is `draft-eagle3`/`draft-dflash`/`draft-dspark` worth screening, or gated on a
   `--help` probe like other capability checks? These need specific draft
   artifacts, so a level that fails for most users is mostly wasted rows.
   `draft-dflash` is the llama.cpp analogue of the DFlash2 drafter in
   [`field-reports.md`](field-reports.md), so it is the one with outside evidence
   behind it.
3. Does the draft model deserve its own KV quality floor? `--min-kv` reasons
   about output quality from the target's KV cache; a lossy *draft* KV degrades
   acceptance rate, not correctness — the verifier still checks every token. That
   makes `-ctkd` a pure throughput knob and arguably exempt from the floor, which
   would be the first place quality and speed genuinely decouple here.

## Checklist

- [ ] `--draft-model PATH` input on `Config`, plumbed to `-md` in
      `build_server_args` (server only — llama-bench has no speculation)
- [ ] `build_factors` emits draft factors only when the input is set (DM1)
- [ ] `spec_type` factor extended with the draft-model variants, gated so DM2
      holds by construction
- [ ] `-ngld` registry entry + level generation from the *draft* model's layer
      count (D5)
- [ ] `-ctkd`/`-ctvd` registry entries; decide open question 3 first
- [x] `predict_fits` prices the draft model and the projector from their file
      sizes, scaled by per-row placement, as a conservative lower bound the
      estimator cannot supply (D3 revised twice / DM3)
- [x] `--draft-model` input, `-md` emission, `-ngld` + `-ctkd`/`-ctvd` as factors
      present only when a draft model is given (D1/D2/DM1)
- [x] `-ngld` levels generated from the DRAFT model's own layer count (D5)
- [x] Draft KV deliberately exempt from `--min-kv`: that floor protects output
      quality, and the drafter emits no output — a token drafted from a degraded
      draft cache is verified by the target, then accepted or discarded, so
      quantising it costs acceptance rate (speed), which is what is measured
- [ ] Stage planner: screen spec type, then tune placement (DM4)
- [ ] `--selftest` coverage: flag emission, DM1/DM2 as property checks over the
      factor grid, and level generation against a captured draft-model metadata
      block — all stdlib-only, no GPU and no draft model needed

## Testing without a draft model

Development here has no draft GGUF, which is the same shape as the multi-GPU
problem: the paths that matter are the ones that cannot be exercised locally, so
they must be testable without the hardware.

1. **Property tests on the factor grid** (free, in `--selftest`). DM1 and DM2 are
   statements about which columns exist and which combinations are constructible.
   Both are checkable over the generated grid with no binary — the same technique
   the issue #8 check uses, and the reason that check is trustworthy.
2. **A tiny draft model** (e.g. a 0.5B against any target) validates flag
   emission and that llama.cpp accepts the combinations. Cheap, and enough to
   catch the D4 value-range traps.
3. **A real draft pairing on a constrained card** is the only way to test what
   this is for — the VRAM trade between two models, which by construction cannot
   reproduce where the draft model fits trivially.

Tier 2 is worth doing before asking anyone to run tier 3: `draft_acc` (F1) already
reports whether speculation is paying, so a small pairing confirms the whole
telemetry loop end to end even when the measured throughput is uninteresting.
