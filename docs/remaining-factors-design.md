# The remaining llama.cpp knobs — design & work log

Plan for the ~dozen perf-relevant flags [`flag-coverage.md`](flag-coverage.md)
found unreachable. Unlike the multi-GPU, draft-model and workload-shape surfaces,
almost none of these need a new mechanism — the design work is deciding **which
ones the tool should sweep on its own**, which is a different question from which
ones it should be able to express.

## Origin

[`flag-coverage.md`](flag-coverage.md). `--factor` rejects names absent from
`FACTORS`, so registry coverage *is* user reachability: a knob we have not
registered is one nobody can ask about, whatever they know about llama.cpp.

## The decision that shapes everything: reachable ≠ swept

Two failure modes pull in opposite directions.

Leave a knob out and a user who knows it matters on their hardware has no way in —
the gap this document exists to close. Add it to the default design and every
sweep pays for a column that, on most machines, does nothing: an orthogonal array
spends rows distinguishing configurations that are identical, and the main effect
comes back as noise around zero, which reads as "measured, doesn't matter" rather
than "wasn't worth measuring here".

**R1 — register every knob; auto-sweep almost none.** A registry entry costs
nothing and makes `--factor swa_full=0,1` work. Entry into `build_factors` is a
separate, higher bar: the knob must plausibly matter on hardware the tool can
detect. This is already how `ncmoe` (MoE models), `mtp` (NextN head) and `numa`
(multi-node) behave; the difference is that those gate on something detectable,
and most of this batch does not.

The honest position for the rest is that the user knows something we cannot
detect, and the tool's job is to not be in their way.

## Sense mismatches are the real trap here

Several of these flags are spelled in opposite directions on the two drivers, and
getting it wrong measures the opposite of what the row claims — silently, since
both spellings are valid.

`llama-bench` takes `-nopo <0|1>` (*no*-op-offload, negative sense).
`llama-server` takes `--op-offload` / `--no-op-offload` (positive sense, default
on). A factor named `op_offload` would need inverting on exactly one driver.

**R2 — name the factor after the negative spelling where llama.cpp does.**
`no_op_offload=1` emits `-nopo 1` on bench and `--no-op-offload` on server: same
level, same meaning, no inversion anywhere. Naming it for our own convenience
would buy a nicer factor name and an inversion bug.

**R3 — a `bool` factor needs an off-spelling when the flag defaults to on.** The
current `bool` kind emits a bare flag when enabled and *nothing* when disabled,
which is correct for `-nkvo` (default off) and wrong for `--repack` and
`--op-offload` (default on), where emitting nothing means leaving it enabled. An
optional `off_flag` on the spec covers this: level 0 emits `--no-repack` rather
than silently keeping the default. Factors without `off_flag` are unchanged.

## The knobs, and what each gets

| factor | bench | server | auto-swept? | why |
|---|---|---|---|---|
| `load_mode` | `-lm` | `--load-mode` | **no** (candidate) | Levels `mmap`/`mlock`/`mmap+mlock`/`dio`/`none`. The one most likely to pay on partial offload, where the CPU-resident half is what pages. Needs the fixed-emission interlock (R4) |
| `no_op_offload` | `-nopo` | `--no-op-offload` | no | Partial-offload lever; nothing detectable says whether it matters |
| `no_host` | `--no-host` | `--no-host` | no | Pairs with repack; both steer buffer-type selection |
| `repack` | — | `--repack` | no | Only pays on CPU-resident tensors, i.e. what `-ngl` leaves behind |
| `swa_full` | — | `--swa-full` | no | Meaningless off sliding-window models. Would be auto-swept if we detected SWA — see open question 1 |
| `backend_sampling` | — | `-bs` | no | Experimental upstream; a fixed per-token tax, so it matters most where tg is already high |
| `prio`, `prio_batch` | `--prio` / — | `--prio`, `--prio-batch` | no | Real on a contended box, which is the normal case — but the effect is about the *machine*, not the model |
| `cpu_mask_batch`, `cpu_range_batch`, `cpu_strict_batch`, `poll_batch` | — | `-Cb`, `-Crb`, `--cpu-strict-batch`, `--poll-batch` | no | The batch-phase twins of affinity knobs we already sweep. We already sweep `threads_batch`, so we already believe the batch phase deserves separate tuning |
| `ctx_checkpoints`, `checkpoint_min_step` | — | `-ctxcp`, `-cms` | no | Recompute cost on context shift; bites at the high-depth end where throughput already sags |

Deliberately **not** in this batch: `kv_unified` belongs to
[`concurrency-kv-design.md`](concurrency-kv-design.md) because it is constrained
by `--parallel`, and `-sps` belongs to
[`workload-shape-design.md`](workload-shape-design.md) because it is unmeasurable
until requests differ. Adding either here would produce a column that cannot move.

**R4 — a factor and a fixed emission must not both fire.** `load_mode` is now
emitted fixed by `load_mode_args`. If it also becomes a factor, every row gets the
flag twice and llama.cpp takes the last one — the row would be valid and would
measure whatever came last, not what the column says. The existing pattern covers
this (`if "fa" not in f`, `if "batch_ratio" not in f`), and `load_mode` joins it.

## Invariants

- **F1 — every registered knob is reachable via `--factor`.** That is the whole
  point; a registry entry that `--factor` rejects is a bug.
- **F2 — a level's emitted flags mean the same thing on both drivers.** No factor
  is inverted on one driver and not the other (R2).
- **F3 — a disabled boolean is emitted, not implied,** wherever llama.cpp's
  default is enabled (R3).
- **F4 — no flag is emitted twice in one command** (R4).

## Open questions

1. Can we detect a sliding-window model from GGUF metadata? `llama_model_n_swa`
   knows at runtime, and the server logs it, but `build_factors` reads metadata
   only. If it is in the header, `swa_full` becomes auto-swept on exactly the
   models where it means something — the `ncmoe` pattern.
2. Should `load_mode` be auto-swept when the model does not fit in VRAM? That is
   detectable (model size vs `detect_vram_mib`), and it is precisely the case
   where paging behaviour decides throughput. Tempting, and it would be the first
   knob gated on a *fit* prediction rather than a capability.
3. Do `ctx_checkpoints` and `checkpoint_min_step` interact strongly enough with
   `n_depth` to need the constrained-factor treatment? Both change recompute cost
   on context shift, and `n_depth` decides how often that happens.

## Checklist

- [ ] `off_flag` support in the `bool` branch of `factor_flags` (R3)
- [ ] Registry entries for all knobs in the table, correct per-driver spellings
- [ ] `load_mode` factor + the fixed-emission interlock (R4)
- [ ] `--selftest`: each new factor emits the expected flags on each driver it
      supports; no factor is inverted between drivers (F2); a disabled default-on
      boolean emits its off-spelling (F3); sweeping `load_mode` emits exactly one
      `--load-mode` (F4) — all pure functions, no GPU
- [ ] Open question 1: check GGUF metadata for an SWA/window field
