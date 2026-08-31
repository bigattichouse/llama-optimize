# Handoff — open work as of 2026-08-30

Working state at the end of the 2026-08-30 session. For what order to do the
remaining work in and why, see [`PLAN.md`](PLAN.md). Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything below is committed and pushed to `main` (through `e96a076`), the
working tree is clean, and `--selftest` passes.

**No live sweep was run this session.** The GPU on the dev box is in use by
another project — 174 MiB free of 32752 at last check — so every change is
verified by `--selftest`, in-process exercise with a stubbed `_help_cache`, and
CPU-only runs with the device hidden. Anything below marked "needs GPU" is
genuinely unverified on hardware, not merely untested by convention.

## What landed

**PR #10 / issue #9 — `-ncffn` dense FFN offload.** Merged in `e87f13a` with
Fathi Boudra's commit cherry-picked intact, plus a fix: the pruner gate dropped
`-ncffn` from the estimate when `llama-fit-params` predated the flag, which made
every level share one cache key and one verdict computed for the *un-offloaded*
config — silently deleting the whole factor on exactly the machines it exists
for. Superseded the same day by `ffn_place` (below).

**`ffn_place` (`2ed5873`)** — one dense placement column spanning `-ot` and
`-ncffn`, because they are different axes (*which tensor* vs *how many layers*)
and cannot be two OA columns: `-ncffn` appends to the same override vector `-ot`
writes to, so `ffn_cpu` swallows every `first_N` level. Added a build-time
invariant that an `emit` factor's levels must emit *distinct* arguments — it
immediately caught a level spelled `up_cpu` that missed the `OT_PATTERNS` key,
emitted nothing, and silently duplicated `none`.

**Free-VRAM preflight (`d422125`)** — the pruner compares against *total* VRAM;
`list_devices` already parsed free per device and nothing read it. On a shared
card that approves configs which abort in the allocator. Warns, never prunes.

**Sweep cost (`71367e7`, `e2f50c9`, `5810272`, `eea04db`)** — `--timeout` became
a per-config deadline (it was per-*request* on the server driver, allowing
`(1+reps)×`), `--min-tgs`/`--min-pps` abandon slow configs by arithmetic, and
`--levels` narrows every auto-generated factor together. A bare invocation on a
TTY now interviews the user. See [`sweep-cost-design.md`](sweep-cost-design.md).

**Draft model (`1c69ad5`)** — `--draft-model` input, `-ngld` and `-ctkd`/`-ctvd`
as factors present only with it. First increment of F2.

**`--mmproj` and artifact pricing (`1099475`)** — a projector is an input too,
and the pruner now prices both it and the draft model from their on-disk size
scaled by per-row placement, rather than standing down. `llama-fit-params`
rejects `-md` and `--mmproj`, so this is the only way those rows get pruned at
all. A lower bound, deliberately: understating wastes time, overstating deletes
configs that would have fit. Closed issue #13, and superseded the stand-down
answer given on #12 earlier the same day.

**The defect class written down (`ff045dc`)** — five instances across four
subsystems, generalised into [`DESIGN.md`](DESIGN.md), with the testing practices
that catch it.

## Do these first

### 1. Issue #11 — I5 fixed; one anomaly in the report is still unexplained

**The check was wrong and is now fixed** (see CHANGELOG *Fixed*, and I5 in
[`measurement-validity.md`](measurement-validity.md)). The reporter's second CSV
closes to its own arithmetic and needs no further data:

| | |
|---|---|
| no profile → `single`, `prefix_reuse=0.0`, `n_depth=32768` | `prompt_len` = 33,280 |
| `secs` 98.34 over warm + 3 reps | **24.6s per request** |
| reported ceiling 10.9 t/s | best rep wall = 256/10.9 = **23.5s** |
| honest decode at tg=43.1 | 256/43.1 = **5.9s** |
| residual | **17.6s** → 33,280 tok ≈ 1,890 t/s prefill |
| verdict | 43.1/10.9 = 3.95 → the "4x" they saw |

Every term reconciles: at reuse 0.0 each rep re-prefills the whole prompt *by
design*, and I5 was bounding a decode-only rate by a whole-request wall.

**Still open, and it is a different bug.** `--prefix-reuse 100` should make all
four prompts byte-identical (`prompt_battery`, the `n_shared >= n_chars` branch),
so the reps are pure decode off a full cache hit, wall ≈ 6s, ceiling ≈ tg, no
trip. The reporter says it tripped anyway. Their first CSV says the same thing
independently: at reuse 0.9 the rep should re-prefill ~4,096 tokens (~2s) for a
ceiling near 31 t/s, but the observed ceiling of 7.5 implies ~28s of prefill.
**The prompt cache is not hitting on their box.**

The I5 fix covers them anyway — `prompt_ms` reports the prefill that actually
happened, not the one we predicted — and the new `cache_hit` column plus its
warning will show whether this is real the moment they re-run. So it is no longer
blocking, but it should not be forgotten.

Leading suspect, unverified: `n_ctx` = `n_prompt + max_depth + n_gen + 256`
= 41,472 against a 40,960-token prompt *sized with the assumed*
`CHARS_PER_TOKEN = 4`. `self.cpt` is calibrated from the warm response but the
prompts are already generated by then (`measure()`), so on a single-config run
the calibration never applies. If their model tokenizes that prose below 4
chars/token, every request overruns the context and re-prefills. Worth deciding
whether the battery should be sized after the first response rather than before.

Also unexplained: their first run's `secs=113.4` does not fit any story — three
reps at the implied 34.1s leave under 11s for a warm request that must prefill
40,960 tokens. Either the deadline cut reps short or `predicted_n < 256`, and
neither is visible in the CSV. The `implausible` column now records the
breakdown, so a re-run will say which.

Unrelated finding from the earlier investigation, still worth acting on: the
guard's comment claims "the honest ratio runs 10-100x in prefill's favour", but
measured on CPU (gemma-3-270m, `-p 8192 -n 256`) it is 11.7x at depth 0, **7.5x
at 8192** and 7.3x at 32768. It flattens rather than inverting, so the check
never trips — but the margin on the deep-context profile is thinner than the
constant was reasoned from. Worth revisiting `TG_OVER_PP_LIMIT`.

### 2. The estimate/answer rule is now written down — apply it

Five instances across four subsystems, generalised into
[`DESIGN.md`](DESIGN.md#be-wary-wherever-an-estimate-can-answer-a-different-question-than-the-one-asked)
with the testing practices that actually catch it (property tests over the whole
input set, mutation-testing the guard, asserting against a stub rather than
against absence).

The refinement worth remembering, because the first answer was wrong: prefer a
**conservative estimate** to no estimate wherever a bound exists — standing down
admits every doomed config, a lower bound admits only some. Stand down only where
no bound exists.

Still open as a design question, not a bug: whether total-vs-free VRAM should
become a blind condition in `fit_blind_flags` rather than a warning.

### 3. Multi-GPU (issue #5) — the free-VRAM reading was a red herring

Last session read 6672 MiB free of 24117 on the reporter's 3090 and concluded
that 16.5 GB free against 36 GB total probably explained their "35B kept getting
offloaded" report. [They have since corrected
it](https://github.com/bigattichouse/llama-optimize/issues/5#issuecomment-5471917817):
a model was loaded when they captured that output, and the box idles at about
1 GiB used per card. Free is ~35 of 36 GB, and `headroom_warning` will never fire
there.

The single-device VRAM bug does not explain it either. `cfg.hw["vram"]` has
exactly one consumer — `predict_fits` — so if the pruner had been active *and*
misreading capacity, their forced `ngl=99` would have been pruned to `SKIP_PRED`
along with everything else. It ran. The low-`ngl` rows were simply the default
grid: `ngl_levels()` spans 0 → n_layers evenly and never consults VRAM, so a
40-layer model that fits gets `[0, 10, 20, 30, 40]` and spends four levels below
the answer. Filed as **issue #14**.

Two corrections landed in `57ff8d7`. `ts_levels()` is now specified against
per-device **total** VRAM — the checklist said *free*, reasoned entirely from
that one transient reading, which on their box would have derived the tensor
split from a number that was true for a minute. And `parse_fit_print`'s
single-pool sum is now written down as a `-ts` prerequisite rather than an
abstraction inside N2: it sums per-device footprints and compares against summed
total, so a config can clear 36 GB and still overflow the 3060.

`-sm`/`-ts` remain unimplemented; [`multi-gpu-design.md`](multi-gpu-design.md) is
still the plan of record, and the device-order trap still needs their hardware.

The transferable part: a device list is a reading of a moment, and the design
treated it as a property of a machine. Anything derived from `free` inherits
that, which is why free warns and total decides.

### 4. Issue #14 — `ngl` levels that know about VRAM

Step 7 in [`PLAN.md`](PLAN.md), which carries the reasoning. The complement of
the OOM pruner: that deletes rows above the fit, this stops spending the grid far
below it. Needs no GPU — level generation is a pure function and its tests are
`--selftest` material.

Two things to get right. What decides "fits" when there is no `llama-fit-params`
binary: only a crude file-size estimate exists, and biasing on one that errs
optimistic drops exactly the partial-offload rows a run would need, so it should
keep the even span unless it can be confident. And the fit test has to be taken
at the *deepest* context in the sweep, not depth 0, or the grid is optimistic
precisely where OOM is likeliest.

**The ordering here is contested, deliberately left open.** `PLAN.md` sequences
this ahead of multi-GPU on cost: it is cheap, needs no hardware, and helps every
user. The argument against is that it unblocks nobody — issue #5's reporter has
the only two-GPU box available to this project, and what they are waiting on is
`-sm`/`-ts`, whose prerequisite is the per-device `parse_fit_print` work in
step 8. Cheapest-first and unblock-the-tester-first genuinely disagree, and the
next session should pick knowingly rather than inherit the order by default.

## Then

- **Draft-model staging (F2 continued).** The staged screen from
  [`draft-model-design.md`](draft-model-design.md) — screen `--spec-type`, then
  tune the winner's placement — is not built; draft factors currently ride the
  flat array. Also `-ncmoed` and the `draft-simple`/`draft-eagle3`/`draft-dflash`
  spec types.
- **Route 1 vs route 2 for MTP (needs GPU, ideally not ours).** Whether passing
  `-md` with an already-embedded NextN head measurably differs from omitting it.
  Now expressible; asked on issue #12 for anyone with the setup.
- **Multimodal as a *workload*** — request shape, `--image-min-tokens`,
  `-mmdev`, and embedding/rerank serving. Still out of scope, and still a
  decision rather than an oversight ([`flag-coverage.md`](flag-coverage.md)).
  The *validity* half is done: `--mmproj` is an input and its footprint is
  priced (issue #13).
- **Report time saved by the floors.** Nothing says, after a sweep, how much
  `--min-tgs` actually bought. That number is the honest way to tune the advice.
- **Issue #3** — fix shipped and retroactive; the upstream cause is still
  unconfirmed. Leading hypothesis is now the tight-fit/total-VRAM interaction
  (`ngl=20 -ncmoe 40` on an 8 GB card). Offered to close as "cause not confirmed"
  if the reporter no longer has the CSV.

## Traps worth remembering

- **`-ngl 0` does not avoid the GPU.** llama.cpp op-offloads matmul to the GPU
  backend with no layers resident, so CPU-only rows still allocate VRAM and abort
  with the rest. To run on the CPU while the card is busy, hide the device:
  `HIP_VISIBLE_DEVICES=` / `CUDA_VISIBLE_DEVICES=`.
- **The CPU test model** is
  `/home/bigattichouse/workspace/gguf/gemma-3-270m-it-Q8_0.gguf`. The
  `../model/gemma-3-270M-it` directory is HF safetensors with no GGUF.
- **`llama-fit-params` and the driver are separate binaries with separate flag
  support**, and they *do* diverge inside one build tree — on this box
  fit-params has `-ncffn` while llama-bench does not.
- **Metadata declares an architecture, not a file's contents.** Both
  `Qwen3.8-27B-UD-IQ4_XS` and the standalone `mtp-*.gguf` report
  `nextn_predict_layers = 1`; only the tensor list distinguishes them. Checking
  a truncated tensor listing is how issue #12's first diagnosis went wrong.
