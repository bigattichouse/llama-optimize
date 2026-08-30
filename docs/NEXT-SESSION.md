# Handoff — open work as of 2026-08-30

Working state at the end of the 2026-08-30 session. For what order to do the
remaining work in and why, see [`PLAN.md`](PLAN.md). Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything below is committed and pushed to `main` (through `1c69ad5`), the
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

## Do these first

### 1. Issue #11 — `--profile agents` false `IMPLAUSIBLE` (blocked on reporter)

**Parked at the maintainer's request; do not spend time until data arrives.**
Asked for the results CSV, the exact command and the model's native context.

Ruled out already, so do not redo them: prefix reuse (`--profile agents` selects
the *bench* driver, where it never applies — the original hypothesis was wrong)
and bench output parsing (pp rows carry `n_gen=0`, tg rows `n_prompt=0`,
unaffected by `-d`).

One real finding fell out: the guard's own comment claims "the honest ratio runs
10-100x in prefill's favour", but measured on CPU (gemma-3-270m, `-p 8192
-n 256`) it is 11.7x at depth 0, **7.5x at 8192** and 7.3x at 32768. It flattens
rather than inverting, so the check never trips — but the margin on the
deep-context profile is thinner than the constant was reasoned from. Worth
revisiting `TG_OVER_PP_LIMIT` regardless of what the reporter says.

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

### 3. Multi-GPU (issue #5) — the reporter's own numbers moved this

`--list-devices` on their box: 6672 MiB free of 24117 on the 3090, 9837 of 11909
on the 3060. **16.5 GB free against 36 GB total**, which likely explains their
"35B kept getting offloaded" report better than the single-device VRAM bug did.
Asked them to re-run and report the new `VRAM free` header line, since it changes
what a multi-GPU split should be aiming at. `-sm`/`-ts` remain unimplemented;
[`multi-gpu-design.md`](multi-gpu-design.md) is still the plan of record, and the
device-order trap still needs their hardware.

## Then

- **Draft-model staging (F2 continued).** The staged screen from
  [`draft-model-design.md`](draft-model-design.md) — screen `--spec-type`, then
  tune the winner's placement — is not built; draft factors currently ride the
  flat array. Also `-ncmoed` and the `draft-simple`/`draft-eagle3`/`draft-dflash`
  spec types.
- **Route 1 vs route 2 for MTP (needs GPU, ideally not ours).** Whether passing
  `-md` with an already-embedded NextN head measurably differs from omitting it.
  Now expressible; asked on issue #12 for anyone with the setup.
- **`--mmproj` (issue #12).** Probably a measurement-*validity* question rather
  than only coverage: a resident projector occupies VRAM whether or not image
  traffic arrives, shifting the boundary the pruner and context probe reason
  about. Currently filed under absent workloads in
  [`flag-coverage.md`](flag-coverage.md); may belong under validity.
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
