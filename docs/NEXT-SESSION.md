# Handoff — open work as of 2026-09-01

Working state at the end of the 2026-09-01 session. For what order to do the
remaining work in and why, see [`PLAN.md`](PLAN.md). Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything below is committed and pushed to `main` (through `63452e7`), the
working tree is clean, and `--selftest` passes.

**A live sweep did run this session**, but only on CPU: gemma-3-270m through the
server driver, two configs, to exercise the calibration probe and the new
delivered-depth column end to end. The GPU is still another project's. Anything
below marked "needs GPU" is genuinely unverified on hardware, not merely untested
by convention.

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

**Issue #11, in three commits (`b1170ed`, `032c153`, `63452e7`)** — the
wall-clock check, then three defects the reporter's data exposed underneath it:
per-rep rejection, prompts running at 2/3 of their stated depth, and four inert
speculative columns sizing the array. Section 1 below has the detail. Verified on
a CPU sweep as well as `--selftest`: delivered depth 1540/2565 tokens against
1536/2560 requested, `cache_hit` 0.90 against a requested 0.90.

**The defect class written down (`ff045dc`)** — five instances across four
subsystems, generalised into [`DESIGN.md`](DESIGN.md), with the testing practices
that catch it.

## Do these first

### 1. Issue #11 — closed

**Everything this issue produced has landed** (`b1170ed`, `032c153`, `63452e7`).
Four defects, only the first of which the issue was filed about:

- **I5 bounded a decode-only rate by a whole-request wall.** Fixed by crediting
  the prefill from the same response's `prompt_ms`, capped at 90% of the wall.
  See I5 in [`measurement-validity.md`](measurement-validity.md).
- **A rejection was wider than its fault.** The check compared the *mean* rate
  across reps to the *kindest* rep's ceiling, so one broken counter zeroed a
  config that had good measurements in it. Reps are now screened individually,
  survivors take the median, and `rejected_reps` records partial rejections.
- **Server prompts ran at ~2/3 of their stated depth.** Chars-per-token was
  measured from the warm request — after the prompts were built. A probe now runs
  at session open and the ratio holds for the session; `prompt_tok` records what
  the prompt became. [`constants-audit.md`](constants-audit.md) C-B.
- **`--factor mtp=0` still swept four inert speculative knobs** — 25 runs of one
  configuration, and a main-effects table crediting each knob with noise. New
  `gated_by` relation, [`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md).

**The two mysteries from the last handoff are both solved, and both guesses in
it were wrong.**

*The prompt cache not hitting* is not a context overrun. Their model is
`general.architecture = qwen35` with `qwen35.ssm.*` metadata — hybrid
SSM/attention — and llama.cpp cannot roll a recurrent state back to an arbitrary
prefix, so it forces full re-processing unless a context checkpoint covers it
(`tools/server/server-context.cpp`). Delivered reuse was 0% at depth 8192 too,
where `n_ctx` had ~6k tokens to spare, which is what rules the overrun theory
out. Prefix reuse is a property the model has or does not:
[`workload-shape-design.md`](workload-shape-design.md).

*`secs=113.4`, which "does not fit any story"*, fits the depth bug exactly. Their
prose tokenizes at **6.05 chars/token** (measured with `llama-tokenize`), so the
prompt was ~26,900 tokens, not 40,960. Four requests × (26.9k prefill at pp=1184
+ 256 tokens at tg=45.6) = **113.3 s** against the recorded 113.36. Their second
CSV agrees independently. No deadline cut anything short.

**The confirming run came back and settled both unknowns** (`mtp=0`,
`n_depth=8192`, on `63452e7`):

- `cache_hit=0.0` with speculation off, at a depth with `n_ctx` to spare — so the
  cache miss is the architecture, not the speculative path. Issue #15.
- `tg=15.98`, `rejected_reps=0`, no discard — so the 333,362 t/s counter follows
  MTP. A llama.cpp accounting problem in the speculative path, not ours; the
  per-rep screening handles it by dropping the rep rather than the row.
- `prompt_tok=16435` against 16,384 requested (**+0.3%**), where the old sizing
  would have built 10,832 (66%). The calibration fix is confirmed on their model,
  not just on gemma-3.
- The row reconciles to 101.8s against a recorded `secs` of 102.89 (1.1%), with
  a 0.3% spread across reps. Nothing unaccounted for.

Incidental, worth remembering: `mtp=1` measured 45.6 t/s at a *deeper* context
against 16.0 with MTP off. Different depths, so a floor rather than a number —
but MTP is earning its keep on this model.

**Spun out rather than left in this issue's tail**, so they do not die with it —
same treatment as #14:

- **Issue #15** — prefix reuse is unattainable on hybrid/recurrent and SWA
  models, and `--ctx-checkpoints` is not swept.
- **Issue #16** — MTP's tuning knobs share a flat array with the `mtp` gate when
  it is swept. `gated_by` fixed only the pinned case.

**Still only written down here:**

- **`TG_OVER_PP_LIMIT`.** The guard's comment claims "the honest ratio runs
  10-100x in prefill's favour"; measured on CPU (gemma-3-270m, `-p 8192 -n 256`)
  it is 11.7x at depth 0, **7.5x at 8192** and 7.3x at 32768. It flattens rather
  than inverting, so the check never trips — but the margin on the deep-context
  profile is thinner than the constant was reasoned from.

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
- **`pkill -f llama-optimize` kills your own shell.** The pattern matches the
  command line of the shell running it. Use a bracket (`llama-optimiz[e]`), or
  kill by PID.
- **Piping a long run into `tail` shows nothing until it exits.** stdout is block
  buffered through a pipe: run with `python3 -u` and redirect to a file.
- **Metadata declares an architecture, not a file's contents.** Both
  `Qwen3.8-27B-UD-IQ4_XS` and the standalone `mtp-*.gguf` report
  `nextn_predict_layers = 1`; only the tensor list distinguishes them. Checking
  a truncated tensor listing is how issue #12's first diagnosis went wrong.
