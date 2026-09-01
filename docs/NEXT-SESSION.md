# Handoff — open work as of 2026-09-01 (evening)

Working state at the end of the 2026-09-01 session. For what order to do the
remaining work in and why, see [`PLAN.md`](PLAN.md). Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything is committed and pushed to `main` (through `eb6ea85`, 41 commits this
session), the working tree is clean, `--selftest` passes, and no GPU work is
running.

**Six issues closed: #11, #14, #15, #16, #18, #19.** Five open: #3, #5, #17, #20,
#21.

## What landed

Almost all of it came out of one field report (#11) and the questions that fell
out of chasing it. Detail is in the CHANGELOG; the short version:

**Measurement validity.** The wall-clock check credits prefill from `prompt_ms`;
each rep is screened against its own clock so one broken counter costs a rep
rather than a configuration; chars-per-token is measured by a probe before the
first prompt is sized, not after; `prompt_tok`, `cache_hit`, `rejected_reps` record
what was *delivered* against what was asked for.

**Things that were assumed and are now measured.** `swa_full` swept on SWA models;
`spec_type` swept when more than one speculative head exists; `repack`,
`no_op_offload`, `no_host` swept instead of inheriting llama.cpp's defaults; the
`ngl` grid narrowed by `llama-fit-params` and, on recurrent models, by a launch
probe that asks *this box* which levels load.

**Bugs found by asking rather than assuming.** The OOM pruner priced the opposite
KV placement from the one each row would run (~6 GB error, both directions). A
supplied `--draft-model` was loaded and then ignored on MTP models. A *plain*
draft model never speculated at all — loaded, charged to VRAM, silent. An EAGLE3
head would have been run as `draft-simple`. The pasted command omitted `-md`
entirely, so it could not reproduce the row above it.

**Crashes are read as data.** A level that failed in *every* one of its rows is
reported as out of bounds and dropped from the next `--iterate` pass.

**The report says what it is conditioned on:** quant, card, backend *and backend
version*, capacity, cores, driver, profile.

## Do these first

### 1. `tg=0.00` with status `OK` at depth 16384 — free, and the right shape

Both rows of the last `fa` sweep did it: `pp` measured fine (88.7 / 98.0),
`secs` 639–689, and decode never produced a number. `measured_ok` stops it
poisoning a pick, so nothing is *wrong* in the report — but something silently did
not measure, which is the defect class this whole session was about. Leading
suspect is the per-config deadline (`slow_budget_secs`) cutting the reps at depth,
which would mean deep-context rows quietly stop reporting decode. Investigable
from `scratchpad/fadepth.csv` and the log; no GPU needed.

### 2. Verify `draft-eagle3` end to end — waiting on a download

`--factor spec_type=draft-eagle3` and the draft-architecture mapping are built and
unit-tested, but the path has never run. `wimmmm/Ex0bit-Qwen3.6-27B-PRISM-EAGLE3-GGUF`
(Q8_0, 1.77 GB) is the only *ready* GGUF EAGLE3 head for a Qwen3.6-27B-family
target; it was being downloaded and had not landed in `../gguf/` at session end.

Caveat when it runs: that head targets the **Ex0bit PRISM fine-tune**, not base
Qwen3.6-27B. Read weak acceptance as "wrong base model", and a layer-id abort as a
dimension mismatch — neither is a tool bug.

**Do not retry the specdrift heads.** `Dogacel/specdrift-qwen3.6-27b-eagle3` is
downloaded to `../model/specdrift-qwen3.6-27b-eagle3` and is **not convertible**:
it is a different EAGLE3 flavour carrying `fcs.0/1/2.weight` (one fc projection per
aux hidden-state layer) and a `t2d` map, where llama.cpp expects a single `fc` and
only `d2t`. `convert_hf_to_gguf.py` stops at `Can not map tensor 'fcs.0.weight'`.
Dimensions matched fine (5120, 24 heads, 4 KV, aux layers [3, 31, 59] against
64–65 blocks); the topology does not.

### 3. Re-measure `fa` properly (#20)

**The earlier "flash attention is 33% slower" finding was retracted** — it was a
44 °C row against a 91 °C row with `--no-thermal-wait` on. Re-run hot and
randomised it was 1.03x. See `constants-audit.md` C-A.

The question is therefore still open, and doing it right costs what cutting the
corner saved: thermal settle **on**, more reps, and read `temp_c` before drawing
anything. The `--factor fa=0,1` footgun is already fixed (quantized `kv_type`
levels are dropped automatically, since `-fa off -ctk q8_0` cannot create a
context). What remains is whether the pin itself is right, and the honest version
needs a constrained relation between `fa` and `kv_type` rather than a level
filter.

### 4. The fingerprint (#21)

Most of it already exists — `device_label`, `backend_version`, `model_hw`,
`cfg.hw` — so it is largely serialisation plus a schema version. JSON, because
`--selftest` promises stdlib-only. It is also the only route to answering the
questions this box cannot: every architecture-conditional behaviour found this
session was measured on one card, one backend, one quant.

## Blocked, and on what

- **#20** — the interesting half needs a backend where FA is absent or slow.
- **#17** — needs a dense-attention MoE. The hybrid MoE case dissolved when #18
  collapsed `ngl` to `[0, 99]`, so the two no longer compete there.
- **#5** — needs two non-identical GPUs.
- **#3** — needs the reporter's original CSV, which may not exist.

## Owed by others

Issue #11's reporter still owes DFlash2-vs-MTP numbers on their box and how they
invoke it, asked twice. Nothing depends on it. For the record, on this MI50 at 8k
depth: `draft-mtp` 13.96 tg t/s, `dflash` 12.51, `none` 7.22 — with DFlash2 ahead
on prefill (116 vs 94.7).

## Small things noticed and not filed

- `spec_draft_ngl`/`spec_draft_kv` are emitted on rows with no draft model. Inert,
  but it is the `gated_by` relation with a live set that static metadata cannot
  express (any draft architecture).
- `ctx_checkpoints`/`checkpoint_min_step` are registered and never swept. Measured
  not to restore prefix reuse (#15); that is one question of several.
- `load_mode` is pinned to `mmap` where llama.cpp's default is `auto`. The one
  knob that is genuinely *not* free to sweep: `none` re-reads the model per launch,
  so it slows every row rather than riding along.

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
- **Never `pkill -f` a pattern that matches your own command line.** It kills the
  shell running it (exit 144), and a broad `llama-server` pattern also kills other
  people's servers on a shared box. Kill by PID, or by a port-specific pattern.
- **`--no-thermal-wait` will hand you an artifact and call it a finding.** A cold
  first row against a hot second one looked like a 33% effect. `temp_c` is in every
  row; read it before believing a comparison.
- **Metadata declares an architecture, not a file's contents.** Both
  `Qwen3.8-27B-UD-IQ4_XS` and the standalone `mtp-*.gguf` report
  `nextn_predict_layers = 1`; only the tensor list distinguishes them. Checking
  a truncated tensor listing is how issue #12's first diagnosis went wrong.
