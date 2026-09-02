# Handoff — open work as of 2026-09-02

Working state at the end of the 2026-09-01 session. For what order to do the
remaining work in and why, see [`PLAN.md`](PLAN.md). Unlike the other files in
`docs/`, this one is **transient**: it records what is in flight and what to pick
up next, and should be pruned as items land. Durable reasoning belongs in the
design docs it points at.

Everything is committed and pushed to `main` (through `877fbf3`), the working
tree is clean, `--selftest` passes, and no GPU work is running.

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

## What landed on 2026-09-02

Four defects, all of the same shape as the ones above: **a guard that reported
nothing when it failed, which is indistinguishable from a guard that works.**

**The thermal settle had three silent no-op paths.** Found by reading `temp_c` on
a comparison that had the settle *enabled*: 44 °C against 87 °C, a 12% apparent
effect, no warning. The plateau rule gave up after one poll that cooled <0.5 °C
(at a 3 s poll that is 10 °C/min, a rate a hot card passes straight through); the
120 s cap was shorter than an MI50 takes to shed a run; and the "idle baseline"
was a single sample, so a hot card produced `idle baseline 99°C — settle to
≤104°C`, a target nothing can exceed. A settle that gives up now names the
temperature it gave up at.

**Warm is now the default** (`--thermal-mode warm`). Inference runs on a hot
card, so a number off an idle one is a *burst* figure. Each config is preheated
with its own workload to its own steady state. The distinction that makes it safe:
warm means "at its own steady state", not "still hot from the last run" — the
latter is the confound that produced 44-vs-87. `--thermal-mode idle` keeps the
old behaviour for bursty agent workloads. Full reasoning in
[`measurement-validity.md`](measurement-validity.md).

**A `--factor` pin was overridden by a probe.** A run pinned to `ngl=99` swept
48/53/59/64, because `probe_loadable_ngl` rebuilt candidates from the
un-collapsed span. A probe now never runs on a factor the user set by hand.

**`--draft-model` heads recorded no telemetry.** `spec_cols_wanted` knew about
NextN heads and `--ngram` but not a draft model on the command line, so an EAGLE3
run on a model with `n_nextn=0` wrote every row with `draft_acc`, `draft_cov` and
`spec_off` **absent from the CSV** — no way to see whether the head drafted a
single token. The `spec_off` guard was blind in the same place.

### `draft-eagle3` is verified end to end

`Qwen3.6-27B-Q5_K_M` + `wimmmm/Ex0bit-Qwen3.6-27B-PRISM-EAGLE3` (Q8_0), MI50 32GB
/ ROCm, llama.cpp `6c84c7d5d`, `--task code`, depth 4096, ngl 99, q8_0 KV:

| spec_type | tg t/s | temp | draft_acc | draft_cov |
|---|---|---|---|---|
| none | 9.09 | 59 °C | — | — |
| eagle3 | 9.06 | 64 °C | 0.47 | 0.58 |

**The head works and buys nothing *here*.** 47% acceptance means speculation
genuinely ran; the overhead cancels the gain. Do not restate this as "EAGLE3 does
not help" — it is one head (trained for the PRISM finetune) against a *base*
target on one card, and a matched head could plausibly do better. An earlier
reading of **+12%** was the thermal confound above.

**Do not retry the specdrift head.** `Dogacel/specdrift-qwen3.6-27b-eagle3` is at
`../model/specdrift-qwen3.6-27b-eagle3` and is **not convertible**: a different
EAGLE3 flavour carrying `fcs.0/1/2.weight` (one fc per aux hidden-state layer) and
a `t2d` map for a reduced 32000-token draft vocab, where llama.cpp expects a
single `fc` and only `d2t`. `convert_hf_to_gguf.py` stops at `Can not map tensor
'fcs.0.weight'`. Dimensions match fine; the topology does not. Not a tool bug.

### The `#18` dead band was reported with the wrong cause

Corrected in [`sweep-cost-design.md`](sweep-cost-design.md). Two errors:

- The `resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta
  Net (chunked) ...` warning was named as the cause. **It is not** — the identical
  two lines, `set to disabled` and all, appear in launches that load fine. The
  crash is later, between `threadpool init` and slot init.
- The map is the **bare** map. On `Qwen3.6-27B-Q5_K_M`, `-ngl 53` and `-ngl 5`
  segfault bare and **load, generating correctly**, under `--spec-type
  draft-eagle3` or `draft-dflash` — the two heads that reuse the target's hidden
  states. `draft-simple`, `draft-mtp` and four `ngram-*` types all still die, as
  does every other flag tried.

So `probe_loadable_ngl` no longer trusts one assignment: a level is dead only if
it fails under every level of `LOAD_SENSITIVE_FACTORS`, and levels that load
conditionally are reported as such.

## Do these first

### 1. `tg=0.00` at depth 16384 — narrowed to the model, root cause still open

Both deep rows of the `fa` sweep did it: `pp` fine (88.7 / 98.0), `secs` 639–689,
decode never produced a number.

**The reporting half is fixed and tested.** That shape now sets `no_result`
("every rep that survived reported 0 t/s") and the row is no longer recorded `OK`
— `_zero_rate` in `--selftest` covers it. So it can no longer poison a main effect
silently.

**What it is not.** The per-config deadline was the leading suspect and is ruled
out: `secs` 639 against a 1200 s budget, `err_rate=0.0`, `rejected_reps=0` — all
three reps returned successfully and reported zero. Nor is it the tool's deep
prompt construction: the same sweep shape at `n_depth=16384` on
`gemma-3-270m-it-Q8_0`, CPU-only, measured **53.9 tg t/s** (`scratchpad/deep0.csv`).

**What is left.** It is specific to `Qwen3.6-35B-A3B` (`qwen35moe`, recurrent) at
depth on ROCm. `cache_hit=0.0` on those rows says every rep re-prefilled all
17060 tokens, which matches the known recurrent reuse loss past a depth boundary
— but that explains the 639 s, not the zero. Needs one GPU repro: a single deep
request against that model, reading `predicted_n` from the server's own timings.
If it is 0, the model emitted EOS immediately and the tool is reporting honestly;
if it is non-zero, the rate is being lost between the server and the row.

### 2. ~~Validate `--thermal-mode warm`~~ — done, and it found a fourth defect

Validated 2026-09-02 on `Qwen3.6-27B-Q5_K_M`, MI50 32GB / ROCm:

```
cap warnings: 0
  spec_type=none    tg=8.78  pp=139.9  temp=94   secs=423
  spec_type=eagle3  tg=9.00  pp= 95.8  temp=94   secs=432
  temp spread: 0 °C
```

**The mechanism passed.** Zero preheat-cap warnings, so the rolling-window
plateau rule detects the measured 99↔100 °C oscillation on real hardware; row
time fell 617 s → ~430 s because the preheat stops at the plateau instead of
running to its cap.

**The first attempt failed on its own instrument**, which is worth keeping. It
recorded 45 °C and 89 °C — a 44 °C spread — for two rows that both measured at
the plateau. `temp0` was sampled before `measure_in_session`, and that function
now contains the preheat, so in warm mode `temp_c` reported what the *previous*
run left behind: exactly what the preheat makes irrelevant. The one column whose
job is checking comparability was manufacturing a confound that was not there.
Now sampled after the preheat, immediately before the first measured rep.

That is the fourth defect of one shape in this session: **a guard that reports
nothing when it fails is indistinguishable from a guard that works.** Warm mode
exposed it; the cold-settle path would have hidden it indefinitely, because
without a preheat the two sampling points agree.

Note the sustained numbers sit slightly below the burst ones measured earlier at
59–64 °C (9.09 / 9.06) — which is the point of the mode, not a discrepancy.

### 3. Re-measure `fa` properly (#20)

**The earlier "flash attention is 33% slower" finding was retracted** — it was a
44 °C row against a 91 °C row with `--no-thermal-wait` on. Re-run hot and
randomised it was 1.03x. See `constants-audit.md` C-A.

The question is therefore still open, and doing it right costs what cutting the
corner saved: warm mode (now the default), more reps, and read `temp_c` before
drawing anything — the rows must land within a few degrees of each other. The `--factor fa=0,1` footgun is already fixed (quantized `kv_type`
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
- **A temperature gap will hand you an artifact and call it a finding, and
  `--no-thermal-wait` is not the only way in.** A cold row against a hot one
  looked like a 33% flash-attention effect; a 43 °C gap made an EAGLE3 head look
  like +12%. The second happened with the settle *enabled* — it had three silent
  no-op paths. `temp_c` is in every row; read it before believing a comparison,
  including one taken with the guards on.
- **Metadata declares an architecture, not a file's contents.** Both
  `Qwen3.8-27B-UD-IQ4_XS` and the standalone `mtp-*.gguf` report
  `nextn_predict_layers = 1`; only the tensor list distinguishes them. Checking
  a truncated tensor listing is how issue #12's first diagnosis went wrong.
