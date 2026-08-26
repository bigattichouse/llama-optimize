# Constants audit — what is universal, what is ours

Every hardcoded value in `llama-optimize.py`, classified by whether it is a
property of llama.cpp, a property of the user's machine that we should *derive*,
or a number that happens to be right on the box this tool was written on.

## Why this file exists

The proven failure mode is already in this repo's history. `batch` was swept at
`2048, 4096, 8192` — a perfectly good level set on the development machine, which
made an audited user's actual optimum (`-b 512 -ub 128`) unreachable **by
construction**. The tool would have reported something else as best, confidently,
with no sign anything was missing.

The fix that worked was not a better constant. It was replacing the constant with
a **relationship**: `batch_ratio` as a multiple of that row's own `ubatch`.
`b >= ub` is true on every machine; `b ∈ {2048, 4096, 8192}` was true on ours.

That is the test this audit applies to everything else: **is this value true
everywhere, derivable from something we can detect, or just true here?**

## The rule the tool should follow

**Turn a finding into an instrument, not a constant.** We measured 2.3x n-gram
inflation on one card. The robust response was `draft_acc` — every user now
measures their *own* acceptance, on their own hardware and traffic. A 2.3x
correction factor, or level sets tuned to make ngram look right on gfx906, would
have exported our hardware to everyone else's results.

Same shape elsewhere: `gpu_visibility` does not assume ROCm, it asks llama.cpp
what it sees and compares. `ngl_levels` does not guess, it reads the layer count.

## Universal — keep as constants

These are properties of llama.cpp or of arithmetic, not of any machine.

| constant | why it transfers |
|---|---|
| `MAX_PLAUSIBLE_TPS`, `TG_OVER_PP_LIMIT`, `WALL_CLOCK_MARGIN` | One-sided safety margins, deliberately generous. Being wrong in the permissive direction costs nothing; the defect they catch overshoots by four orders of magnitude ([`measurement-validity.md`](measurement-validity.md)) |
| `KV_QUALITY` ordering | The quality ranking of llama.cpp's KV quant formats |
| `NGRAM_MAP_LEVELS`, `NGRAM_MOD_LEVELS` | Taken from llama.cpp's own defaults and bounds (`common/common.h`) |
| `OT_PATTERNS` | llama.cpp tensor-name patterns |
| `GPU_ONLY_FACTORS` | A property of what the flags do |
| `_ARRAY_TABLES`, `RESULT_COLS`, `MORRIS_JSON_SCHEMA` | Internal structure |

## Already derived — the pattern to copy

| generator | derived from |
|---|---|
| `thread_levels` | physical / logical core count |
| `ngl_levels` | the model's layer count |
| `ncmoe_levels` | the model's layer count |
| `depth_levels` | the model's native context |
| `ts_levels` (planned) | per-device capacity from `--list-devices` ([`multi-gpu-design.md`](multi-gpu-design.md)) |

## Ours — the actual findings

Ranked by how much damage each can do.

### C-A. `FIXED_FA = 1` — pinned from *our* GPU

[`DESIGN.md`](DESIGN.md) justifies flash attention being fixed on with "reduces
KV-cache bandwidth and **helps gfx906**". That is literally the development
hardware, generalised to every user by pinning rather than sweeping.

It is not a free assumption: FA support and benefit vary by backend and model,
llama.cpp itself now defaults `-fa` to `auto` (it decides per build), and we
override that decision on every run. `fa` *is* a registered factor, so the fix is
a policy change rather than new machinery.

**Complication, and why this is not a one-line change:** `-fa` is a precondition
for quantized KV in llama.cpp, and `kv_type` is one of the most valuable factors
we sweep. Letting FA fall to `auto` on a backend that declines it would silently
invalidate every `q8_0` row. So the honest fix is to *probe* whether FA is
active and gate `kv_type` on the answer, not simply to stop pinning it. Needs a
GPU on more than one backend to validate.

### C-B. `CHARS_PER_TOKEN = 4` — model-poisoned, and load-bearing

Used to turn a requested token count into prompt text, so it sets the real
`n_depth` of every server measurement and the size of every battery prompt.
Tokenizer ratios vary widely by model and language — code and CJK are nowhere
near 4 chars/token — so on some models the sweep tests a materially different
depth than the column claims.

**Derivable, and cheaply:** llama.cpp reports `prompt_n` (actual tokens) in the
`timings` of every response. The first warm request per session yields a measured
chars-per-token for *this* model, and subsequent prompts can be sized with it.
This is the `draft_acc` pattern again — replace an assumption with the number the
server is already handing us. **Fixed below.**

### C-C. `THERMAL_BAND_C = 5.0`, `THERMAL_CAP_S = 120.0` — MI50 thermal policy

The settle-between-runs policy: wait until the GPU is within 5 °C of idle, giving
up after 120 s. Both numbers come from watching one card. A card that idles hotter,
throttles differently, or has no usable sensor gets a policy tuned for an MI50 —
either wasting two minutes per run or failing to settle at all.

`temp_c` is already recorded per row, so the raw material for deriving a band from
observed behaviour exists. Left as a finding: changing the settle policy changes
every measurement's thermal conditions, which needs live validation rather than
reasoning.

### C-D. `FIXED_BATCH = 2048` — the batch floor's surviving cousin

The value used for `-b` whenever `batch_ratio` is not swept. Same literal that
caused the original defect, in the one place it still applies. Lower risk (it is a
pin, not a level set, so it truncates nothing) but it is our number, and
`batch_ratio`'s own derivation shows how to remove it.

### C-E. `DEFAULT_UBATCH_LEVELS`, `DEFAULT_MAX_DEPTH` — undelivered derivations

`DEFAULT_UBATCH_LEVELS = [128 … 2048]` is a flat literal; nothing about it adapts
to VRAM, though the useful micro-batch range plainly does. `DEFAULT_MAX_DEPTH =
65536` is justified in-code as "deep contexts on CPU prefill very slowly" — a
judgement about *our* prefill speed, applied as a cap to everyone.

### C-F. The sweep time estimate — flat, and obviously wrong at both ends

`est = len(runs) * 90` assumed 90 s per run regardless of model size, so a 270M
model and a 27B model reported the same ~187 minutes. Users decide whether to
start a multi-hour sweep from this number. **Fixed below.**

## Fixed in this pass

- **C-B**: chars-per-token is now measured from llama.cpp's own `prompt_n` and
  used to size subsequent prompts (`calibrate_chars_per_token`). The constant
  remains only as the bootstrap value for the first request.
- **C-F**: the estimate now scales with model size and driver, and says which
  inputs it is a guess from.

## Left as findings

C-A, C-C, C-D and C-E all change what gets measured, so each needs live
validation rather than an argument — and C-A needs it on a backend we do not own.
They are recorded here rather than half-fixed.

## Not in scope

`BENCH_N_PROMPT` / `BENCH_N_GEN` / `BENCH_REPS` and the `PROFILES` numbers are
workload *defaults*, user-facing and overridable by flag. They describe an
intended request shape rather than a claim about hardware.
