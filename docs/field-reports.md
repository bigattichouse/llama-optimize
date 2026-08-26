# Field reports — other people's tuned configurations

A log of third-party inference setups we have read, what each one claims, and
what it implies for *our* factor model. [`DESIGN.md`](DESIGN.md) records the
priors we hold and intend to measure; this file records evidence from hardware we
do not own and cannot sweep.

## Why record these at all

A published tuned config is a cheap prior. It cannot tell us what is fast — the
hardware, model and build all differ — but it does tell us **which knobs carried
the weight for someone who cared enough to tune them**, which is exactly the
information a factor registry needs and a heuristic cannot supply.

The risk is copying a number instead of a question. So:

- **R1 — A field report may add a factor or widen a level set. It may never set a
  recommendation.** Every number here belongs to someone else's box. The sweep
  still has to measure it on yours.
- **R2 — Record the non-transferable parts explicitly.** Half of what follows is
  vLLM-only or build-level. Writing down *why* it does not apply is what stops
  the next reader re-deriving the same boundary from the same links.

## Sources

Collected 2026-08-25 (was `faster-inference-qwen.txt` at the repo root), plus
a tuning *method* added 2026-08-26 — see F6.

| Source | Engine | Hardware | Headline |
|---|---|---|---|
| [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) | vLLM 0.27.1 + patches | 1× RTX 3090 (24 GB) | 118–124 tok/s single stream; ~1035 tok/s at 64 concurrent; 381 tok/s reproducing document context |
| [syvai/qwen3.8-27b-3090-fast-variant](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant) | vLLM | 1× RTX 3090 | W4A16, int4 `lm_head`, MTP draft module + 40k draft vocabulary; +0.6% perplexity vs bf16 |
| [seanyourhighness/vllm-sm12x-nvfp4-dflash2](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) | vLLM on SM120/121 | RTX 5090 / DGX Spark | All-NVFP4 (weights + KV + draft), DFlash2 K=7 drafter, ~61% draft acceptance; 108–151 tok/s single, ~349 tok/s at 4 streams |
| [anng-phtk/rocm-vega-llama.cpp](https://github.com/anng-phtk/rocm-vega-llama.cpp) | **llama.cpp**, ROCm 7.2.1 | Vega64 (8 GB, gfx900) + MI50 (16 GB, gfx906) | Mixed legacy-GPU Docker build; `--tensor-split 1,4`; 57.6 tok/s low context, 52.8 tok/s at 8K |

Four Reddit threads pointed at the first three and are unreachable from here
(reddit blocks both fetch paths); the repos and model card they link carry the
substance, so nothing appears to be lost.

## The engine boundary

Three of the four sources are vLLM. Most of what makes them fast is **upstream of
a command line** and therefore outside what this tool can search:

- NVFP4 / W4A16 weight quantization, requantized `lm_head` and embeddings
- A trained MTP draft module and a 40k-token draft vocabulary built from model outputs
- Patched vLLM source (9 patches in one repo, 51 files in the other) and the
  DFlash2 block-diffusion drafter
- FlashInfer kernels, CUDA-graph capture, SM120/SM121 build targets

We tune flags on an existing binary and an existing GGUF. None of the above is a
flag. Recorded so the boundary does not have to be rediscovered — and noting that
their *quality* batteries (perplexity, GSM8K, IFBench) exist precisely because
requantization changes the model's math. Ours does not: quality here is
essentially only KV-type-deep, which `--min-kv` already gates
(`llama-optimize.py`, `apply the KV quality floor`). A quality battery would be
real cost for no signal.

What crosses the boundary is the **shape of the tuning problem**, and four things
did.

## F1 — Acceptance rate is the number that explains a speculative result

Every speculative-decoding source reports acceptance as the headline diagnostic:
~61% draft acceptance on the NVFP4 build, and the 3090 repo's whole `SPEC=` axis
(4-token MTP vs a 7-token DFlash2 block drafter) is argued in those terms.

We sweep six speculative factors — `mtp`, `spec_n_max`, `spec_n_min_frac`,
`spec_p_min`, `spec_p_split`, and five `ngram` variants with their conditional
knobs — and record **none** of it. A row's score says a config was faster or
slower; nothing in the results says whether the drafter was accepted 60% of the
time or 5%.

It is already in the response we parse. `ServerDriver.measure` reads
`r["timings"]` for `predicted_per_second`; llama.cpp puts `draft_n` and
`draft_n_accepted` in that same dict
(`tools/server/server-common.cpp`, `to_json` at ~line 82, checkout `1d2869c6e`).
We discard them.

The sharper point is the guard, not the diagnostic. llama.cpp emits those two
keys only `if (n_draft_tokens > 0)`, so **their absence is a direct detector for
"speculation was silently off"** — precisely the
[issue #8](https://github.com/bigattichouse/llama-optimize/issues/8) failure class
that `FACTORS` documents at `spec_n_min_frac` and `ngram_mod_n_max_off`: an
inverted `n_min`/`n_max` makes llama.cpp discard every draft while the row still
records `mtp=1`, poisoning the `mtp` main effect rather than merely its own score.
[`CONSTRAINED-FACTORS.md`](CONSTRAINED-FACTORS.md) closes that hole *by
construction*, which is the right fix. Acceptance telemetry is the independent
check that the construction held — the same relationship the wall-clock ceiling
has to the server's self-reported rate in
[`measurement-validity.md`](measurement-validity.md): a number the component
cannot fake because it does not produce it.

## F2 — The draft-side surface is a mirror we cannot reach yet

Both vLLM repos put the drafter's placement and quantization at the centre of
their gains (int4 draft head, NVFP4 draft weights, an explicit 8 GB KV pin). On a
24 GB card the draft model competes directly with the target model for VRAM, and
that trade is a *placement* decision — exactly the kind this tool exists to
measure.

`--spec-draft-ngl` (`-ngld`, `common/arg.cpp` ~4146) is llama.cpp's version and is
absent from `FACTORS`,
so promoting it looked like an ordinary scalar factor and no new mechanism.

**It is inert without a draft model.** Traced through llama.cpp (checkout
`4d19b2876`), the flag is consumed in one place:
`common_base_params_to_speculative` copies `params_spec.n_gpu_layers` into the
draft context's params **only inside `if (has_draft)`**
(`common/speculative.cpp` ~2331), where `has_draft` is
`params.speculative.has_dft()` — simply "was a `-md` path given"
(`common/common.h:382`). With no `-md`, MTP takes the other branch of
`common_speculative_init_result` and builds the draft context from the
already-loaded target model (`llama_init_from_model(model_tgt, cparams)`,
~2422). No second model, nothing for `-ngld` to place.

**But "we never pass `-md`" is a fact about this tool, not about llama.cpp.** The
first draft of this note treated it as a reason to drop the factor; that was
scoped to the machine it was written on. Nothing stops a user from owning a draft
model, and a tuning tool that cannot ask their question is the gap, not their
setup. The correct reading is the opposite one: **`-md` is a missing *input*, and
adding it unlocks a whole factor family we currently cannot express.**

That family is close to a complete mirror of the target-side registry
(`common/arg.cpp` ~3927-4165):

| target factor | target flag | draft twin |
|---|---|---|
| `ngl` | `-ngl` | `-ngld` |
| `kv_type` | `-ctk` / `-ctv` | `-ctkd` / `-ctvd` |
| `ncmoe` | `-ncmoe` | `-ncmoed` |
| `ot` | `-ot` | `-otd` |
| `threads` | `-t` | `-td` |
| `threads_batch` | `-tb` | `-tbd` |
| `cpu_mask` / `cpu_range` / `cpu_strict` | `-C` / `-Cr` / `--cpu-strict` | `-Cd` / `-Crd` / `--spec-draft-cpu-strict` |
| `poll` | `--poll` | `--spec-draft-poll` |
| (multi-GPU, planned) | `--device` | `-devd` |

There is no `--spec-draft-head`: the int4 draft-head requantization in the vLLM
sources is a model-artifact change, not a flag. What llama.cpp does have is the
`--spec-type` family — `draft-simple`, `draft-eagle3`, `draft-dflash`,
`draft-dspark` (`common/speculative.cpp:34-38`) — all of which need `-md` and are
therefore invisible to us today. `draft-dflash`/`draft-dspark` are the direct
llama.cpp analogue of the DFlash2 drafter in the NVFP4 source, so the one thing
that source is *about* is reachable here, just not by us yet.

Design and checklist: [`draft-model-design.md`](draft-model-design.md).

## F3 — The `-ts` optimum sits well off VRAM-proportional

The one llama.cpp source is the useful one here, and it bears directly on
[`multi-gpu-design.md`](multi-gpu-design.md) **open question 2** — whether `-ts`
levels should be generated from detected VRAM or from measured bandwidth.

That setup runs `--tensor-split 1,4` on an 8 GB Vega64 + 16 GB MI50 pair. Capacity
says 1:2. They chose 1:4, described as necessary for stability, giving the larger
share to the card that is both bigger *and* (gfx906 HBM2) substantially faster.
The optimum is a long way off the proportional prior, in the direction bandwidth
predicts and capacity does not.

Two consequences for the planned `ts_levels()` generator:

- The design's proposed set — even, VRAM-proportional, "two or three
  interpolations either side" — must **bracket well past proportional**, not
  merely nudge around it. A span that only reaches 1:2.5 on this hardware would
  never have found their answer, and would have reported the proportional prior
  as the winner because nothing better was in the array.
- It is evidence for the bandwidth side of open question 2 without settling it.
  The honest reading is the design's own: keep the VRAM-proportional prior as a
  *prior*, and make the level span wide enough that the measurement can overrule
  it.

There is a second, blunter datapoint in the same benchmark table. The MI50 **on
its own** does 58–59 tok/s at low context; the tuned pair does 57.6. Adding the
second, slower GPU made it slightly worse at every context depth they measured.
Whatever `-sm`/`-ts` levels we generate, the single-device configuration has to
remain reachable as a level — otherwise the sweep cannot report the finding that
the second card is not worth using. This strengthens the design's existing
`-sm none` handling from a completeness argument into a correctness one.

## F4 — Prefix caching is unmeasurable under our current profiles

The largest single number in the whole set is the 3090 repo's `PREFIX_CACHE=1`:
381 tok/s reproducing document context, against 124 tok/s otherwise. The llama.cpp
analogues are `--cache-reuse N` (`common/arg.cpp` ~3524) and `--cache-ram`
(~1706), neither in `FACTORS`.

Adding them today would produce a confident null result, and the reason is a
workload defect rather than a registry gap:

- `ServerDriver.measure` sends the **same prompt** every rep with `cache=True` —
  deliberately, so single-stream reps measure pure decode without re-prefilling.
  The prefix is therefore already 100% reused, and there is nothing left for
  `--cache-reuse` to recover.
- `--cache-reuse` handles *partial* prefix overlap between *differing* prompts.
  None of the three `PROFILES` (`single`, `agents`, `multi`) generates a
  shared-prefix-with-varying-suffix workload, which is the shape that makes it pay
  — and the shape real agent and app traffic actually has.

So this is a **workload gap first, a factor gap second**, and the ordering
matters: the factor is worthless without traffic that can move it. Design and
checklist: [`workload-shape-design.md`](workload-shape-design.md).

Chasing this turned up something worse than a missing capability. Because every
request in a sweep is byte-identical, n-gram speculation — which feeds on
repetition and keeps state across requests — has been measured at 100% acceptance
and better than double its honest throughput. The workload gap is not only
hiding a knob; it has been actively distorting one we already ship. Worth noting that
the `agents` profile (8192-token prompts, 32768 ctx floor) is exactly where a
long shared system prompt would live in practice.

## F6 — A Bayesian autotuner, and what a different search method teaches

[SergioMorillas/vllm-bayesian-autotuner](https://github.com/SergioMorillas/vllm-bayesian-autotuner)
is the odd one out here: not a tuned *configuration* but a tuning *method* — Optuna
TPE over a 24-dimensional vLLM configuration space, 50–100 trials, with random
search as the baseline. Its `docs/` are unusually explicit about methodology
(mostly Spanish), which is what makes it worth reading rather than just noting.

**It reports the honest negative result.** On complex spaces TPE beats random by
+5.4% objective / +3.1% peak throughput; on a simpler 6-dimensional space the
advantage collapses to +0.4% — a tie. Their reading is that "the advantage of
Bayesian optimization grows with configuration-space complexity."

That is a useful calibration rather than a threat. Our surviving factor count
after Morris screening is single digits, which is exactly where their own data
says BO buys ~nothing over much simpler search. It also marks where the argument
would change: if `--factor` counts ever ran to twenty-plus *without* screening,
their result says a surrogate would start to pay.

The deeper difference is what the two methods return.
[`DESIGN.md`](DESIGN.md) exists because a Taguchi array yields **per-knob main
effects** and Morris yields **μ\* and σ** — which knob matters, and whether its
effect depends on the others. BO returns a good point and a surrogate that is not
meant to be read. For a tool whose output is "here is the command, and here is
which knobs mattered on your box", attribution is not a nice-to-have; it is half
the product. Neither method dominates — they answer different questions.

### Independent convergence on our own invariants

Two of their design decisions are ours arrived at from the other direction, which
is the strongest evidence available that the principles are real and not just
house style:

- **Conditional sampling.** Mamba parameters are sampled *only* when prefix
  caching is on, explicitly to avoid "poisoning the surrogate on phantom
  configurations". That is [`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md)'s
  I2/I3 restated in Bayesian terms: an inactive parameter must not participate,
  or it corrupts the model of every other parameter.
- **Inactive dimensions emit no flag** and retain engine defaults — the same rule
  as our `is_active` emission gate.

Where we differ is in *how* an invalid combination is prevented. They **prune**
(discard the trial) or **repair** (force to feasible). We **derive**, so the
invalid assignment cannot be constructed at all
([`CONSTRAINED-FACTORS.md`](CONSTRAINED-FACTORS.md)). Ours needs no runtime check
and cannot be forgotten at a new call site, but it only works where the
constraint is a simple ordering; their repair strategy covers constraints ours
cannot express, and is worth remembering if a future factor pair is messier than
`b >= ub`.

### What is worth taking

**Their memory model is our OOM pruner, already specified.** ROADMAP item 2 asks
for a predictive VRAM footprint and has been blocked on what to actually sum.
They sum six terms — model weights, KV cache, Mamba state, CUDA-graph capture
overhead, activations, and a safety margin — against two inequalities: total
within `gpu_memory_utilization × VRAM`, and total plus driver context within
physical VRAM. Deliberately first-order additive rather than empirical,
*because* a conservative estimate is the point. That matches our own standard for
the pruner exactly ("a wrongly-skipped viable config is worse than a wasted OOM
run"), and it is a blueprint rather than an idea. The llama.cpp translation drops
Mamba and CUDA graphs, keeps weights/KV/activations/margin, and gains the
partial-offload split that has no vLLM analogue.

**Objectives we do not measure.** Their multi-objective mode runs NSGA-II over up
to seven axes: throughput, context, latency, goodput, TTFT, TPOT, and
**tokens/joule**. Our Pareto is two axes (context vs tg). TTFT is already ROADMAP
item 5 and currently *derived* rather than measured; TPOT and goodput are cheap
once per-request timings are recorded, which F1 now does. Energy is a genuinely
new axis and an interesting one on an MI50, where the thermal ceiling is already
the dominant noise source.

**A designed prompt battery.** They have a whole document on it
(`06-diseno-bateria-prompts.md`). We send one identical prompt for every rep,
which F4 established is not a neutral choice but an active distortion. Worth
reading before building `--prefix-reuse`.

## F5 — Legacy/mixed-vendor builds (recorded, out of scope)

The ROCm repo's actual subject is making current llama.cpp *run* on gfx900 —
ROCm 7.2.1 ships no rocBLAS/Tensile kernels for it, so a two-stage Docker build
transplants gfx900 kernels from ROCm 6.3.4 into a patched 7.2.1 image, then builds
with `-DGGML_HIP=ON -DGPU_TARGETS="gfx900;gfx906"`. It also warns against
`HSA_OVERRIDE_GFX_VERSION` when the hardware is natively detected.

All of this is upstream of us: it produces the binary we then tune, the same
relationship [`docker.md`](docker.md) describes. Recorded because our own
reference hardware is a gfx906 MI50 and this is the most likely thing a user on
that card will hit *before* they can run a sweep at all.

## Checklist

- [x] **F1** — `draft_acc` column: `_draft_totals` sums `draft_n` /
      `draft_n_accepted` over the measured reps in `ServerSession.measure`
- [x] **F1** — `spec_off` flag for a run that asked for speculation and drafted
      nothing, covered in `--selftest` against a synthetic response payload
      (no GPU). Note the gate is often *not* a factor — `build_server_args`
      emits MTP fixed-on — so the check reads the config, not just the row
- [ ] **F2** — a `--draft-model` input, then the draft-side mirror as conditional
      factors gated on it ([`draft-model-design.md`](draft-model-design.md)).
      `-ngld` alone was the wrong unit of work: inert without `-md`, and one of a
      dozen twins once `-md` exists
- [ ] **F3** — fold the `1,4`-on-1:2 datapoint into `ts_levels()` span selection
      when [`multi-gpu-design.md`](multi-gpu-design.md) is implemented; keep the
      single-device configuration reachable
- [ ] **F4** — a workload *shape* input (`--prefix-reuse`) before any
      `--cache-reuse` / `--cache-ram` factor
      ([`workload-shape-design.md`](workload-shape-design.md)). The profile was
      the wrong unit too: shape is an input describing the user's traffic, not a
      canned profile to pick from
