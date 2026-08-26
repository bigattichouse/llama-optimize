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

Collected 2026-08-25 (was `faster-inference-qwen.txt` at the repo root).

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

## F2 — Where the draft model lives is a first-class knob

Both vLLM repos put the drafter's placement and quantization at the centre of
their gains (int4 draft head, NVFP4 draft weights, an explicit 8 GB KV pin). On a
24 GB card the draft model is in direct competition with the target model for
VRAM, and that trade is a *placement* decision.

`--spec-draft-ngl` (`-ngld`, `common/arg.cpp` ~line 4112) is llama.cpp's version
and is absent from `FACTORS`, so promoting it looked like an ordinary scalar
factor and no new mechanism.

**It would be an inert column, and this is why.** Tracing the flag through
llama.cpp (checkout `1d2869c6e`) it is consumed in exactly one place:
`common_base_params_to_speculative` copies `params_spec.n_gpu_layers` into the
draft context's params **only inside `if (has_draft)`**
(`common/speculative.cpp` ~2318), and `has_draft` is
`params.speculative.has_dft()` — true only when a separate draft model path
(`-md`) was given.

We never pass `-md`. Our speculation is MTP against the target model's own NextN
head, which takes the other branch of
`common_speculative_init_result`: `llama_init_from_model(model_tgt, cparams)`
(`common/speculative.cpp` ~2405) reuses the already-loaded target model, so there
is no second model to place and nothing for `-ngld` to move. Sweeping it would
add a column whose every level produces an identical run — the "no inert columns"
defect ([`multi-gpu-design.md`](multi-gpu-design.md), M1), and worse than an
omission because a null main effect would read as "draft placement doesn't
matter" rather than "we never tested it".

So F2 is **blocked on a prerequisite, not merely unimplemented**: it needs a
draft-model input (`-md`) before `-ngld` means anything, and a draft model is a
*model-selection* decision — a second GGUF the user supplies — not a tuning knob
we can derive. Worth revisiting if a `--draft-model` input is ever added; the
vLLM evidence for it being a real lever stands, it just does not reach our
configuration surface yet. Recorded in place of the entry, so the next reader
does not re-derive the same dead end from the same two repos.

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

So this is a **profile gap first, a factor gap second**, and the ordering matters:
the factor is worthless without a workload that can move it. Worth noting that
the `agents` profile (8192-token prompts, 32768 ctx floor) is exactly where a
long shared system prompt would live in practice.

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
- [x] **F2** — investigated and **rejected**: `-ngld` is inert without a `-md`
      draft model, which we never pass. Blocked on a draft-model input, not on
      the factor entry (see above)
- [ ] **F3** — fold the `1,4`-on-1:2 datapoint into `ts_levels()` span selection
      when [`multi-gpu-design.md`](multi-gpu-design.md) is implemented; keep the
      single-device configuration reachable
- [ ] **F4** — a shared-prefix workload profile *before* any `--cache-reuse` /
      `--cache-ram` factor
