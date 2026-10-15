# Sweep cost — design & work log

Why a sweep takes as long as it does, which dials actually change that, and why
the tool asks questions instead of trusting the dials to be found. The
*principles* live in [`DESIGN.md`](DESIGN.md); this is the cost-specific model
and checklist, the same split [`ngram-design.md`](ngram-design.md) and
[`multi-gpu-design.md`](multi-gpu-design.md) use.

## Origin

Field feedback, 2026-08-30:

> Be prepared to wait an extremely long time for tests to complete. […] It will
> test everything under the sun and take days with enough iterations but you can
> shorten it to a few hours with adjustments.
>
> - set nkvo to vram only if you plan to leave fit off
> - limit context sizes
> - limit thread count
> - limit cache types
> - look at the dense and moe knobs and decide if you want to test the tradeoffs

Every item is sound advice. Taken one at a time, four of the five do nothing —
and that is the defect, not the advice.

## The five tips, audited

Checked against the tree as of `5810272`. Four were already satisfiable; the
failure was that three of them did not *appear* to work when applied singly, and
one was already the default.

| Tip | Disposition |
|---|---|
| "set nkvo to vram only if you plan to leave fit off" | **Already closed, and not our hazard.** Auto-fit is never on in a sweep: `build_server_args` forces `--fit off` where the binary has it, and `llama-bench` has no `--fit` at all — only `--fit-target`/`--fit-ctx`, both default-off and never emitted. `supports_flag`'s boundary match is what keeps `--fit` from matching `--fit-target`, so the server flag is not accidentally sent to bench. `nkvo` is therefore free to sweep in both directions; the tip applies to hand-run llama.cpp |
| "limit context sizes" | **Works, and is the biggest single saving** — `--ctx-size N` pins `n_depth` to one level. It just did not shrink the array on its own (C1) |
| "limit thread count" | **Works** — `--factor threads=8`. Same caveat: no effect on run count alone |
| "limit cache types" | **Already the default.** `--min-kv` defaults to `q8_0`, which drops `q5_1`/`q4_1`/`q4_0`: five levels become two before anyone asks |
| "look at the dense and moe knobs (`ot`, `ncmoe`)" | **Partly.** The dense side is now one `ffn_place` column spanning `-ot` and `-ncffn` ([`flag-coverage.md`](flag-coverage.md)); `ncmoe` is unchanged for MoE. The interview does not ask about offload tolerance — see Open |

The pattern across all five: the capability existed, the composition did not, and
nothing in the output distinguished "no effect" from "ignored".

## Defect: the dials do not compose the way anyone would assume

`choose_array` sizes the design on `max(level counts)` across the varying
factors. So narrowing *one* knob leaves the array exactly where it was. Measured
before any of this work, on `gemma-3-270m-it-Q8_0`:

| invocation | runs |
|---|---|
| default | 125 |
| `--factor kv_type=q8_0 --factor threads=8` | **125** |
| `--ctx-size 8192` | **125** |
| ngl + threads + ubatch capped to 3 levels **and** `--ctx-size` | **27** |

A user who followed the advice, limited context sizes, saw 125 runs anyway and
concluded the flags were ignored would be drawing the only reasonable conclusion
available to them. Nothing in the output distinguishes "your narrowing had no
effect" from "your narrowing was ignored".

`n_depth` made this worse than it looks. `depth_levels` routed through
`five_levels_span`, which hard-coded its 5, so the context axis could be five
levels or — via `--ctx-size` — exactly one. There was no way to ask for three.

## Two things drive wall clock, and they are not the same thing

**C1 — the run count**, set by the array, set in turn by the widest factor.
Addressed by `--levels N`, which narrows every auto-generated factor *together*
because that is the only way the array moves: L125/125 runs at 5, L27/27 at 3,
L16/16 at 2. Explicit `--factor` values are never touched — a user who names
levels has said something specific and it is not the dial's business to override
it.

Fewer levels is a coarser grid, not a wrong answer: 3 levels still show
curvature, 2 only a direction. This is the ordinary Taguchi resolution trade and
should be described as such rather than as a discount.

**C2 — the slow configs**, which dominate and are invisible in the run count. At
`--n-gen 256` a config running at 0.5 t/s takes ~8.5 minutes *per rep*; the run
count says 125 either way. Addressed by `--min-tgs` / `--min-pps`, which tighten
the per-config deadline rather than adding a second mechanism: a config that
would *meet* the floor finishes `reps x n_gen` tokens within `tokens / floor`
seconds, so exceeding that budget is itself the finding.

That is arithmetic, not instrumentation, which is why it works on `llama-bench`
where nothing can be observed mid-run — no streaming parser, no partial-output
plumbing. On the server driver `--min-pps` is answerable the moment the warm
request returns, so a config failing it never pays for its decode reps.

Worst case on an L125, `--n-gen 256`, `--reps 3`: **41.7h** at the default
`--timeout 1200`, **17.8h** at `--min-tgs 2`, **3.5h** at `--min-tgs 10`.

## S1 — `--timeout` was not a bound on anything nameable

Prerequisite for C2, and a defect in its own right. On `llama-bench` it bounded
the whole process, which is what it reads like. On the server driver it was
handed to *each HTTP request*, and the single-stream path sends one warm request
plus `reps` measured ones — so a config could legitimately spend
`(1 + reps) x timeout`, i.e. 80 minutes at a "20 minute" timeout. It is now a
deadline for the whole config on both drivers.

## Q1 — why an interview rather than better documentation

The dials now compose correctly, but a user still has to know that `--levels`
exists and that it is the one that matters. The knob reference is long and the
cost dials are not what anyone is looking for when they arrive.

An interview is the only interface where the answers compose into a design *by
construction*: the user describes their workload and the tool derives a factor
set that is internally consistent, rather than the user assembling flags whose
interaction is the thing they got wrong.

Design constraints, in the order they mattered:

- **Intent first, cost second.** Ask about the workload (context actually
  served, slowest useful speed, breadth, repeats), then show the run count and
  estimate. The number appears while it can still change the answer. Asking for
  a time budget first would invert that and imply a precision
  `estimate_secs_per_run` explicitly disclaims.
- **Print the derived command; never silently apply it.** The argv is the record
  of what ran. A questionnaire that configured a sweep invisibly would take that
  away — the user could not save it, edit it, diff it against a previous run, or
  paste it into a bug report. It is printed, then a `y/N`.
- **Answers → argv is a pure function** (`intent_args`). Testable with no GPU and
  no terminal, which is what keeps the interactive layer thin enough to trust.
- **Hard to trigger by accident.** Nothing happens when stdout is redirected — a
  script blocking on `input()` is a hang, not a prompt — or when any
  intent-bearing flag is present, since the interview would then be overriding a
  decision rather than helping make one. The guard is asserted over the whole
  `_INTENT_FLAGS` list, so a flag added later without a matching guard fails the
  selftest instead of silently prompting.

## The `ngl` grid spends its levels where the answer cannot be (issue #14)

`ngl_levels` spans `0 .. n_layers` evenly and never asks whether the model fits.
On a 40-layer model that fits entirely in VRAM the grid is `[0, 10, 20, 30, 40]`
and four of five levels put layers on the CPU — every one of them a near-certain
loser.

### What it actually costs, which is not what it first looks like

The tempting framing is "80% of a factor's levels wasted". That is wrong, and
getting it right decides the shape of the fix.

In an orthogonal array **every row informs every factor's main effect**. A row at
`ngl=0` is still a legitimate observation of `kv_type`, `ubatch` and the rest. So
those rows are not wasted in the information sense. The costs are different ones,
and both are real:

- **Wall clock, disproportionately.** `ngl=0` is CPU-only decode: on a 27B model
  that is an order of magnitude slower than full offload. The low-`ngl` rows are
  a minority of the design and a majority of its runtime. This is the big one,
  and it is why this is a sweep-cost item and not only a correctness one.
- **Additivity.** Main effects assume a factor's effect is roughly independent of
  the others. `kv_type`'s effect at `ngl=0` (KV in system RAM, CPU attention) is
  not the same phenomenon as at `ngl=40`, so averaging across both estimates an
  effect for a regime nobody will deploy. Measuring the other factors in the
  regime the user will actually run is worth more than the extra span.

`choose_array` sizes on the widest factor, so the level *count* is what fixes the
array at L25 or L125 — narrowing `ngl` alone buys nothing (SC1). This change
moves where the levels sit; it does not remove any.

### Why not cluster at the top

The first sketch on the issue was `[0, top-2, top-1, top]`. Rejected for two
reasons, and the second is the one that holds at every model size.

**Below the noise floor.** On a 40-layer model those three levels differ by ~7%
of the model's compute, against an observed run-to-run spread of 5–27% (thermal;
the `--verify-picks` motivation). Three levels resolving a difference smaller
than the noise is the same waste in a new place. This argument weakens on small
models, where a quarter of the range *is* a couple of layers.

**No fallback if the verdict is wrong.** A design with nothing between `0` and
the top has no level left where a partially-offloaded optimum could appear. The
anchor keeps the sweep producing *a* measurement; a spread keeps it able to find
the *right* one. This is the argument that does not depend on model size.

So: keep the CPU anchor, and span the remaining levels across the **top quarter**
of the layer range rather than the whole of it. For 40 layers at `--levels 5`:
`[0, 30, 33, 37, 40]` instead of `[0, 10, 20, 30, 40]` — the two slowest rows in
the design are gone, the gradient and the anchor survive.

The window widens when a quarter is too narrow to hold `levels - 1` distinct
values, or a small model would silently lose levels to deduplication: an 18-layer
model gets `[0, 14, 15, 17, 18]` and an 8-layer one `[0, 4, 5, 7, 8]`, both still
five levels. A model with fewer layers than levels cannot be biased at all and is
not pretended otherwise — every `ngl` value that exists is already in the span.

### Why the anchor stays, always

`ngl=0` is kept at every level count, including on MoE models where `ncmoe`
already spans CPU offload and the pair is arguably redundant. It is not there for
information, it is there because **the fit verdict can be wrong**. If the
estimate says "fits" and it does not, every biased level OOMs and the anchor is
the only row that can still produce a measurement. A grid that deletes its own
rescue rows on the strength of an estimate is the failure mode the OOM pruner was
already written to avoid (P3: a wrong estimate deletes rows, a missing one merely
runs them).

### What decides "fits"

`predict_fits` — llama.cpp's own `llama-fit-params`, the same estimator the OOM
pruner uses, so the two cannot disagree about what fits. It answers True / False /
**None**, and None keeps the even span. Three conditions on the probe:

- **The deepest depth in the design, not the shallowest.** `ngl` levels are
  generated once for the whole design while `n_depth` varies per row, so a model
  that fits at depth 0 and not at 64k would otherwise get a grid that is
  optimistic exactly where OOM is likeliest.
- **The most demanding cell otherwise:** highest-quality (largest) `kv_type` in
  the design, KV on the GPU (`nkvo=0`), no tensor offload. Biasing only when even
  the worst cell fits means the answer errs toward keeping the even span, which
  is the safe direction.
- **`--no-oom-prune` disables it.** A user who has told the tool not to trust the
  estimator enough to delete rows has not asked it to trust it enough to shape
  the grid either.

### Recurrent models: two levels, not five (issue #18)

A separate reason the `ngl` grid can be wrong, and a harder one — those rows do
not merely lose, they **abort**. On a hybrid SSM model a *partial* offload
core-dumps `llama-server` shortly after `llama threadpool init`, before the slots
are set up.

> **Correction (2026-09-02).** Earlier revisions of this section named
> `resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
> (chunked) is assigned to device ROCm0` as the cause, because the crash follows
> it. It is not. That warning and its `set to disabled` follow-up appear
> **identically in launches that load fine**; llama.cpp is reporting a fallback
> it took successfully. Post hoc, not propter hoc — the crash is later.

Mapped on `Qwen3.6-35B-A3B` (`qwen35moe`, 40 blocks, `-ncmoe 40`):

```
-ngl   0   1   2   3   5  10  20  30  38  39  40  41  99
      ok  ok  ok   X   X   X   X   X   X   X   X  ok  ok
```

Two things in that map were not what the first pass assumed, and both matter.

**It is not "any split".** `-ngl 1` and `-ngl 2` run fine — presumably too few
layers on the GPU for the fused op to be placed on ROCm at all. They are simply
useless placements, roughly CPU speed on a 40-layer model. The crash band is
**3 .. n_layers**.

**`-ngl n_layers` dies; `n_layers + 1` lives.** At 40 the output tensor takes the
last slot and block 0 is left on the CPU — exactly the straddle that crashes. The
first pass chose `99` over `n_layers` on the reasoning that the output tensor was
an untested boundary; that guess was right, and is now measured.

Confirmed on a second model of the same family: `Qwen3.6-27B-Q5_K_M` (`qwen35`,
dense, 64 blocks) dies at `-ngl 5`, `-ngl 40` and `-ngl 53`.

**But the map above is the *bare* map, and the dead band is not a property of
`-ngl` alone.** On `Qwen3.6-27B-Q5_K_M`, the same `-ngl 53` that core-dumps bare
loads and generates correctly under `--spec-type draft-eagle3`:

| added to the crashing launch | `-ngl 53` |
|---|---|
| *(nothing)* | segfault |
| `--spec-type draft-eagle3` — even with **no** `-md` | **loads** |
| `--spec-type draft-dflash` | **loads** |
| `--spec-type draft-simple`, `draft-mtp` | segfault |
| `--spec-type ngram-mod`, `ngram-simple`, `ngram-cache`, `ngram-map-k` | segfault |
| `-md <head>` alone, `-ngld 0/1`, `-ctk q8_0`, `-ub 512`, `-t 8`, `--no-repack`, `--op-offload`, `--fit off`, `--fit` on, `--load-mode mmap` | segfault |

The two survivors are the heads that reuse the target's hidden states. This was
checked for *correctness*, not just liveness — the rescued server answers "The
capital of France is Paris" at 6.8 t/s, against 18.8 t/s at full offload. Partial
offload on this model is slow, not broken, and `-ngl 5` is rescued by the same
flag.

For this quant, of this model, on this hardware: `Qwen3.6-27B-Q5_K_M`, MI50 32GB
(ROCm), llama.cpp `6c84c7d5d`.

So on `ssm_state > 0` the grid is `[0, 99]`: the only two placements that
execute. Three of five default levels would otherwise be recorded as `SIGNAL`,
which is honest but useless, and on a first run it looks like the tool is broken
rather than the placement being impossible.

**Under `--run` the box is asked, not told.** `probe_loadable_ngl` launches each
candidate level, checks whether the server stands up, and keeps the ones that do
— so the collapse above is a *default* for plan-only, not a ceiling. Verified
against the model it was measured on: given `0, 30, 33, 37, 40, 99` it returns
`0, 99` and reports the four that died. On a backend where the fused op is fine,
it returns all six and the layers-for-context axis survives intact.

**And it asks under more than one assignment.** The table above means the probe
cannot fix every other factor at one level and call the answer general: with
`spec_type` swept, probing `none` would report `-ngl 53` dead when the `eagle3`
rows would have run it. So a candidate is dead only if it fails under **every**
level of `LOAD_SENSITIVE_FACTORS`, retried only for the ones that failed — a
sweep with no such factor costs exactly what it did before. The two errors are
again asymmetric: a wrong "dead" deletes the axis silently, while a wrong "live"
costs one run that is recorded as `SIGNAL` and reported by `dead_levels`. Levels
rescued this way are printed as such, because "loads only with
`spec_type=eagle3`" is a different claim from "loads".

**A probe never overrides `--factor`.** `--factor ngl=99` is the user answering
this question themselves. This shipped broken: a run pinned to `ngl=99` to
isolate a draft head swept `48/53/59/64` instead, because the probe re-expanded
the pin from the un-collapsed candidate span, and the rows then varied in the one
factor that had been held fixed.

That matters because which levels load is a property of the model, the backend
and the build, and none of those is knowable from the machine this was written
on. The cost is one server launch per candidate, weights usually already in page
cache, and only on models with recurrent memory.

**The gate is broader than the evidence, deliberately.** `ssm_state > 0` also
matches Mamba, Jamba, RWKV and Falcon-H1, none of which has been tested, and all
of the evidence is one architecture family on one backend (ROCm). Partial offload
is ordinary and works fine for most models — it is only this class that has been
seen to abort. The gate is applied anyway because the two errors are not
symmetric:

- **Wrong here**: three grid levels lost on a model that could have used them, on
  a knob the user can hand back with `--factor ngl=0,16,32,48,64`. The printed
  note says exactly that.
- **Wrong the other way**: three fifths of every sweep spent on rows that cannot
  produce a number, on an architecture where nothing the user does helps.

A recoverable loss against an unrecoverable one. If a recurrent architecture
turns up that *does* support partial offload, the honest fix is to narrow the gate
— or to make it adaptive, dropping the remaining partial levels only after one has
actually died (`died_on_signal` already detects that).

**99 rather than `n_layers`**, deliberately: `-ngl 99` is the spelling that was
verified, and `-ngl n_layers` still leaves the output tensor on the CPU — a split
of a different kind that was not tested. llama.cpp clamps 99 to the real count.

**The CPU anchor stays**, for a different reason than in the fit case: if the
model does not fit, the OOM pruner drops the `99` row and CPU-only is genuinely
the one placement that runs. That is a real answer for the report to give.

A side effect worth noting for issue #17: on a hybrid **MoE** the `ngl` column
stops competing with `ncmoe` for the offload axis, because `ngl` no longer has
intermediate levels. `-ngl 99 -ncmoe N` is measured and works — that is how the
prefix-reuse experiments ran the 25 GiB model in 3.1 GiB.

### Not addressed here — issue #17

Whether `ngl` and `ncmoe` should both be arguing about what lives on the CPU on
MoE models. The bias makes them agree more often — a fitting MoE model now gets
its `ngl` levels near the top, leaving `ncmoe` as the offload axis — but that is a
side effect, not a fix: nothing stops them contradicting each other, and on a
model that does *not* fit the even span is restored and the overlap is back in
full. `ngl=0 × any ncmoe` is the sharp case, where `ncmoe` cannot act yet still
votes on its own main effect. The fix is a constrained or conditional relation,
not a level-set change, and it wants a real MoE model to choose between them.

## What the arrays can hold, and what that costs

Checked against the vendored library (`robust/optimize/taguchi`) because a level
set is not free just because a *column* is.

**Arrays implemented:** L1–L5, L8, L9, L16, L18, L25, L27, L32, L64, L81, L125,
L128, L243, L256, L512, L625, L729, L1024, L2187, L3125.

**Five levels is the ceiling.** There is no six-level array in the library, and
`generate_runs` rejects one outright — `No available array supports 6 levels
(maximum available: 5)`. That is a property of orthogonal arrays, not a gap in
the tool, and it is why the `ngram` gate screens five variants and reaches
`ngram-cache` only through `--ngram-type ngram-cache` or `--factor ngram=`.

**L18 is mixed-level (one 2-level column, seven 3-level) and is implemented**,
but `choose_array` still skips it, and the reasoning it was skipped for still
holds: L18 applies only when the widest factor has three levels *and* at most
eight factors vary. At `--levels 3` the default set has thirteen varying factors,
so L27 wins anyway. Wiring a second selection path would buy nothing until a real
factor set wants it.

**Where the headroom actually went.** Adding `repack`, `no_op_offload` and
`no_host` costs zero runs at every `--levels` setting — L16 holds 15 factors, L27
holds 13, L125 holds 31, and the default set fits all three. But at `--levels 3`
the set is now **exactly** at L27's cap of thirteen. One more three-level factor
tips it to L81: 27 runs to 81, a 3x jump for one column. "Free" is true today and
should be re-checked before the next factor is added, not assumed.

## C3 — the time around the measurement (2026-10-14)

A code review (no GPU) of where a server-driver config spends its wall clock
found that the measured reps are the minority of it, and that the floors did
not deliver what `--tgs-timeout` reads like: a config is not judged after 60s,
it is judged when `n_gen x (reps+1) / min_tgs` runs out — 512s at
`--n-gen 256 --reps 3 --min-tgs 2`. Five items, cheapest-to-trust first. None of
them has been timed on hardware yet; the savings below are arithmetic from the
code, not measurements.

### C3a — the preheat sampled once per request, so it could not end early

`run_until_warm` read the sensor once per `work()` call and needed
`THERMAL_PLATEAU_POLLS + 1` readings to see a plateau. So every warm-mode config
paid **at least five full, unmeasured requests** before its three measured ones —
including a config that inherited a card already sitting at its plateau from
the previous config in the same session. For a slow config those five requests
were most of its budget (bounded only by the one-third cap).

Fix: the load runs in a worker thread and the sensor is polled on a clock, at
the same 3s cadence `wait_until_cool` and `idle_baseline_c` already use. The
plateau test is unchanged (`_plateaued`, same window, same band), so a card on
its ramp is still not mistaken for a flat one; a card already at its plateau is
now released after ~12s of load instead of after five requests. The in-flight
request is always allowed to finish before the reps start, because a preheat
request still holding the slot would queue the first rep behind it.

What this does not change: a config is still heated under *its own* request,
and the window that defines "flat" is the same one that was tuned against the
MI50 99↔100°C oscillation.

**Measured afterwards, and the cost was not what this section first claimed.**
A/B on gemma-3-270m, MI50, llama.cpp b11639, one config (`ngl` 18, 8k depth,
~0.2 s requests), two trials of each sampler alternating on the same card:

| sampler | verdict | time | requests | temperature |
|---|---|---|---|---|
| per-request (old) | warm | 1.5 s, 1.6 s | 4, 4 | 94, 95 °C |
| on a clock (new) | warm | 239 s, 34 s | 898, 122 | 98, 100 °C |

The old sampler's plateau was **false**: its five readings were taken within
1.5 s and were identical (94, 94, 94, 94, 94) because the sensor had not
updated between them — the window measured the sensor's refresh rate, not the
card. Under sustained load the same config settles at 98–100 °C, so for fast
configs the old preheat measured the burst rate `warm` mode exists to avoid.

So C3a is a **correctness fix with a mixed cost**: cheaper on slow configs (no
five full unmeasured requests), more expensive on fast ones, which now heat
until they are actually hot — up to `THERMAL_WARMUP_CAP_S`. One config of
sixteen hit that cap in a real sweep. `--thermal-mode idle` remains the fast,
burst-rate alternative. The "~12 s" above holds only for a card already at its
plateau.

### C3b — the warm request already answers `--min-tgs`

The single-stream path sends one warm request (`n_gen` tokens) before anything
else, and `prompt_per_second` from it is already checked against `--min-pps`.
Its `predicted_per_second` was thrown away. It is checked now, **before** the
preheat.

Why the cold number is a safe basis for rejection: throttling only lowers the
rate, so a card that cannot reach the floor cold will not reach it hot. Two
conditions keep that argument honest:

- **A margin** (`WARM_TG_EXIT_FRACTION = 0.75`): the warm request is a single
  sample, and the observed run-to-run spread is 5–27% (the `--verify-picks`
  motivation). Only a config below three quarters of the floor exits here; one
  near the floor is measured as before and judged on its median.
- **Not under speculation.** At reuse 1.0 the reps repeat the warm prompt, so a
  self-speculative head (ngram-cache and friends) accepts far more on the reps
  than on the first sight of the prompt ([`ngram-design.md`](ngram-design.md)).
  The warm rate is then not a bound on the rep rate, and the check is skipped.

### C3c — defect: the floor budget had no room for prefill

`slow_budget_secs` counted decode tokens only. Two ways that produced a false
`SLOW` — a config the floor should keep, recorded as below it:

- **Depth.** `llama-bench -d D` re-fills the context before every rep of both
  of its tests, and the server's warm request fills it once. At
  `--n-gen 64 --reps 3 --min-tgs 10` the budget was 60s (the grace floor); a
  32k-deep row prefilling at 400 t/s needs 80s before it decodes a token.
- **Concurrency.** Under `--parallel`/`concurrency` the server driver's `tg` is
  the *aggregate* over the streams, and each round generates `par x n_gen`
  tokens. A config exactly at the floor needs `par` times the budget it was
  given.

Fix: the budget is `decode_tokens / min_tgs + prefill_tokens / prefill_floor`,
with both token counts derived per driver and per row (depth, `par`). The
prefill floor is `--min-pps` when set, and otherwise `--min-tgs` itself — a
prefill slower than the decode floor is not a configuration anyone is looking
for, and batched prefill below decode speed has not been seen on any backend
this tool has run on. Without a prefill floor the deep rows now get roughly
their full `--timeout`; that is the honest cost of not having said how slow a
prefill may be, and C3b is what still exits them early on decode.

### C3d — a level that has only ever been slow stops being run

`SLOW` rows carry no weight in the analysis: `measured_ok` keeps only `OK`
rows, so they enter neither the main effects nor the picks. Running a third
`ngl=0` row on a model where the first two were `SLOW` therefore buys nothing
the report uses, and on a large model it is the most expensive row in the
design.

So once a level has at least `--slow-skip N` (default 2) rows, **all** `SLOW`,
and some *other* level of the same factor has produced an `OK` row, the
remaining rows at that level are recorded as `SKIP_SLOW` without being run. The
rule is `dead_levels` — already trusted to name a level from balanced evidence —
applied during the sweep instead of after it. Conditions, each for a reason:

- **Measured, never estimated** (SC5): only rows that ran and came back `SLOW`
  count. `SKIP_SLOW` rows are not evidence, or one skip would beget the next.
- **At least two rows, at different partners.** The orthogonal array puts each
  level beside different settings of every other factor, and the execution
  order is shuffled, so two rows are two different combinations. One `SLOW` row
  is a combination; two are starting to be a level.
- **Some other level is OK.** If every level of a factor is slow, the box is
  below the floor, not the level — the same exclusion `show_dead_levels` makes.
- **Only where the factor could act.** Rows where the factor was inert or gated
  off (`is_active` / `is_inert`) carry no evidence about its level.

The risk is a level that is slow only next to particular partners (an `ngl`
level slow only with `nkvo=1`). That costs the rows that would have shown it,
which are rows the analysis was going to drop anyway. The status names why each
row was skipped, and `--slow-skip 0` turns it off.

### C3e — a predicted OOM still started a server

On the server driver the OOM prediction ran per row, *after* the group's
`ServerSession` had been launched. A group whose every pending row is
`SKIP_PRED` or `SKIP_SLOW` now launches nothing.

Test coverage is partial: `pre_skip` is unit-tested, but the selftest has no
harness that drives the main sweep loop, so the group-level call site itself
is checked only by reading it.

### C3f — defect: refinement passes ran without the floors

Found while wiring `--slow-skip` through. `build_child_argv` builds each
`--iterate` / `--screen` pass's command line explicitly, and never forwarded
`--min-tgs`, `--min-pps` or `--tgs-timeout`. Every pass after the first ran
unbounded at `--timeout`. They are forwarded now, with `--slow-skip`.

Note what C3d does across passes: each child pass starts with an empty `rows`
list, so a level retired in pass 1 is run again (twice) in pass 2 before it can
be retired there. Refinement narrows the levels anyway, so this is left as is.

### Not done here: streaming

C3b judges on the warm request, which still runs to `n_gen`. Judging after
`--tgs-timeout` seconds of decode regardless of `n_gen` — the literal reading of
the flag — needs a streamed `/completion` so tokens can be counted as they
arrive. It is the larger change (the request path every test fake replaces) and
is left as the next item.

## C4 — size knobs on a doubling scale (2026-10-14)

Prompted by a field config (RTX A2000 12GB, `Qwen3.6-35B-A3B` with every expert on
the CPU, 262k context) and the question "why walk 16000, 17000, 18000?".

Sizes — context depth, a cache in MiB, `ubatch` — have effects that scale with
the *ratio* between levels, not the difference: going from 4k to 8k context
changes far more than going from 60k to 64k. An even span spends most of its
levels at the top of the range, where neighbours barely differ, and for depth
those are also the most expensive rows in the design (prefill grows with depth).

**Levels.** `log_levels_span(lo, hi, n, quantum)` spaces levels by a constant
ratio, rounded to a quantum. With `lo = 0` the zero level is kept (it is the
"off" or "shallowest" anchor) and the rest are halvings of `hi`:

| knob | before (even) | now (doubling) |
|---|---|---|
| `n_depth`, floor 0, cap 64k | 0, 16k, 32k, 48k, 64k | 0, 8k, 16k, 32k, 64k |
| `n_depth`, floor 8k, cap 64k | 8k, 22k, 36k, 50k, 64k | 8k, 13k, 22k, 38k, 64k |
| `moe_cache`, ceiling 8 GiB | 0, 2k, 4k, 6k, 8k | 0, 1k, 2k, 4k, 8k |

`ubatch` was already powers of two. Factors carry `"scale": "log"` and a
`"quantum"` in the registry, and `refine_numeric` honours both, so an
`--iterate` refinement around `ubatch=512` brackets 256..1024 on the same
doubling scale instead of walking 256, 448, 640, 832, 1024.

This moves the depth levels of every new sweep, so depth-sliced comparisons
against older CSVs should compare like depths (the CHANGELOG says so).

**The context probe gallops instead of bisecting evenly.** `probe_max_context`
used to test `cap`, then bisect `0..cap` at its arithmetic midpoint. Now:

1. **Seed from the estimator.** `estimated_max_depth` bisects with
   `llama-fit-params` — a metadata read, no GPU, no prefill — for the deepest
   depth predicted to fit. The real probe starts there, so on a box where the
   estimator is right the answer costs two or three launches.
2. **Gallop.** From the seed, step outward in doubling offsets — +2k, +4k,
   +8k while it loads, or −2k, −4k, −8k while it does not — until the answer
   flips. The estimator is usually close, so this takes a step or two; when it
   is far off, the offsets reach the boundary in `log2(error)` launches.
   Doubling the depth itself was the first version and was worse: from a seed
   of 39k it jumped straight to the 64k cap, discarding the estimate.
3. **Bisect the bracket** arithmetically to 2048 tokens, as before.

Without an estimator the seed is `cap`, which is what the probe tested first
anyway. Galloping assumes the boundary is monotone in depth — true for memory,
which is all this probe asks ("does it load"); it is *not* assumed for
throughput, which the design measures at fixed levels instead.

**The `moe_cache` ceiling comes from headroom, not a quarter of the card.** On
the A2000 config the experts are all on the CPU and the KV cache is small, so
most of the 12 GB is free and a quarter (3 GB) undershoots. The ceiling is now
`total VRAM - --fit-headroom - footprint` of the design's most demanding
all-experts-on-CPU cell (deepest depth, largest KV, `ngl` 99), as predicted by
`llama-fit-params` — the same worst-case cell `full_offload_fits` uses, so the
top level fits in every row it lands in. No estimator: the quarter-of-VRAM
fallback. Headroom under one step: no column.

## C5 — the row must describe what ran (2026-10-14)

A real sweep on gemma-3-270m (MI50, llama.cpp b11639) after C3/C4 turned up
four ways the tool ran something, got back something other than it asked for,
and recorded the wrong thing. User's framing: "when it might run stuff and not
get what it expected back so it reports something incorrect".

| # | What was asked | What happened | Recorded as |
|---|---|---|---|
| 1 | depth = n_ctx_train (32768) + 512 prompt | llama-server caps a slot at n_ctx_train; HTTP 400 | `ERROR`, no reason — every deepest row |
| 2 | ~130+ requests to one session | the unread stderr pipe filled at 64 KB; server froze | `TIMEOUT`/`SLOW`, "depth 0 failed to load" |
| 3 | `-b ub×16` at an 8k context | llama.cpp clamps `n_batch` to `n_ctx`, `n_ubatch` to `n_batch`, silently | the requested value |
| 4 | a 64-token rep after a 32k re-prefill | ~0.9 s of tokenize/HTTP overhead outside both server timers | `IMPLAUSIBLE` — a true number rejected |

1 and 2 are fixed above (`depth_ceiling`, the stderr drain). 3 and 4:

**Read back what the server actually did.** llama.cpp logs the values it
settled on (`llama_context: n_ctx = …`, `n_batch = …`, `n_ubatch = …`) and warns
when it turns a feature off (`swa_full is not supported by this model, it will
be disabled`, `speculative decoding not supported by this context`, `… not
supported, set to disabled`). The session's drained log tail is parsed once per
launch (`parse_server_effective`), compared with what the row asked for
(`server_adjustments`), and every difference goes in a new `adjusted` column.
A difference in a swept factor's own value — `ubatch` clamped, `swa_full`
disabled, speculation refused — makes the row `ADJUSTED`: its numbers are kept
(they are true) but it is out of the picks and the main effects, because it
measured a different level from the one it is filed under (the issue #16
principle). `batch` above the context is recorded but does not change the
status: a batch at least as large as the context already processes any prompt
the context can hold in one batch, so the level still means "as large as it
gets". llama-bench is not covered; it prints no such log.

**Credit the overhead the warm request measured.** The wall-clock cross-check
(I5) subtracts prefill from a rep's wall time and treats the rest as decode,
with a flat 2× margin for everything else. That margin is multiplicative and
the overhead is additive: tokenizing a 200 KB prompt costs the same ~1 s
whether 64 tokens or 1024 follow, so short generations after deep prompts were
rejected — 149 t/s reported, 51 t/s "allowed", 0.43 s of decode beside 0.87 s
of overhead. The warm request now measures that overhead directly (its wall
time minus both server timers) and each rep is credited it, capped at half the
rep's post-prefill time (`WALL_OVERHEAD_CREDIT_MAX`) so a server with a broken
counter cannot switch the check off. The defect class the check exists for
overshoots by 1000×; this changes nothing there.

## Invariants

- **SC1 — a narrowing that changes nothing must not look like one that does.**
  Any future cost dial narrows every competing factor or it is not a dial.
- **SC2 — `SLOW` is not `TIMEOUT` and not `IMPLAUSIBLE`.** Nothing hung, and the
  numbers are true; they are merely below a floor the user set. A `SLOW` row
  keeps its measurements and is excluded from the picks by the existing
  `status == "OK"` gates.
- **SC3 — the derived command is always shown before it is run.**
- **SC4 — non-interactive behaviour is byte-identical to before the interview.**
- **SC5 — a cost heuristic that reads an estimate keeps the row that survives the
  estimate being wrong.** The `ngl` grid biases toward full offload only on a
  positive fit verdict, and keeps `ngl=0` regardless, because a wrong "fits"
  would otherwise delete every row that could still have measured something.
- **SC6 — a time saving says when it is a bet.** C3a, C3c and C3e never change
  which rows reach `measured_ok`, and C3b only exits rows that are below the
  floor with a margin. C3d is the exception: a skipped row *might* have come
  back `OK` beside different partners. That is why it waits for measured
  evidence, records `SKIP_SLOW` rather than `SLOW` (nothing was measured), and
  has an off switch.
- **SC7 — a floor's budget covers every token the config is asked to process**,
  prefill and every concurrent stream included. A budget that omits work the
  config must do turns a fast config into a false `SLOW`.

## Open

- [ ] Streamed decode so `--tgs-timeout` judges after N seconds regardless of
      `n_gen` (C3, "Not done here").
- [ ] Time C3a–C3e on hardware: preheat seconds per config before/after, and
      how many rows `SKIP_SLOW` removes on a large model with the `ngl=0` anchor.

- [ ] `--levels` does not reach the 2-level arrays below L16 (`poll`,
      `batch_ratio` and `ffn_place` thin to 2, but `kv_type` is already floored
      by `--min-kv`). Probably fine; L16 is a screening design already.
- [ ] The interview does not ask about CPU offload tolerance (`ncmoe` /
      `ffn_place`), which is the one remaining item from the field feedback. It
      is a harder question to phrase than the other four — "how much are you
      willing to spill to CPU" is not something most users know in the abstract,
      and the sweep exists to answer it.
- [ ] Nothing yet reports, after a sweep, how much time the floors saved. That
      number would be the honest way to tune the advice.
