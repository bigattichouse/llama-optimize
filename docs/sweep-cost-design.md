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
not merely lose, they **abort**. On a hybrid SSM model every *partial* offload
core-dumps `llama-server`, right after:

```
resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
  (chunked) is assigned to device ROCm0 (usually due to missing support)
```

Measured on `Qwen3.6-27B-Q5_K_M` (`qwen35`, 64 blocks) at `-ngl 5` **and**
`-ngl 40`; `-ngl 0` runs, and `-ngl 99` runs. It is the split that kills it, not
the amount — the recurrent half of the model ends up straddling two devices.

So on `ssm_state > 0` the grid is `[0, 99]`: the only two placements that
execute. Three of five default levels would otherwise be recorded as `SIGNAL`,
which is honest but useless, and on a first run it looks like the tool is broken
rather than the placement being impossible.

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

## Open

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
