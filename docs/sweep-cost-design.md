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

## Invariants

- **SC1 — a narrowing that changes nothing must not look like one that does.**
  Any future cost dial narrows every competing factor or it is not a dial.
- **SC2 — `SLOW` is not `TIMEOUT` and not `IMPLAUSIBLE`.** Nothing hung, and the
  numbers are true; they are merely below a floor the user set. A `SLOW` row
  keeps its measurements and is excluded from the picks by the existing
  `status == "OK"` gates.
- **SC3 — the derived command is always shown before it is run.**
- **SC4 — non-interactive behaviour is byte-identical to before the interview.**

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
