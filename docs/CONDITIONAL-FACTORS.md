# Conditional (nested) factors — design

How `llama-optimize` sweeps parameters that are only meaningful when *another*
parameter takes a particular value. The motivating case is ngram
self-speculative decoding (`--ngram`): the `--spec-type` variant selects one of
several ngram algorithms, and each algorithm has its own tuning knobs that mean
nothing to the others. This document defines the problem in general terms,
states the invariants a correct design must hold, specifies the mechanism, and
lays out the test plan — so the same machinery covers the next parameter of this
shape (rope/yarn, tensor-split mode, NUMA modes, …) without new special cases.

This is a companion to [`DESIGN.md`](DESIGN.md); read the "Methodology recap"
there first — the DOE funnel (Morris screen → Taguchi array → `--iterate`
refinement → confirm) is the machine this design plugs into.

## The shape of the problem

A **conditional factor** `P` is a tunable whose effect on the objective exists
only when a **gate factor** `G` resolves to a value in some set `S_P ⊆ levels(G)`.
Outside `S_P`, `P` is **inert**: with correct emission it does not change the
launched command, and therefore cannot change the measurement.

ngram is the first instance:

| Gate `G` | value | conditional children (active only for that value) |
|----------|-------|-----------------------------------------------------|
| `ngram` (`--spec-type`) | `none` | *(none)* |
| | `ngram-simple` | `size-n`, `size-m`, `min-hits` |
| | `ngram-map-k` | `size-n`, `size-m`, `min-hits` |
| | `ngram-map-k4v` | `size-n`, `size-m`, `min-hits` |
| | `ngram-mod` | `n-match`, `n-min`, `n-max` |

The pattern — **one mode selector plus mode-specific knobs** — recurs across
llama.cpp: `--rope-scaling yarn` gates the `--yarn-*` family; `-sm row` gates
`-ts` tuning; a future multi-GPU or NUMA mode selector will gate its own knobs.
The design below is written against the general pattern, with ngram as the
worked example, precisely so those later cases need only *data*, not code.

## Why a flat orthogonal array is wrong here

The tool's core is a Taguchi orthogonal array (OA): pick balanced level
combinations so every factor's main effect reads cleanly in ~25 runs. Folding
conditional factors into one flat OA breaks in four distinct ways.

**F1 — Illegal or ignored flags (emission).** A flat OA assigns `P` a level in
every row, including rows where `G ∉ S_P`. The command builder then emits `P`'s
flag while the active mode is a different one — e.g. `--spec-ngram-simple-size-n`
while `--spec-type ngram-mod` is running. Best case llama.cpp ignores it (wasted
column); worst case it rejects the unknown-for-mode option and the row is scored
as a crash, injecting a **false OOM** into the Pareto frontier and the
main-effects table. This is the correctness bug.

**F2 — Array inflation.** OA size is a step function of the *varying*-factor
count. At five levels the tables are:

```
L25  ≤ 6 factors
L125 ≤ 31 factors
L625 ≤ 156 factors
```

The base sweep (`ngl`, `n_depth`, `threads`, `kv_type`, `ubatch`, …) sits around
six factors — an **L25**. Adding the ngram gate plus `spec_n_max` plus ~12
per-variant knobs pushes past six, so the binding jumps to **L125 — 125 model
loads instead of 25**, five times the wall-clock, the bulk of it spent varying
knobs that are inert in the row that varies them. (Cf. `ROADMAP.md §2`: "at L125
with big models that's 20+ minutes of known-doomed runs.")

**F3 — Biased main effects (analysis).** `refine_factors` ranks a factor by
`factor_level_means(rows, P)`: the mean objective grouped by `P`'s level. When
`G ∉ S_P` in most rows, each level-mean of `P` is dominated by rows where `P`
did nothing, so the between-level spread is noise. Refinement then either
settles `P` at a spurious "winner" or — because the spread looks tiny relative
to real knobs — settles it at `levels[0]`, a value chosen essentially at random.
**The tool reports a tuned parameter it never actually tuned.**

**F4 — Broken orthogonality.** An OA's whole guarantee is *balance*: every level
of every factor co-occurs with every level of every other an equal number of
times, so main effects separate. A conditional factor's column is balanced
against `G` overall, but *live* only in the `S_P` cells — within the live subset
it is no longer balanced against the other factors, and the separability the
analysis assumes no longer holds. Conditioning the analysis to live rows (see
F3's fix) removes the bias but leaves too few live rows per level (25 of 125 for
a one-of-five gate) — and those rows are not themselves a balanced sub-array — to
estimate cleanly. The array structure has to be right, not just the arithmetic
on top of it.

These are four faces of one violated invariant.

## The invariants

> **I1 — Liveness.** No orthogonal-array column may be inert in any of its rows.
> A factor may appear in an OA only over rows in which it is active.

> **I2 — Emission.** No launched command may carry a flag for a factor that is
> inactive in that row.

> **I3 — Honest estimation.** No factor's effect may be estimated from rows in
> which it was inactive.

> **I4 — Pinned gates.** Any design that *sweeps* a conditional factor must hold
> its gate at a single value (a constant) across that design.

I1 is the generalization of a rule the codebase already follows for *pinned*
factors: `choose_array`/`generate_runs` drop settled single-level factors from
the array because "constants carry no information for an orthogonal array"
(`DESIGN.md`, "Don't run a 25-row array to sweep one factor"). A conditional
factor that is inert in a row is a *local* constant — the same principle, applied
per-row rather than per-design.

## Mechanism

Two pieces, both driven off declarative metadata so no factor gets bespoke code.

### 1. Declarative applicability in the registry

`FACTORS` is already "the one place to add a tunable." Extend an entry with an
optional `active_when` clause:

```python
"ngram_mod_n_match": {
    "bench": None, "server": ("--spec-ngram-mod-n-match",),
    "kind": "num", "server_only": True,
    "active_when": ("ngram", {"ngram-mod"}),          # gate factor, live values
},
```

A factor with no `active_when` is unconditional — today's behavior, unchanged.

Where several gate values share a knob that differs only in flag spelling, use
one logical factor with a `flag_for` resolver instead of one factor per spelling.
This collapses the nine simple/map-k/map-k4v knobs to three:

```python
"ngram_size_n": {
    "bench": None, "kind": "num", "server_only": True,
    "active_when": ("ngram", {"ngram-simple", "ngram-map-k", "ngram-map-k4v"}),
    "flag_for": lambda v: (f"--spec-{v}-size-n",),    # v is the gate value
},
```

All downstream logic reads this metadata through two pure functions:

```python
def is_active(name: str, assignment: dict) -> bool:
    """True if factor `name` participates under the given factor→value
    assignment (a run row, or a pinned config). Unconditional factors are
    always active; a conditional factor is active iff its gate is present in
    the assignment and its value is in the factor's live set."""

def active_factors(names: Iterable[str], gate_assignment: dict) -> list[str]:
    """The subset of `names` active under `gate_assignment`."""
```

`is_active` is the single source of truth for I1, I2, and I3. It replaces the
ngram-specific prefix hack (`_ngram_param_inactive`) introduced as the first-cut
emission gate — that function is deleted and `factor_flags` calls `is_active`.

### 2. The stage planner

Given the factor set plus `active_when`, decompose the design into an ordered
sequence of **stages**, each a *flat* OA whose every factor is active in every
row (I1 by construction). This is the reusable core.

The design must **preserve the search intent**: the original request was to
explore *every* variant and its parameters and return the best. A naive greedy
plan — pick the single best gate value at default params, then tune only it —
quietly shrinks that search: a variant that is mediocre at defaults but excellent
once tuned would never get tuned. So the planner *screens* gate values rather than
committing to one, exactly mirroring the tool's existing Morris-screen → refine
funnel (screen knobs, drop the clearly-unimportant, refine the survivors):

- **Stage 0 — gate screen.** All unconditional factors plus every *gate* factor
  (any factor named in some other factor's `active_when`). Conditional children
  are excluded; each candidate gate value launches with its documented **default**
  child parameters. ~6 factors → **L25**. Outcome: every gate value *measured* at
  defaults, plus tuned unconditional knobs. This ranks the variants; it does not
  yet choose one.

- **Keep the contenders, not just the argmax.** Carry forward every gate value
  within a margin of the best (policy: keep top-`K`, default `K ≥ 2`, or "all
  within `x%` of the leader"). Only variants that are *clearly* dominated at
  defaults are dropped — the same "don't prematurely discard" caution the Morris
  screen already applies to knobs. A budget knob (`--ngram-fast` ⇒ `K=1`) is
  available for users who explicitly want the cheap greedy path.

- **Stage k — child tuning (one per kept gate value).** For each surviving gate
  value, pin the gate to it (a constant) and sweep exactly that value's children;
  every factor is active in every row → a clean flat OA. Unconditional knobs are
  held at their Stage-0 winners so the runs are *about the children*. ngram-mod
  tuning is `n-match × n-min × n-max` → 3 factors → **L25**.

- **Selection off measured evidence.** The final pick is the best *measured*
  config across Stage 0 and all Stage-k branches — the tool already chooses from
  the measured Pareto frontier rather than trusting a main-effects extrapolation
  (`DESIGN.md`), so a variant that only wins once tuned still wins on its merits.

For ngram with the default `K=2` that is **~25 + 2·25 ≈ 75 runs, every one a
fully-live data point**; tuning all five variants is ~150. Either way F2/F3/F4
dissolve — no stage ever contains an inert column — and the *breadth* the author
asked for is preserved: the full variant set is measured, and every contender is
actually tuned, which the flat L125 never did (it diluted all of them at once).

The planner generalizes cleanly:

- **Multiple independent gates** → each selected in Stage 0, each spawning its
  own child-tuning stage (or a shared one when the combined child count still
  fits an L25).
- **Nested gates** (a child that is itself a gate) → ordered by the gate
  dependency graph via topological sort. That graph **must be a DAG**; a cycle is
  a registry error caught at validation (I4's structural precondition).

### Riding the existing funnel

Staging is not a parallel execution path; it lives on the `--iterate` machinery
already in `run_iterations`.

- The gate factor is swept in the first Taguchi pass (Stage 0). `refine_factors`
  already narrows a categorical factor toward its top levels across passes — it
  keeps the top few, not just the argmax, which is exactly the "keep the
  contenders" behavior we want. We add one rule to the refinement transition:
  **when a gate factor narrows to its kept set, the next pass opens the children
  of each kept value** — they enter the design at the pass where they first become
  tunable, and thereafter refine and settle like any numeric knob. When several
  gate values survive, their child sets are disjoint by construction (each child's
  `active_when` names one value), so a pass can carry them as parallel branches or
  the funnel can tune them in sequence; the merged measured rows decide the winner.
  This is the whole integration: today `refine_factors` only *narrows*; it gains
  "and open the children of the surviving gate values."

- **Single pass** (`--iterate 1`, the default) reduces to Stage 0 alone: every
  variant is *measured* at default params and the best-measured is reported, but
  no children are swept — so no conditional factor ever enters a flat array
  (correct by construction). The report states plainly that this ranked variants
  at defaults and that per-variant *tuning* needs `--iterate ≥ 2`, so a single-pass
  user is never misled into thinking the knobs were optimized.

- **`--ngram-type <variant>`** is the escape hatch: pin the gate up front and run
  a Stage-k tuning standalone in a single pass to tune one variant's knobs
  directly. This is also how the general mechanism exposes "tune the children of a
  known mode" for any future gate.

- **Morris screen** (`--screen`) only screens a conditional factor once its gate
  is pinned — before that it is not a live knob. Same rule, so the screen stays
  honest.

### Emission gating and analysis conditioning as invariant guards

With staging, an inactive conditional factor should never reach the command
builder or the analyzer. Both still enforce their invariant directly — defense in
depth, and coverage for the fixed-default path and for hand-authored or
`--merge-results` rows that did not come from a planned stage:

- **`factor_flags`** skips any factor where `not is_active(name, row)` (I2).
- **`factor_level_means`** restricts to rows where `is_active(name, row)` (I3). A
  factor active in *zero* rows yields **no estimate** — it is not refined and
  settles to its documented default, flagged as untuned, rather than being
  assigned a fake zero-effect winner.

## Registry validation (fail fast, not mid-sweep)

A bad `active_when` should fail at startup, not after an hour of GPU time. A
validator runs in `selftest` **and** as an assertion when `build_factors`
assembles the set:

1. Every `active_when` gate names a real key in `FACTORS`.
2. Every value in `S_P` is a producible level of that gate.
3. `flag_for` returns a non-empty flag tuple for every value in `S_P`.
4. The gate dependency graph is acyclic (topological sort succeeds — I4).

## Test plan

The tool's tests are assert blocks inside `selftest()`, exercised by the CI
action in `.github`. All of the following are **pure-Python and need no GPU** —
they run offline while the card is busy. Layered from cheapest to broadest:

**Registry validation.** Assert the four rules above hold for the real registry;
assert a deliberately broken synthetic registry (missing gate / bad value /
`flag_for` gap / introduced cycle) is *rejected*. Catches authoring errors as
data errors.

**`is_active` truth table.** Unconditional factor → always `True`. Conditional
factor → `True` iff gate present and in `S`; `False` for gate = `none`, gate =
other value, and gate key absent from the assignment.

**Emission gating (table-driven over gate × factor).** For each gate value assert
the emitted ngram flag set equals exactly the active children (minus their fixed
defaults) and excludes every inactive one. Includes the collapsed `flag_for`
spelling: `ngram_size_n` under `ngram-map-k4v` emits `--spec-ngram-map-k4v-size-n`,
under `ngram-mod` emits nothing. `ngram=none` emits no ngram flags at all. The
existing MTP-plus-ngram `--spec-type` merge tests are retained unchanged.

**Stage planner (the heart).** On a synthetic registry — one gate `G` (3 values)
+ unconditional `{A, B}` + children on two of `G`'s values — assert Stage 0 =
`{A, B, G}` with no children, and that pinning each winner activates exactly that
value's children with `G` constant. Add the **liveness property test** (I1 as an
executable check):

```python
for stage in plan_stages(factors):
    for row in generate_runs(stage.factors, choose_array(stage.factors))[1]:
        assert all(is_active(f, row["factors"]) for f in stage.factors)
```

Plus a two-independent-gates case, a nested-gate ordering case, and a
cycle-raises case.

**Array sizing (F2 witness).** `choose_array(stage0_factors) == "L25"` for the
ngram config — explicitly *not* `"L125"` — and the summed run count across stages
is strictly less than the flat design's, pinning the 125→~50 win as a regression
guard.

**Analysis conditioning (F3 witness).** Build synthetic rows where, within
`G = ngram-mod`, `ngram_mod_n_match` has a clear monotone effect peaking at 32,
while rows with `G ≠ ngram-mod` carry decoy `n_match` values whose apparent best
is a *different* level. Assert the conditioned `factor_level_means` recovers 32
**and** that the unconditioned version recovers the decoy — demonstrating both
that the bug is real and that the fix removes it. Separately assert a
zero-live-rows factor produces no estimate and settles to its default.

**Refinement transition (stage hop).** Feed `refine_factors` a pass-1 result set
where `ngram-mod` wins; assert the returned factor set has `ngram` settled to
`ngram-mod` **and** `ngram_mod_*` newly present at documented levels, while no
non-winning variant's children ever appear. Then assert a subsequent refine
settles those children numerically (existing `refine_numeric` behavior).

**End-to-end plan (offline).** A dry-run assertion that dumps the staged command
sequence for a representative `--ngram --iterate 3` config and checks its shape
(Stage-0 variant sweep, Stage-k child tuning), covering the whole path without a
server.

**Regression.** The full pre-existing `selftest` stays green; the ctx-floor and
`--spec-type` merge tests are untouched.

## Rollout

1. Land the declarative core — `active_when`, `flag_for`, `is_active`,
   `active_factors`, and the registry validator — with no behavior change for
   existing unconditional factors.
2. Convert ngram to the metadata; delete `_ngram_param_inactive`; point
   `factor_flags` and `factor_level_means` at `is_active`.
3. Add the stage planner and the one-line refinement transition; keep it behind
   the existing `--ngram` / `--iterate` surface. Add `--ngram-type` as the
   single-pass Stage-k escape hatch.
4. Update the README ngram section, cross-link this document from `DESIGN.md`,
   and tick the ROADMAP item.

## Adjacent patterns (same family)

Two related shapes surface alongside conditional factors; the registry should
name them rather than let them slip in as flat factors:

- **Globally inapplicable factor.** A knob wired to a mode it does not affect at
  all — e.g. the ngram work found `--spec-draft-n-max` (a *draft-model* knob)
  being swept "for ngram", where it influences no ngram variant. This is the
  degenerate conditional factor: an empty live set for the gate in question.
  Variant-gating does **not** catch it (it is not tied to any one variant); the
  fix is not to add the factor to that design at all. The guard is the same
  registry validation — a factor's `server`/`flag_for` mapping must actually
  reach the code path the design exercises.

- **Sibling ordering constraint.** Two factors with an implied relation
  (`n_min ≤ n_max`). An OA varies them independently, so it will emit inverted
  rows. These are not conditional-on-a-gate; they are a *joint* constraint. Options:
  derive one from the other (sweep `n_max` and an offset), clamp at emission, or
  accept inverted rows as expected-poor data (never as crashes). Flag which, so a
  low score on an inverted row is not misread as a real effect.

## Prior art

Conditional / hierarchical search spaces are standard in hyperparameter
optimization — a knob that exists only under a particular choice of a parent knob
(e.g. a kernel's parameters existing only when that kernel is selected). Mature
optimizers model them explicitly rather than flattening them: nested choice
spaces, per-branch parameters, and staged/tree-structured search. This design is
the orthogonal-array equivalent: a gate factor with declared live sets, and a
per-branch flat OA once the gate is fixed. The invariants above are just the
concrete form those tools' "a parameter is only sampled when its condition holds"
rule takes in a Taguchi setting.
