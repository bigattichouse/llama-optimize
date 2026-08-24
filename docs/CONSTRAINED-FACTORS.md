# Constrained (derived) factors — design

How `llama-optimize` sweeps parameters that are only *jointly* meaningful: pairs
carrying an implied ordering, like `-b >= -ub` or `n_min <= n_max`. The motivating
case is issue #8 — the Taguchi array happily emitted `--spec-draft-n-max 1
--spec-draft-n-min 2`. This document defines the problem in general terms, states
the invariants a correct design must hold, specifies the mechanism, and lays out
the test plan, so the next parameter of this shape needs only *data*, not code.

This is the sibling of [`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md), which
covers the other way a factor can fail to be independent: being *gated* by another
factor's value rather than *bounded* by it. That document's "Adjacent patterns"
section named this shape and deferred the choice; this document makes it.
Read [`DESIGN.md`](DESIGN.md)'s "Methodology recap" first for the DOE funnel both
plug into.

## The shape of the problem

A **constrained pair** is two tunables `A`, `B` where the objective is defined
across the whole grid `levels(A) × levels(B)`, but a sub-region of that grid is
*semantically degenerate*: the configuration is accepted, runs, and returns a
number that does not measure what the row claims to measure.

Three instances in the current registry:

| dependent | base | relation | flags |
|---|---|---|---|
| `spec_n_min` | `spec_n_max` | `n_min <= n_max` | `--spec-draft-n-min` / `--spec-draft-n-max` |
| `ngram_mod_n_max` | `ngram_mod_n_min` | `n_max >= n_min` | `--spec-ngram-mod-n-max` / `-n-min` |
| `batch` | `ubatch` | `-b >= -ub` | `-b` / `-ub` |

An orthogonal array varies its columns *independently* — that is the entire point
of it — so sweeping both members of a pair as absolutes **will** generate rows in
the degenerate region. With `spec_n_max ∈ {1,2,3,4,6}` and `spec_n_min ∈ {1,2}`,
the combination `(1, 2)` is one of ten, and an L125 draws it about a dozen times.

## Defect: an inverted row is mislabelled, not merely bad

The tempting reading is that an inverted row is *expected-poor data* — it scores
badly, main effects rank it last, no harm done. `ngram-design.md` originally took
that position. It is wrong, and the reason matters.

Verified against llama.cpp:

- `common/arg.cpp:4065-4077` parses `--spec-draft-n-max` and `--spec-draft-n-min`
  independently. Neither is validated against the other. Inverted bounds are
  **legal** and do not crash.
- Every drafting path caps the draft at `n_max` and then discards any draft
  shorter than `n_min`: `common/speculative.cpp:378` (draft model), `:1263`
  (MTP / DFlash / DSpark), `:1690`, `:1920` (ngram-mod). When `n_min > n_max` the
  draft can never reach `n_min`, so **every** draft is discarded.

So an inverted row does not run a *bad* speculative-decoding config. It runs with
**speculation entirely off** — and records `mtp=1`.

That is the defect. `mtp` is itself a swept factor (`["1","0"]`), so the inverted
row contributes a speculation-off measurement to the `mtp=1` level mean. The
damage is not confined to the row's own score; it biases the main effect of a
*different* factor toward "MTP does nothing". A knob whose whole purpose is to
answer "is MTP worth it on this box" is answered with contaminated data.

The same argument applies with less force to `ngram_mod_n_min`/`n_max`, where the
gate is pinned to `ngram-mod` for the whole tuning stage: there the contamination
stays inside the pair's own effects. It applies not at all to `-b`/`-ub`, where
llama.cpp clamps and the row is merely aliased. All three are still worth fixing
by one mechanism.

## Why the obvious repairs are worse

- **Clamp at emission.** Emit `--spec-draft-n-min 1` for a row recorded as
  `spec_n_min=2`. Now the CSV disagrees with the process that ran — the one thing
  a measurement tool must never do. It also silently aliases cells, so `n_min=1`
  gains rows and `n_min=2` loses them without anything saying so.
- **Drop inverted rows.** Costs a run *and* unbalances the array, which is the
  property the main-effect estimates depend on.
- **Shrink the level sets until inversion is impossible.** This is what `batch`
  did (floor of 2048, chosen so the `-b >= -ub` clamp never fires), and the cost
  is recorded: the entire low-batch regime became unreachable, so a real optimum
  at `-b 512 -ub 128` was outside the search space by construction. It also does
  not hold: `refine_factors` rebuilds each numeric grid independently per pass,
  so pass 2 can re-derive an inverting grid from a conflict-free pass 1.

## The invariants

- **C1 — validity by construction.** Every row of every generated design
  satisfies every declared relation. Not repaired into satisfying it: incapable
  of violating it.
- **C2 — orthogonality preserved.** No row is clamped, aliased, or dropped. The
  dependent factor's column stays a full, balanced design axis.
- **C3 — the record matches the run.** What the CSV holds is what reached
  llama.cpp: both the relative level swept and the absolute value it materialized
  to.
- **C4 — locality.** A derived factor's absolute value is a pure function of its
  own level and its base's level *in the same row* — no chains, no global state,
  so a row can be reproduced from the row alone.

## Mechanism: derive the dependent from the base

The dependent member of a pair stops being swept as an absolute. Its **levels
become relative to its base**, and its absolute value is materialized at emission
time from the base's value in the same row.

### 1. Declarative constraint in the registry

```python
"spec_n_min_frac": {..., "derived_from": ("spec_n_max", "scale"),
                    "relation": "at_most",  "abs_name": "spec_n_min"},
"ngram_mod_n_max_off": {..., "derived_from": ("ngram_mod_n_min", "offset"),
                        "relation": "at_least", "abs_name": "ngram_mod_n_max"},
"batch_ratio":      {..., "derived_from": ("ubatch", "scale"),
                     "relation": "at_least", "abs_name": "batch"},
```

Two operators cover every case so far:

| op | materialized value | levels are valid when |
|---|---|---|
| `scale` | `floor(level × base)` for `at_most`, `ceil(level × base)` for `at_least` | `level <= 1` / `level >= 1` |
| `offset` | `base + level` | `level <= 0` / `level >= 0` |

Rounding is not incidental. `round()` would break C1 at the bottom of the range:
banker's rounding sends `0.5 × 1` to `0` but `0.5 × 3` to `2`, and only `floor`
makes level `1.0` land exactly on the base. Hence floor for `at_most`, ceil for
`at_least`.

### 2. Relative level sets

| factor | levels | note |
|---|---|---|
| `spec_n_min_frac` | `0.0, 0.5, 1.0` | `0.0` is llama.cpp's own default (`common/common.h:326`) |
| `ngram_mod_n_max_off` | `0, 16, 32, 48, 64` | default pair (48, 64) is `n_min=48, off=16` |
| `batch_ratio` | `1, 4, 16` | over `ubatch ∈ 128..2048` this spans `-b 128..32768` |

The relative level is also the *better* design axis. "Require half the draft to
survive" is a question with an answer; "require 2 tokens" only means something
next to `n_max`. And `batch_ratio` restores the low-batch regime the old absolute
floor hid, without reintroducing a clamp.

### 3. Materialization at one choke point

`derived_value(name, f, cfg)` is the only place a relative level becomes an
absolute. `factor_flags` calls it for any factor carrying `derived_from`, which
is also where the old ad-hoc `if name == "batch": max(int(val), ub)` clamp used to
live — that clamp is now an instance of the mechanism rather than a special case,
and `factor_flags` no longer needs its `ub` parameter.

When the base is not itself part of the design, `DERIVED_BASE_FALLBACK` supplies
the fixed value that run will use for it (`cfg.spec_draft_n_max`, `-ub` 512,
`ngram_mod` `n_min` 48). This keeps C4 true for partial designs — a refinement
pass that settled `spec_n_max` to one level still derives correctly.

### 4. Both numbers in the CSV (C3)

A derived factor contributes **two** columns: its own, holding the relative level
(the axis `factor_level_means` computes over), and `abs_name`, holding the
materialized absolute. A row is therefore reproducible from the CSV alone.
`derived_abs_cols` leaves the absolute blank where a *conditional* derived factor
is inactive in that row — nothing was emitted for it, so recording a number would
be a small lie of exactly the kind C3 forbids.

### 5. The rename is part of the fix

`batch` → `batch_ratio`, `spec_n_min` → `spec_n_min_frac`, `ngram_mod_n_max` →
`ngram_mod_n_max_off`. Keeping the old names would mean an existing
`--factor batch=2048` silently becoming "2048 × ubatch" — the same species of
quiet misinterpretation this whole mechanism exists to remove. `RENAMED_FACTORS`
turns the old spelling into a targeted argparse error naming the new factor and
its units, rather than the generic "unknown factor" list.

## Registry and level validation (fail fast, not mid-sweep)

`validate_factor_registry` (asserted at `build_factors` time) additionally checks
that every derived factor's base exists, is numeric, is not itself derived (C4
forbids chains), has a `DERIVED_BASE_FALLBACK` entry, that `op` and `relation` are
recognised, and that the derivation graph is acyclic — reusing the same DAG walk
the `active_when` gate graph uses, now factored out as `has_cycle`.

`validate_factor_levels` is separate because it checks a *level set*, not the
registry: a `scale`+`at_most` factor's levels must all be `<= 1`, and so on. It
runs both on what `build_factors` produces and on every `--factor` override, so
`--factor spec_n_min_frac=2.0` fails at argparse rather than as an inverted row
three hours into a sweep.

## Test plan

In `selftest()`:

1. **Rounding table** for `derived_value` across `(n_max, frac)`, including the
   edges that motivated floor/ceil: `n_max=1, frac=0.5 → 0`; `frac=1.0 → n_max`
   exactly; and `assert got <= n_max` on every case.
2. **Both operators**: `offset` reproduces llama.cpp's default pair (48, 64).
3. **Base not swept**: derivation falls back to the run's fixed value.
4. **Inactive conditional derived factor** records an empty absolute.
5. **Registry validation** rejects a non-numeric base and a derived cycle;
   **level validation** rejects `spec_n_min_frac=2.0`, `batch_ratio=0`,
   `ngram_mod_n_max_off=-16`.
6. **The property test that closes #8**: build the real MTP factor set and the
   real `ngram-mod` tuning stage, run them through `generate_runs`, materialize
   every derived factor in every row, and assert the declared relation on all of
   them. This is the check a level-set-only fix could not have provided, since
   `refine_factors` rebuilds grids per pass.

## Prior art

Constrained / conditional search spaces are standard in hyperparameter
optimization, and reparameterizing to make a constraint structural — sweeping a
ratio or an offset rather than two absolutes with a forbidden region — is the
usual answer where the constraint is a simple ordering. It preserves the
sampler's balance guarantees instead of asking a rejection or repair step to
paper over their loss. The alternative families (rejection sampling, repair
operators, penalty terms) all trade either sample budget or fidelity of the
record, which C1–C3 rule out here.

## Postscript: pinning an unswept base

C1 is a claim about what reaches llama.cpp, not about what the row says. If a
derived factor is swept while its base is not, the derivation used
`DERIVED_BASE_FALLBACK` — but llama.cpp would apply *its own* default for the
base, and the two need not agree. `--factor spec_n_min_frac=1.0` with
`--spec-draft-n-max 8` on a design that does not sweep `spec_n_max` would compute
`n_min = 8` against our fallback while llama.cpp used its default `n_max = 3`:
inverted again, by a different route.

So `factor_flags` pins any such base explicitly, at the value the derivation
assumed (`derived_base_pins`). The MTP emitter skips its own
`--spec-draft-n-max` line when that pin already covers it, so the flag appears
exactly once.

## Verified on hardware

`gemma-3-270m-it-Q8_0` on a gfx906 (ROCm), 12 live runs, all `OK`.

**The pair from issue #8.** An L8 forcing `spec_n_max=1,2` × `spec_n_min_frac=0.0,1.0`
alongside `batch_ratio=1,4`: `spec_n_min <= spec_n_max` and `batch >= ubatch` held in
all 8 rows. The array cell the ticket reported — `spec_n_max=1` crossed with the top
level of the min factor — materialized to `n_min = 1`, and one row emitted
`-b 128 -ub 128`, in the low-batch regime the old 2048 floor could not reach.

Replaying the *identical* design through the pre-fix `build_factors`/`generate_runs`
produces `--spec-draft-n-max 1 --spec-draft-n-min 2` in **2 of 8** rows. Same design,
same forced levels: 2 inverted before, 0 after.

**The ngram-mod pair.** Four runs with the gate pinned to `ngram-mod`, sweeping
`n_min=16,96` × `n_max_off=0,32`, materialized to `(16,16) (16,48) (96,96) (96,128)` —
zero violations, including both `off=0` rows where `n_max` lands exactly on `n_min`.
Under the old absolute levels `n_min=96` crossed with `n_max ∈ {16,48}` would have
been two inverted rows out of four. This sweep also exercised a live drafting path
(678–932 t/s against 84–300 t/s without ngram), and its winner was `n_min=96` — a
region half of whose rows used to be inverted.

Caveat on scope: gemma-3-270M ships no MTP/NextN head, so in the first sweep the
`--spec-draft-n-*` flags were emitted and accepted but governed no drafting. That
sweep validates design and emission, not the contamination itself, which needs an MTP
model. The second sweep exercises the mechanism against a drafting path end to end.
