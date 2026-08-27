# Concurrency and the unified KV cache — design & work log

Concrete plan for making `--parallel` and `kv_unified` describable together. The
principle it leans on — that two knobs llama.cpp couples cannot ride an
orthogonal array as free columns — is
[`CONSTRAINED-FACTORS.md`](CONSTRAINED-FACTORS.md); this file is the
concurrency-specific model and checklist.

## Origin

[`flag-coverage.md`](flag-coverage.md) C2. `-kvu`/`--kv-unified` documents its
default as "enabled if number of slots is auto", which sounded like a footnote
and is not.

## The mechanism, verified

Three facts, each checked against llama.cpp `4d19b2876`:

1. **`llama-server` defaults slots to auto.** `common/arg.cpp:1400` sets
   `params.n_parallel = -1` for `LLAMA_EXAMPLE_SERVER` specifically — the struct
   default elsewhere is `1` (`common/common.h:447`).
2. **Auto means 4 slots *and* unified KV.** `tools/server/server.cpp:151-155`:
   `if (params.n_parallel < 0) { params.n_parallel = 4; params.kv_unified = true; }`.
   Nothing else sets `kv_unified`; its struct default is `false`
   (`common/common.h:563`).
3. **Any explicit `--parallel` disables unified KV — including `--parallel 1`.**
   The branch tests `< 0`, so passing the number that *matches* the auto result
   still changes the KV regime.

Confirmed live: `llama-server` with no `--parallel` logs
`n_slots = 4, kv_unified = 'true'`.

## Defect

`build_server_args` emits `--parallel` only when `cfg.parallel > 1`. So the tool
runs in two different KV regimes depending on the use case, and has never
measured either one from the other side:

| use case | `--parallel` emitted? | slots | `kv_unified` |
|---|---|---|---|
| `app`, `single` (parallel 1) | no | **4** (auto) | **true** |
| `agents` (4), `multi-user` (8) | yes | as asked | **false** |
| `parallel` swept as a factor | yes, every row | as asked | **false** |

Two problems, and they point in opposite directions.

**The single-stream case measures something other than what it says.** A
`--use-case single` sweep is described as "one user, one slot" and actually
allocates four slots with a shared KV cache. The measurement still issues one
request at a time, so the throughput number is not wrong — but the memory
footprint and cache behaviour are those of a 4-slot server, and the OOM pruner is
reasoning about a configuration nobody asked for.

**The concurrency case can never reach the default.** Every `--parallel` row runs
`kv_unified = false`, while a user who starts `llama-server` without `--parallel`
gets `true`. The regime llama.cpp actually ships is outside the search space.

This is not a wrong *recommendation*: the emitted command carries the same
`--parallel` the sweep used, so what the user runs is what was measured. It is a
truncated *space* — the tool cannot answer a question that has a real answer.

## Why this is a constrained pair, not two columns

The obvious fix — add `kv_unified` as a boolean factor — produces rows that lie.
`--parallel 4 --kv-unified` is constructible on the command line, but "4 slots
with unified KV" reached that way is not the same configuration as auto: it is
the *only* way to get the pair without also accepting llama.cpp's slot count, and
`--parallel N --no-kv-unified` for N=4 is a third thing again. Meanwhile
`parallel = auto` cannot be expressed as a level of a numeric factor at all.

So the honest model is one factor over the *combinations that exist*:

**K1 — `concurrency` as a categorical whose levels are (slots, unified) pairs.**
Levels are pre-rendered from what llama.cpp can actually be put into:

- `auto` — emit nothing: 4 slots, unified. **The default, and currently unreachable.**
- `N` — `--parallel N`: N slots, split KV (today's behaviour)
- `N-unified` — `--parallel N --kv-unified`: N slots, shared cache

The level *set* is generated from the use case's concurrency the way `ngl_levels`
derives from the model, rather than being fixed — a `multi-user` sweep wants
levels around 8, a `single` sweep wants `1` and `auto` and little else.

**K2 — `--parallel 1` must stop being special-cased.** The current
`if cfg.parallel > 1` is what silently routes single-stream sweeps into auto. Once
`auto` is an explicit level, "one slot" and "let llama.cpp decide" are different
levels and the emission follows the level, not a threshold.

## Interactions the design must settle

**K3 — per-sequence context is not the same in both regimes.** A slot gets
`llama_n_ctx_seq(ctx)`, which is the full `-c` under unified KV and a share of it
when split. `ServerSession` computes `n_ctx` as `prompt + gen + 256`, multiplied
by `par` only when `par > 1`. Under `auto` that multiplication does not happen
while four slots exist, and under split KV the per-slot context is smaller than
the arithmetic assumes. Any `concurrency` level therefore has to carry its own
`n_ctx` rule, or the `ctx_floor` guarantee means different things per row — which
would be the batch-floor defect again, in a new place.

**K4 — the OOM pruner sees a different footprint per regime.** A shared cache and
N private caches do not cost the same, and `predict_fits` currently reasons about
neither. The pruner must take the regime as an input, and must stay conservative:
a wrongly-skipped viable config is worse than a wasted OOM run.

## Implemented, and verified against llama.cpp's own log

`concurrency` is a categorical over the states llama.cpp can actually be put
into. Every level was checked by launching a real server and reading what it
reported, rather than by reasoning from the source:

| level | flags emitted | llama.cpp reports | K3 multiplier |
|---|---|---|---|
| `auto` | *(none)* | `n_slots = 4, kv_unified = 'true'` | 1 |
| `1` | `--parallel 1` | `n_slots = 1, kv_unified = 'false'` | 1 |
| `4` | `--parallel 4` | `n_slots = 4, n_ctx_slot = 512, kv_unified = 'false'` | 4 |
| `4u` | `--parallel 4 --kv-unified` | `n_slots = 4, n_ctx_slot = 2048, kv_unified = 'true'` | 1 |

Two things this confirms that were previously read out of the source:

**`--parallel 1` really does disable unified KV.** Not a quirk of the branch as
written — llama.cpp logs `kv_unified = 'false'` for it, while emitting nothing at
all gives `'true'`. Asking for the number that matches the default still changes
the regime.

**K3 is settled empirically.** At `-c 2048` the split regime gives each slot
`n_ctx_slot = 512` — exactly 2048/4 — and the unified regime gives 2048, the
whole thing. So `n_ctx = per_slot x (slots if split else 1)`, which is what
`ctx_slots_multiplier` implements.

A pleasant discovery while settling it: the existing `if par > 1: n_ctx *= par`
was already correct in all three reachable cases, because the multiplication
happened to coincide with exactly the split regime. The rule is now explicit
rather than a coincidence, which is what makes the `auto` level safe to add.

## Invariants

- **KV1 — a row's emitted flags determine its regime, with no thresholds.**
  No `> 1` special cases: the level says what to emit.
- **KV2 — the recorded configuration is the one that ran.** `kv_unified` is
  derivable from the emitted flags for every row, and appears in the results CSV
  as an absolute column the way derived factors already do (C3).
- **KV3 — `auto` is reachable.** If the level set cannot express llama.cpp's own
  default, the sweep cannot tell a user whether to keep it.
- **KV4 — the context floor means the same thing in every row.** Whatever K3
  settles, `ctx_floor` is a per-slot guarantee or a whole-server one, stated once
  and applied identically.

## Open questions

1. Is `N-unified` worth a level, or only `auto` and `N`? Unified KV with an
   explicit slot count is the combination with no default behind it, so it may be
   the one users never want — but it is also the only way to separate "unified"
   from "4 slots", which is what makes the auto result interpretable.
2. Should `concurrency` replace the `parallel` factor outright, or sit beside it
   with `parallel` deprecated via `RENAMED_FACTORS`? The rename machinery exists
   and was built for exactly this (`spec_n_min_frac` and friends), so the second
   is nearly free and keeps old `--factor parallel=...` invocations working.
3. Does `--kv-unified` interact with `-sps`/`--slot-prompt-similarity`? Slot
   selection and a shared cache both decide what gets reused; if they are not
   separable, `-sps` belongs here rather than in
   [`workload-shape-design.md`](workload-shape-design.md).

## Checklist

- [ ] `concurrency` categorical factor with `(slots, unified)` levels, generated
      from the use case's concurrency (K1)
- [ ] Emission follows the level; delete the `cfg.parallel > 1` threshold (K2)
- [ ] `kv_unified` recorded as an absolute column per row (KV2)
- [ ] Settle and implement the per-regime `n_ctx` rule (K3/KV4)
- [ ] `predict_fits` takes the regime as an input (K4)
- [ ] `parallel` kept working via `RENAMED_FACTORS` (open question 2)
- [ ] `--selftest`: every level emits flags that reproduce its own regime, `auto`
      emits no `--parallel`, no level emits a threshold-dependent result, and the
      recorded `kv_unified` matches the flags for every level — all pure
      functions over the factor grid, no GPU

## Testing without a long sweep

Everything above except the throughput itself is a statement about which flags a
level emits and what regime they produce, which is checkable with no binary — the
same technique that makes the issue-#8 grid check trustworthy.

The one thing that needs hardware is whether unified KV actually differs in
throughput or memory at a given slot count. That is a handful of runs rather than
a design, and it is the measurement this whole entry exists to make possible.
