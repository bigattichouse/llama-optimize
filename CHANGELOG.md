# Changelog

Notable changes to `llama-optimize`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Why this file has an extra section

Most changelogs answer "what changed?". A measurement tool has to answer a
second question: **"does this change what my existing results mean?"**

A fix that makes a measurement more honest also makes every earlier measurement
less trustworthy, and nothing in the standard Added/Changed/Fixed categories says
so. Filing "ngram throughput was inflated 2.3-3.4x" under *Fixed* would be true
and useless. So each release carries an **Affects existing results** section
naming which past numbers moved and whether to re-run. If a release has none,
it says so explicitly rather than leaving you to infer it.

Results CSVs are stamped with `tool_version` and `llama_build` from 0.2.0 onward,
so a file found later can be matched against this list without relying on memory.
Files produced before 0.2.0 carry no stamp — that absence dates them to 0.1.0 or
earlier.

## [Unreleased]

### ⚠️ Affects existing results

- **ngram sweep results are upper bounds, not estimates.** llama.cpp's n-gram
  speculation keeps state *across requests*, and the sweep sends one identical
  prompt for every rep. Measured 2026-08-26 (gemma-3-270m-it-Q8_0, ROCm,
  `--spec-type ngram-mod`): the first request drafts nothing, every subsequent
  one reaches **100% draft acceptance**, and throughput goes 273 -> 601 t/s. On
  genuinely distinct prompts the drafter never fires at all (176-259 t/s).

  Since the harness discards the first request as warmup, *every measured rep*
  sat in the saturated regime. Two consequences: ngram-vs-off is inflated
  roughly **2.3-3.4x**, and the variant screen has been ranking `ngram-simple`,
  `ngram-mod`, `ngram-map-k` and `ngram-map-k4v` at a shared ceiling where the
  differences it exists to resolve cannot appear.

  **Re-run any sweep that used `--ngram`.** Unaffected: `ngl`, `threads`,
  `ubatch`, `batch`, `kv_type`, `nkvo`, `poll`, `ot` — these do not depend on
  what the tokens say. MTP drafts from the model's NextN head rather than from
  context n-grams and is very likely fine, but this has not been measured and
  should not be assumed. Fix in progress:
  [`docs/workload-shape-design.md`](docs/workload-shape-design.md).

- **Concurrency sweeps have never measured `kv_unified = true`.** llama.cpp
  enables the unified KV cache only when slot count is automatic
  (`tools/server/server.cpp:153`); the struct default is `false`. This tool
  always passes `--parallel` explicitly, so every `--parallel` sweep ran with
  unified KV *off* — while a user starting `llama-server` with no `--parallel`
  gets it *on*, with 4 slots. The emitted command is self-consistent, so this is
  not a wrong recommendation; it is an unexplored regime, and the one the default
  lands in. No factor currently reaches it. See
  [`docs/flag-coverage.md`](docs/flag-coverage.md) C2.

### Added

- `draft_acc` column: the fraction of drafted tokens llama.cpp accepted, for
  every run where speculation can happen. Sourced from `draft_n` /
  `draft_n_accepted`, which llama.cpp already returned in the same `timings`
  block the throughput rate is read from.
- `spec_off` flag: set when a run asked for speculation and llama.cpp drafted
  nothing. This is the [issue #8] shape — a row recording `mtp=1` that silently
  measured the baseline. It is a flag, never a status: the measurement is real,
  it simply is not measuring what its factor column claims.
- Loud warning when llama.cpp reports no GPU but a vendor tool sees one. A build
  without a GPU backend does not fail; it runs on the CPU and reports plausible
  numbers, making every `-ngl` level the same run. Observed at **3.9x** (115 vs
  444 t/s on the same model) after a stale build directory silently left
  `GGML_HIP=OFF`. CPU-only *machines* are a supported case and get a one-line
  note instead of the alarm.
- `tool_version` and `llama_build` columns stamped on every results row, so a
  CSV can be traced to what produced it after the fact.
- Fourteen previously unreachable llama.cpp knobs registered and usable via
  `--factor`: `load_mode`, `no_op_offload`, `no_host`, `repack`, `swa_full`,
  `backend_sampling`, `prio`, `prio_batch`, `ctx_checkpoints`,
  `checkpoint_min_step`, and the batch-phase CPU affinity set (`cpu_mask_batch`,
  `cpu_range_batch`, `cpu_strict_batch`, `poll_batch`). Deliberately **not**
  auto-swept — a registry entry costs nothing and makes the knob askable, while
  entry into the default design would spend rows on columns that do nothing on
  most machines. See
  [`docs/remaining-factors-design.md`](docs/remaining-factors-design.md).
- `backend` column on bench-driver rows, recording what llama.cpp says actually
  ran ("ROCm", "CPU", ...). The after-the-fact companion to the GPU warning: a
  sweep whose rows all say `CPU` while `ngl` varies is the same fault, still
  visible in the results file long after the console output is gone — and
  visible to whoever is handed the CSV, who never saw the warning.
- Documentation: [`docs/NEXT-SESSION.md`](docs/NEXT-SESSION.md) (working handoff
  — what is in flight, blocked on what, and already verified),
  [`docs/field-reports.md`](docs/field-reports.md) (published third-party setups
  *and methods*, and what transfers — including a Bayesian/TPE autotuner whose own
  numbers show BO tying random search at our dimensionality),
  [`docs/flag-coverage.md`](docs/flag-coverage.md) (every llama.cpp flag audited
  against the registry), [`docs/draft-model-design.md`](docs/draft-model-design.md),
  [`docs/workload-shape-design.md`](docs/workload-shape-design.md),
  [`docs/concurrency-kv-design.md`](docs/concurrency-kv-design.md),
  [`docs/remaining-factors-design.md`](docs/remaining-factors-design.md).

### Fixed

- **One failed request no longer discards the whole measurement.** A concurrent
  round died on the first exception (`ThreadPoolExecutor.map` re-raises while
  iterating), so "7 of 8 requests served quickly" was recorded as `ERROR` —
  identical to "the model would not load". Those are very different configs to
  deploy. Failures are now counted into a new `err_rate` column; only a config
  where *every* request fails is an `ERROR`. No extra score penalty was added,
  because a round's wall clock already covers the failed requests: a config that
  serves less measures slower, which is the proportional penalty, arrived at
  honestly. What throughput cannot say is *why* it is low, so a recommended
  config that dropped requests now carries an explicit warning.
- **A completed-but-empty run could be recommended as the max-context config.**
  A run that finishes without generating anything keeps `status == "OK"` with a
  zero score — `implausible_reason` deliberately passes on `tg <= 0`, since
  nothing impossible happened. `pick_recommendations` filtered on status alone
  and computes `longest` as `max(ok, key=(depth, score))`, depth first, so such a
  row at the deepest depth won outright and was handed over as a paste-ready
  command that loads and then produces nothing. `pareto_frontier` was already
  guarded; recommendations and the max-context probe seed now share the same
  `measured_ok` filter. Counting paths still use plain status, because a run that
  completed did complete.
- **Stop emitting the deprecated `-mmp` / `--no-mmap`.** Model loading is now
  pinned with `--load-mode`, probed via `supports_flag` so builds predating it
  keep the old spelling. This was a latent whole-sweep failure: llama.cpp does
  remove deprecated arguments (`--draft`, `--draft-min` and
  `--spec-ngram-size-n` are already gone), and argument parsing failing takes
  every row with it, not one. It also removes a silent conflict — `common/arg.cpp`
  warns that the old and new spellings must not be combined, so a user passing
  `--load-mode` was fighting a flag we inserted for them. A side benefit: the two
  drivers finally say the same thing, where `-mmp 0|1` and a bare `--no-mmap`
  never agreed.
- Source citations across the docs and in-code comments refreshed against
  llama.cpp `4d19b2876`; four of twelve had drifted.

### Known issues

- `-kvu`/`--kv-unified` and `-sps`/`--slot-prompt-similarity` are still
  unreachable. Both are blocked on design rather than effort: `kv_unified` is
  constrained by `--parallel`
  ([`docs/concurrency-kv-design.md`](docs/concurrency-kv-design.md)) and `-sps`
  cannot move until requests differ
  ([`docs/workload-shape-design.md`](docs/workload-shape-design.md)).

## [0.1.0] — 2026-08-24

Baseline: the tool as it stood before provenance stamping, at commit `bbb206a`.
Results CSVs from this era carry no `tool_version` column.

Highlights of what was already in place: Taguchi orthogonal-array sweeps over the
llama.cpp parameter space with `llama-bench` and `llama-server` drivers,
conditional factors ([`docs/CONDITIONAL-FACTORS.md`](docs/CONDITIONAL-FACTORS.md)),
constrained/derived factors ([`docs/CONSTRAINED-FACTORS.md`](docs/CONSTRAINED-FACTORS.md)),
measurement-validity checks ([`docs/measurement-validity.md`](docs/measurement-validity.md)),
crash journal, `--resume`, `--iterate`, `--diff`, and a GPU-free `--selftest`.

### ⚠️ Affects results from before this release

- **The batch floor hid the low-batch regime** (fixed in `f7ed38d`, issue #8).
  `-b` was swept as an absolute at `2048, 4096, 8192` so that `-b >= -ub` held
  without llama.cpp's silent clamp firing. The cost was that no configuration
  below `-b 2048` was reachable *by construction* — including one audited user's
  own hand-tuned optimum of `-b 512 -ub 128`. `-b` is now swept as a multiple of
  each row's `-ub` (1x/4x/16x) and spans 128-32768.
- **Inverted speculative rows measured the baseline** (fixed in `f7ed38d`).
  `--spec-draft-n-min` above `--spec-draft-n-max` is not rejected by llama.cpp:
  it drafts at most `n_max` tokens and discards any draft shorter than `n_min`,
  so the row ran with speculation silently off while recording `mtp=1` —
  poisoning the `mtp` main effect rather than just its own score. Both
  ordered pairs are now derived so the inverted assignment cannot be constructed.

[issue #8]: https://github.com/bigattichouse/llama-optimize/issues/8
