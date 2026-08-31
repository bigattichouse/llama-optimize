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

### Fixed

- **The wall-clock validity check (I5) rejected honest deep-context runs.**
  Reported as [issue #11]. A server's self-reported `tg` covers decode alone, but
  the check bounded it by the whole request's elapsed time — sound only while the
  request *is* decode. It stopped being that when workload profiles gained a
  shared-prefix fraction: below `--prefix-reuse 100` every rep sends a different
  prompt and re-prefills its differing suffix inside the request being timed. On
  the reporter's box that was 17.6s of a 23.5s request, so an honest 43.1 t/s was
  discarded as "4× faster than the wall clock permits". Deep context makes both
  terms worse at once — the prompt to re-prefill grows while the generation stays
  at `n_gen` — so the profiles that hurt most are exactly the ones aimed at long
  contexts.

  The prefill is now subtracted before the ceiling is formed, from the same
  response's `timings.prompt_ms`, falling back to `(1 − reuse) × prompt_len / pp`
  priced off the warm request for servers that do not report it. The credit is
  capped at 90% of the wall: it comes from the same clock this check exists to
  distrust, and uncapped, a response claiming it spent the entire request on
  prefill would switch the check off against precisely the fault it was written
  for. Capped, a broken server can loosen its own ceiling by at most 10×, against
  a defect ([issue #3]) that overshot by ~1500×.

### Added

- **A rejected row now carries the evidence for its own appeal.** [Issue #11]
  took five round trips with a reporter because I5 zeroes `pp_tps` and `tg_tps`
  on the row it discards, wrote the reason only to the console, and never
  recorded what the server said about the request. So: `implausible` is a
  results-CSV column, its text carries the breakdown (tokens, wall, credited
  prefill) and the measured `pp`; `--report-only` prints discards as well as the
  live sweep, since it is the one command a reporter can run with no GPU and no
  llama.cpp build.

- **`cache_hit` column: the prefix reuse the server actually delivered.** `reuse`
  records only what the prompt battery asked for. llama.cpp matches on tokens and
  can decline to reuse a prefix the client believes it shares — a context too
  small to hold prompt plus generation, a slot reset, an eviction — and when it
  does, a rep the tool believes is pure decode pays a full prefill instead.
  Nothing in the row could tell those apart. A large gap between requested and
  delivered is now warned about rather than rejected: the numbers are real, they
  just measure a colder workload than the one being tuned for.

- **`concurrency` factor, and `kv_unified` recorded per row.** llama.cpp couples
  slot count and the unified KV cache, so they cannot ride an orthogonal array as
  two free columns: auto means 4 slots *and* unified, and any explicit
  `--parallel` disables unified — including `--parallel 1`. The factor is one
  categorical over the states that exist: `auto` (emit nothing), `N`
  (`--parallel N`, split cache), `Nu` (`--parallel N --kv-unified`).

  `auto` is the point of it. It is llama.cpp's own default and was previously
  unreachable, because the tool always passed `--parallel` explicitly whenever
  concurrency was non-trivial. Every level was verified against a real server's
  log, including that `-c 2048` gives each of 4 split slots `n_ctx_slot = 512`
  while a unified 4-slot server gives each the full 2048 — the context rule the
  design was blocked on. The older `parallel` factor still works and always
  splits the cache.
- **`ncffn` factor** (`-ncffn`, llama.cpp tag b10645): for dense models, how
  many of the first N layers keep their dense FFN weights on CPU — the
  finer-grained form of the `-ot` lever (per layer, instead of all-or-nothing).
  On builds that have the flag it replaces `ot` in the default dense-model
  design (5 levels, `0 .. n_layers`); previous builds keep `-ot`. `-ot` stays
  reachable via `--factor ot=...` — `ncffn` varies *how many layers* offload
  while `ot=ffn_up_cpu` varies *which tensor*, so the two are different axes and
  the finer one does not subsume the lighter one.
- **`--mmproj`: a multimodal projector is an input the sweep can be told about.**
  A projector occupies VRAM from load whether or not any image traffic arrives,
  which made it different from the other absent multimodal *workloads*: those
  cost nothing when unused, this one silently shrinks the budget `predict_fits`
  and the max-context probe both reason about. Passing `--mmproj` now emits it to
  the server, adds `--mmproj-offload` as a factor (the one genuine placement
  decision it carries), and includes its size in the footprint.

  Coverage of multimodal as a workload — request shape, `--image-min-tokens`,
  `-mmdev` — remains out of scope. This is about not being wrong when one is
  loaded.
- **`--draft-model` / `-md`: the draft-side surface becomes reachable.** First
  increment of F2 ([`draft-model-design.md`](docs/draft-model-design.md)). A
  draft model is an *input*, not a factor — a sweep either has one or it does
  not — and giving one adds `-ngld` and `-ctkd`/`-ctvd` as swept factors. Without
  it they are omitted entirely rather than swept inert, because llama.cpp reads
  them only when `-md` was given and a null main effect would read as "draft
  placement doesn't matter" rather than "never tested".

  This also makes the second route to MTP expressible (issue #12). `has_dft()` is
  just "was a `-md` path given", and it decides whether llama.cpp drafts from the
  target's own NextN head or loads a separate head file as a second model. Both
  are real configurations of the same capability, at different VRAM costs; only
  one of them was reachable before.

  Draft `-ngld` levels come from the **draft** model's layer count, not the
  target's. Draft KV is deliberately exempt from `--min-kv`: that floor protects
  output quality and the drafter emits no output — a token drafted from a
  degraded draft cache is verified by the target, then accepted or discarded, so
  quantising it costs acceptance rate (speed), which is the thing being measured.

  **The OOM pruner now prices resident artifacts it cannot ask about.**
  `llama-fit-params` rejects `-md` and `--mmproj` ("invalid argument"), so it
  estimates the text model alone while the machine holds two or three. Rather
  than skip pruning on those rows, `resident_extra_mib` adds what can be measured
  without it: weights on disk are a hard lower bound on weights in VRAM — the
  draft GGUF scaled by `-ngld`'s share of its layers, plus the projector when
  `--mmproj-offload` leaves it resident. Both are per-row, so the figure reaches
  the fit cache key too.

  A lower bound is the only safe direction, and asymmetrically so: understating
  prunes fewer rows, so some doomed configs run and cost time, while overstating
  deletes configs that would have fit and costs information. Compute buffers and
  the draft model's own KV cache are therefore deliberately not modelled, and an
  unreadable draft geometry prices as zero rather than as "all of it".
- **Setup interview on a bare invocation.** `llama-optimize.py model.gguf` on a
  terminal now asks four questions — context actually served, slowest useful
  generation speed, how much of the space to search, repeats — then prints the
  derived command with its run count and estimate and offers to run it. It stays
  silent when stdout is redirected (a script blocking on a prompt is a hang) and
  when any intent-bearing flag is passed, so existing usage is untouched.

  It exists because the cost dials do not compose the way people reasonably
  assume. `choose_array` sizes on the WIDEST factor, so narrowing one knob
  changes nothing: `--factor kv_type=q8_0 --factor threads=8` still yields 125
  runs, and so does `--ctx-size` alone. A user who "limited context sizes" and
  saw 125 runs anyway would reasonably conclude the flags were ignored.
- **`--levels N` — the cost dial that actually works.** Narrows every
  auto-generated factor together, which is the only way the array shrinks:
  L125/125 runs at the default 5, L27/27 at 3, L16/16 at 2. Explicit `--factor`
  values are untouched. `five_levels_span` had the count hard-coded, so this
  also generalised it (and `ngl_levels`, `depth_levels`, `thread_levels`,
  `cpu_offload_levels`) to take a level count. Previously `n_depth` in
  particular could only be 5 levels or, via `--ctx-size`, exactly 1.
- **`--timeout` is a per-config deadline on both drivers.** It used to mean two
  different things. On llama-bench it bounded the whole process, which is what it
  reads like. On the server driver it was handed to each HTTP request, so one
  config could legitimately spend `(1 warm + reps) x timeout` — at the defaults,
  **80 minutes for a "20 minute" timeout**. It was not a bound on anything a user
  could name. It now covers warm-up and every rep together, and the per-request
  value is whatever is left of it.
- **`--min-tgs` / `--min-pps` — stop paying for configs you would not ship.** A
  config generating at 0.5 t/s takes ~8.5 minutes per rep at `--n-gen 256`, and
  the only previous guard was the blunt `--timeout`. A throughput floor tightens
  the deadline instead of adding a second mechanism: a config that would MEET the
  floor finishes `reps x n_gen` tokens within `tokens / floor` seconds, so
  exceeding that budget is itself the finding.

  That arithmetic is why it works on llama-bench too, where nothing can be
  observed mid-run. On the server driver `--min-pps` is answerable the moment the
  warm-up returns, so a config failing it skips its decode reps entirely — the
  "once pp is done, skip the slow ones" case.

  Worst-case sweep time on an L125, `--n-gen 256`, `--reps 3`: **41.7h at the
  default `--timeout 1200`, 17.8h at `--min-tgs 2`, 3.5h at `--min-tgs 10`.**

  Rows are marked `SLOW`, a status distinct from both `TIMEOUT` (it did not hang,
  we stopped waiting on purpose) and `IMPLAUSIBLE` (those numbers cannot be true;
  these are true and unwanted). A `SLOW` row **keeps its measured numbers** and is
  excluded from the picks like any non-OK status. `--tgs-timeout` (default 60s)
  floors the derived budget so a small `--n-gen` cannot produce one that measures
  model load instead of throughput.
- **Free-VRAM preflight.** `detect_vram_mib` reports TOTAL VRAM, which is the
  right basis for "can this model ever fit on this card" and the wrong one for
  "can it fit right now". The OOM pruner compares against total, so on a card
  another process is holding it passes every config, each run then aborts inside
  the GPU allocator, and the sweep spends its entire budget discovering that the
  card was busy. The device list already carried free memory per device; it was
  simply never read. The header now prints `VRAM free : N of M MiB at start` and
  warns loudly below a quarter free.

  Free VRAM warns but does **not** prune: it is one instant's reading of a number
  the vendor tool can misreport (issue #7's APU counts GTT), and whatever holds
  the card may release it before the sweep reaches the rows that need it.
  Refusing to run would trade a wasted sweep for a sweep that never starts.

  The warning also names the escape, because the obvious one does not work:
  `-ngl 0` still allocates VRAM, since llama.cpp op-offloads matmul to the GPU
  backend even with no layers resident. Tuning on the CPU while the card is busy
  needs the device *hidden* (`HIP_VISIBLE_DEVICES=` / `CUDA_VISIBLE_DEVICES=`).
  Found on the development box: 174 MiB free of 32752, every bench row aborting
  in `ggml_cuda_pool_leg::alloc`, and the pruner cheerfully approving them all.
  This is the mirror of the fit-params blindness fixed earlier in this section —
  that one deleted valid rows, this one admitted doomed ones.
- **`ffn_place` — one dense FFN placement factor spanning `-ot` and `-ncffn`.**
  The two are different axes, not rival spellings: `ot=ffn_up_cpu` moves the
  up-projection across *every* layer, `-ncffn N` moves the whole FFN for the
  *first N*. At equal VRAM freed they shape PCIe traffic differently — a thin
  slice touched in every layer against a contiguous block — so which wins is a
  measurement, and both belong in the default design.

  They cannot ride as two orthogonal-array columns. `-ncffn` appends per-layer
  overrides to the same `params.tensor_buft_overrides` vector `-ot` writes to
  (llama.cpp `common/arg.cpp:2787-2798`), so they compose and `ot=ffn_cpu`
  swallows every `-ncffn` level — rows recording a level that changed nothing.
  So it is one categorical whose levels are mutually exclusive by construction,
  the same shape the `concurrency` factor uses and for the same reason. Levels
  are pre-rendered from the layer count (`none`, `ffn_up_cpu`, `first_N`,
  `first_M`, `ffn_cpu`), so a level is self-describing in the CSV.

  The factor NAME does not change with build capability — a build without
  `-ncffn` gets the three `-ot` levels under the same column, so results stay
  comparable across a llama.cpp upgrade rather than changing shape. Raw `ot`
  and `ncffn` remain reachable via `--factor`. Replaces the straight
  `ot`→`ncffn` substitution that shipped moments earlier in this same
  unreleased section.
- **`emit` factors: level sets are checked for distinctness at build time.** A
  factor whose *level* picks its flag can silently contain two levels that emit
  the same arguments — two names for one run. The array then balances a column
  that measures nothing, and the main effect reads "placement doesn't matter"
  when the truth is it was never varied. Found the hard way: a level spelled
  `up_cpu` missed the `OT_PATTERNS` key `ffn_up_cpu`, emitted nothing, and
  duplicated `none`.
- **OOM pruning is skipped, not guessed, when the estimator is too old.** The
  driver and `llama-fit-params` are separate binaries with separate flag
  support, so a row can set a placement factor the estimator cannot parse. Such
  a flag must not simply be dropped from the estimate: `-ncmoe`/`-ncffn` exist
  to move weights *off* the GPU, so an estimator blind to one reports the
  un-offloaded footprint — the largest configuration in the row. On the machine
  that needed the offload, that overshoots VRAM and every level of the factor is
  predicted OOM, deleting the whole factor from the sweep without a word. Rows
  carrying a factor the estimator cannot see now skip pruning and run, with one
  notice naming the flag. Same lineage as issue #5: a config that would have run,
  discarded on the strength of a number that was never about it.
- **`SIGNAL` status** — a run whose process was killed by a signal is no longer
  filed as a generic `ERROR`. The two mean different things and only one is about
  the config: a process that *ran and returned an error* was rejected by
  llama.cpp, while a process killed by a signal crashed inside it.

  Found while validating the OOM pruner: Qwen3.8 (Gated Delta Net) on gfx906
  segfaults during context init above roughly 64k with **no allocation failure
  logged**, so the OOM patterns do not match. It is genuinely not an OOM — the
  same model cleanly reports `cudaMalloc failed: out of memory` at 200k, well
  above where the segfaults start — so calling it one would have been wrong too.

### Verified

- **The OOM pruner is not over-conservative** (ROADMAP item 2's last open
  bullet). On Qwen3.8-27B / 32 GB MI50 the decision boundary sits between 143,996
  and 145,073 tokens of context, and the predicted-OOM side is real: 200k fails
  allocating the KV cache. No viable configuration was wrongly skipped.

## [0.2.0] — 2026-08-26

### ⚠️ Affects existing results

- **Server-driver rows discarded as `IMPLAUSIBLE` may have been honest.** The
  wall-clock check rejected real measurements wherever prefill was a large share
  of the request — which, since profiles gained a shared-prefix fraction, is any
  single-stream server run at `--prefix-reuse` below 100 with a deep `n_depth`.
  Rejected rows score as 0, so a sweep did not merely lose them: it ranked the
  configs that tripped the check *below* configs that were genuinely slower.

  **Re-run any single-stream `--driver server` sweep that reported IMPLAUSIBLE
  rows at depth.** Rows that came back `OK` are unaffected — this check only ever
  removed measurements, never altered one. Concurrency runs (`--parallel > 1`)
  are unaffected: that path computes its rates from our own wall clock and never
  ran the check. Bench-driver sweeps never ran it either.

- **ngram sweep results are upper bounds, not estimates.** llama.cpp's n-gram
  speculation keeps state *across requests*, and the sweep sends one identical
  prompt for every rep. Measured 2026-08-26 (gemma-3-270m-it-Q8_0, ROCm,
  `--spec-type ngram-mod`): the first request drafts nothing, every subsequent
  one reaches **100% draft acceptance**, and throughput goes 273 -> 601 t/s. On
  genuinely distinct prompts the drafter never fires at all (176-259 t/s).

  Since the harness discards the first request as warmup, *every measured rep*
  sat in the saturated regime. Two consequences: ngram-vs-off is inflated
  (**~1.46x** on throughput, ~1.75x on speculative coverage — earlier drafts said
  2.3-3.4x, from a generator that was itself contaminated), and the variant screen
  has been ranking `ngram-simple`,
  `ngram-mod`, `ngram-map-k` and `ngram-map-k4v` at a shared ceiling where the
  differences it exists to resolve cannot appear.

  **Re-run any sweep that used `--ngram`.** Unaffected: `ngl`, `threads`,
  `ubatch`, `batch`, `kv_type`, `nkvo`, `poll`, `ot` — these do not depend on
  what the tokens say.

  **MTP is unaffected — now measured, not assumed.** On Qwen3.8-27B with
  `--spec-type draft-mtp`, acceptance moves only 0.78 → 0.70 across the whole
  reuse range and reproduces to ±0.02. MTP drafts from the model's own NextN head
  rather than from cross-request n-gram state. **MTP sweep results stand.** See
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
- **`--prefix-reuse PCT`** — describes the *shape* of your traffic: how much of
  each prompt is a prefix shared across requests. It is an input, not a factor:
  an agent stack with a fixed system prompt has ~90% shared prefix whether or not
  that is convenient, and the right settings for it are the ones that win at its
  reuse level. Sweeping it would report "your workload should have more prefix
  reuse", which is not advice.

  **Defaults per profile: 0 everywhere except `agents` at 90**, whose name is
  itself a claim about traffic. 0 means "assume nothing shared unless told
  otherwise" — chosen because the failure modes are asymmetric: overstating
  speculation is invisible and ships a config that will not deliver, while
  understating it is visible and recoverable. Non-speculative results are
  unaffected either way (`tg` measured flat across the whole reuse range:
  371.6 / 372.0 / 368.9 / 368.6), so the change moves only the results that were
  wrong.
- **Category-weighted prompt battery** — 40% short-QA, 20% reasoning, 20% code,
  15% RAG, 5% long-context, following the reasoning that real traffic is
  dominated by cheap requests but conditioned at high percentiles by expensive
  ones. Prompt *text* only; per-category output lengths would redefine what a
  single tg number means and were deliberately not smuggled in.
- **`reuse` column** — the prefix fraction the battery *actually* shared,
  measured from the generated prompts rather than echoed back from the request,
  so a result can be read in light of the traffic that produced it.
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
- Documentation: [`docs/PLAN.md`](docs/PLAN.md) (execution order and what blocks
  what), [`docs/constants-audit.md`](docs/constants-audit.md) (every
  hardcoded value classified as universal / derivable / ours — the
  hardware-poisoning audit),
  [`docs/NEXT-SESSION.md`](docs/NEXT-SESSION.md) (working handoff
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

- **Chars-per-token is measured, not assumed to be 4.** The constant decided the
  real `n_depth` of every server measurement and the size of every battery
  prompt, and tokenizer ratios vary widely by model and language — code and CJK
  are nowhere near 4. It is now calibrated from `prompt_n`, which llama.cpp
  already reports in every response; the constant survives only as the bootstrap
  for the first request. A ratio outside 1–20 chars/token is rejected as a
  malformed response rather than adopted, because a wrong ratio sizes every
  prompt after it.
- **The sweep time estimate no longer assumes 90s per run.** A 270M model and a
  27B model both reported ~187 minutes, and users decide whether to start a
  multi-hour sweep from that number. It now scales with model size, driver, reps
  and `n_gen` — 20m / 1h18m / 3h21m for 270M / 8B / 27B — and says it is a guess.
  The decode-rate prior inside it scales too, since a fixed "20 tok/s" would just
  have moved the hardcoded assumption somewhere less visible.
- **The prompt generator was inflating speculative acceptance, at two scales.**
  First, prompts were built by tiling a short passage, making each internally
  self-similar — n-gram speculation feeds on repetition inside one context, not
  just across requests. Second, after switching to a sentence pool, the pool held
  14 sentences / 1,529 characters against the 32,768 an `agents` prompt needs, so
  the whole corpus repeated ~20x per prompt. Both pinned acceptance at 100% even
  at 0% reuse. Sentences are now generated combinatorially (~14,400 distinct from
  a few lines of source), and `repeated_fraction` measures the property directly
  rather than assuming it: 0.000 at 256 and 512 tokens, 0.094 at 8,192.
- **`draft_acc` alone ranked speculative configs backwards.** Acceptance is
  accepted/*drafted* — draft quality — and the two slowest ngram configurations
  measured scored a perfect 1.00 while the fastest scored 0.84. A drafter that is
  always right about the few tokens it dares to guess is not helping much. The
  new **`draft_cov`** column is accepted/*generated*, the share of output that
  came free from speculation, and it tracks throughput (0.81 → 846 t/s, 0.46 →
  579). Both are kept: quality and contribution are different questions.
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
[issue #3]: https://github.com/bigattichouse/llama-optimize/issues/3
[issue #11]: https://github.com/bigattichouse/llama-optimize/issues/11
