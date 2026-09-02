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

- **Any comparison between two rows measured at different temperatures.** The
  thermal settle was on by default and had three independent ways of giving up
  without saying so (see *Fixed*), so `--no-thermal-wait` was never the only way
  to get thermally incomparable rows. A between-run swing on this box has been
  measured at ~27%, larger than most effects a sweep is looking for, so a
  confounded pair can invert a ranking rather than merely blur it.

  **Re-read first, re-run only if it matters.** `temp_c` was recorded on every
  row throughout, so this is checkable without the GPU: if the rows behind a
  conclusion differ by more than a few degrees, the difference between them is
  not attributable to the factor. A worked example is in this session's own
  results — EAGLE3 read as +12% against a baseline measured 43 °C hotter.

  Sweeps whose rows are all within a few degrees of each other are unaffected,
  and that is the common case on an idle card with the settle working.

- **Main-effects tables from server-driver sweeps may understate a factor.** A
  config whose reps never finished was recorded as `OK` with `tg=0.0`, and
  `factor_level_means` averages `OK` rows — so an unmeasured row counted as a
  measured zero and pulled down every factor level it appeared at. On a balanced
  design one such row halves the apparent effect at those levels, and
  `refine_factors` used the same means to choose what the next `--iterate` pass
  swept, so a bad pass could narrow toward the wrong region.

  **The picks, the Pareto frontier and the recommended command are unaffected** —
  those go through `measured_ok`, which already required a positive score. Only
  the main-effects table and `--iterate` refinement were touched.

  **Re-read, do not re-run.** If a sweep's CSV has rows with `status=OK` and
  `tg_tps=0`, its main-effects ranking was computed with them in.
  `--report-only` on the same CSV is enough: the stored status cannot be
  recomputed, but the *number* can be read, and `factor_level_means` now uses
  `measured_ok`'s criterion — a row that produced nothing is excluded whatever
  its recorded status says. Verified against a hand-built historical CSV: a
  level whose true mean was 7.06 read as 3.53 before and reads 7.06 after.

- **OOM-pruning decisions on any sweep that varied `nkvo` were made against the
  wrong KV placement.** The estimator had the two levels the wrong way round, a
  ~6 GB error on a large model. `SKIP_PRED` rows with `nkvo=1` were very likely
  fine and should have run; `nkvo=0` rows recorded as `OOM` or `SIGNAL` at launch
  were admitted on an estimate that ignored their KV cache. Rows that produced a
  number are unaffected — this only ever decided whether a config was *attempted*.

  **Re-run sweeps that show `SKIP_PRED` rows, or launch failures at high depth**,
  if the pruner was on (it is by default). Sweeps that never varied `nkvo` used
  its default level throughout and are internally consistent, though the pruner
  was still pricing that level's opposite.

- **Server-driver rows were measured shallower than their `n_depth` says.** The
  prompt is built in characters at an assumed 4 chars/token, and the measured
  ratio arrived too late to size it. Any model whose tokenizer is not close to 4
  chars/token ran a different depth than the column records — on the model in
  issue #11 (6.05 chars/token) a row labelled `n_depth=32768` ran ~27,000 tokens,
  about two thirds of it. The error is systematic per model, so *rankings within
  one sweep are largely intact*; what is wrong is the depth each row is labelled
  with, and therefore any `-c` recommendation or Pareto point read off it.

  **Re-run `--driver server` sweeps where the depth label matters** (max-context
  picks, the `-c` to paste). Bench-driver rows are unaffected — llama-bench is
  given token counts directly. New rows carry `prompt_tok`, so from now on the
  gap is visible rather than inferred.

- **`tg_tps` on server rows is now the median of the reps, not the mean.** Values
  will move slightly on any config whose reps disagreed. This is the same change
  `--verify-picks` already applied to its own re-measurements; no re-run is
  needed for it alone.

### Added

- **`--thermal-mode warm` (now the default)** preheats each config with its own
  workload until the GPU stops heating, then measures it at that steady state.
  That is the **sustained** rate — what a deployment gets from a card that is
  already hot, throttling included. `--thermal-mode idle` keeps the previous
  settle-to-baseline behaviour, which measures **burst** performance and is the
  right question for a bursty agent workload. `--no-thermal-wait` disables both.

  Warm means "at its own steady state", not "still hot from the last run" — the
  latter is the confound that produced rows at 44 °C and 87 °C. Preheating under
  the config's own load defines the state by the config rather than by its
  neighbour. Different configs plateau at different temperatures, and that is the
  honest answer: one that heats harder really does throttle harder.

  Plateau detection is a rolling **window**, not a step-to-step comparison, which
  was measured rather than assumed: an MI50 holding steady under load oscillates
  99↔100 °C, so a 0.5 °C step test resets on every other poll and never
  converges — the preheat ran to its cap while sitting at the steady state it was
  waiting for.

  Warm costs less wall-clock than it replaces: no idle baseline to establish and
  no cooldown between runs.

- **`--thermal-cap SECS`** bounds the idle-mode settle (default 600).

### Fixed

- **`temp_c` recorded the temperature *before* the run, not the one it ran at.**
  Harmless until warm mode, which puts a preheat in between: a validation sweep
  recorded 45 °C and 89 °C for two rows that both measured at the ~99 °C plateau.
  The one column that exists to check thermal comparability was reporting a 44 °C
  confound the preheat had already removed. It is now sampled after the preheat,
  immediately before the first measured rep, falling back to the old
  pre-measurement sample for drivers that do not report one.

- **The thermal settle had three ways of silently not settling**, all found by
  reading `temp_c` on a comparison that had the settle *enabled*: two rows at
  44 °C and 87 °C, a 12% apparent effect, no warning anywhere.

  - The plateau rule ended the wait after **one** poll that cooled by less than
    0.5 °C — at a 3 s poll that is 10 °C/min, a rate a hot card passes straight
    through. Now four consecutive stalled polls.
  - The 120 s cap was shorter than an MI50 takes to shed a run. Raised to 600 s
    and exposed as `--thermal-cap`, since it is a wall-clock/comparability trade
    the tool should not make silently.
  - The "idle baseline" was a single instantaneous sample, so a card still hot
    from the previous sweep produced `idle baseline 99°C — settle to ≤104°C`, a
    target nothing can exceed. `idle_baseline_c` now measures it by watching
    until the reading stops falling.

  A settle that gives up now names the temperature it gave up at.

- **A `--factor ngl=` pin was overridden by the load probe.** On a recurrent
  model under `--run`, `probe_loadable_ngl` rebuilt its candidate list from the
  *un-collapsed* level span, so an explicit pin was widened back out: a run
  pinned to `ngl=99` to isolate a draft head swept `48, 53, 59, 64` instead, and
  the rows varied in the one factor that had been held fixed. A probe now never
  runs on a factor the user set by hand.

- **Speculative telemetry was missing whenever the draft head came from
  `--draft-model`.** `spec_cols_wanted` recognised a NextN head and `--ngram` but
  not a draft model on the command line, so an EAGLE3 run on a model with
  `n_nextn=0` wrote every row with `draft_acc`, `draft_cov` and `spec_off`
  **absent from the CSV** — no way to tell whether the head had drafted a single
  token, which is the one thing you want to know about a draft head. The
  `spec_off` guard was blind in the same place, so a draft head that never ran
  was silently accepted rather than flagged. `spec_type=none` still means off for
  that row: a draft model on disk does not override an explicit level.

- **The `-ngl` load probe drew a general conclusion from one assignment.** It
  fixed every other factor at its first level, but whether a level loads depends
  on them: on `Qwen3.6-27B-Q5_K_M` (`qwen35`, MI50 32GB / ROCm, llama.cpp
  `6c84c7d5d`) `-ngl 53` core-dumps bare and loads under `--spec-type
  draft-eagle3`, so the verdict depended on which `spec_type` level sorted first.
  A level is now dead only if it fails under every level of
  `LOAD_SENSITIVE_FACTORS`, retried only where it failed, and levels that load
  conditionally are reported as such.

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

- **`--factor mtp=0` still swept the four speculative-tuning knobs.** Reported on
  [issue #11]: turning speculation off left `spec_n_max`, `spec_n_min_frac`,
  `spec_p_min` and `spec_p_split` at full level sets, so the array was sized on
  four columns that could no longer change anything — 25 runs of one
  configuration. The wasted time is the smaller half: the main-effects table then
  credited each of those knobs with an effect computed from run-to-run noise. A
  factor whose gate is pinned to a value that makes it inert is now dropped from
  the design (and said so on stdout), and the inert `--spec-draft-n-max` default
  is no longer pasted into a command that has no drafter. Emission is untouched,
  so `--draft-model` runs — which speculate with no `mtp` column at all — keep
  every flag they had.

- **One bad rep no longer discards the whole configuration.** Follow-up on
  [issue #11]: I5 compared the **mean** rate across reps against the **kindest**
  rep's ceiling, so a single request with a broken server counter (the reporter's
  last run: three reps at ~600, 333,362 and ~620 t/s) carried the outlier into
  the verdict and zeroed a row that had two perfectly good measurements in it —
  and the surviving reason could not say whether one rep had failed or all of
  them had. Each rep is now screened against its own request's duration, the
  survivors decide the number by median rather than mean, and the row is
  IMPLAUSIBLE only when nothing survived. Partial rejections are counted in a new
  `rejected_reps` column and printed, since a row that keeps its numbers is
  exactly where a silently dropped sample would go unnoticed.

- **Server prompts were built two thirds as deep as they claimed.** The prompt is
  sized in characters from an assumed 4 chars/token; the ratio was measured from
  the warm request, but the prompts had already been built by then, so on a
  single-config run it sized nothing at all, and in a sweep only the first config
  in each session ran uncalibrated — one row per session measuring a different
  depth from its siblings, in a design that randomizes execution order precisely
  so that cannot happen. Measured on the reporter's model with `llama-tokenize`:
  6.05 chars/token, so a row labelled `n_depth=32768` had run about 27,000
  tokens. A short calibration probe now runs at session open, before any prompt
  is sized, and the ratio is held for the whole session.

- **The recommended `-c` could exceed the context the sweep ever loaded.** It was
  derived from the requested depth; it is now capped by the delivered one.

- **A draft head combined with `--ngram` loaded and then did nothing.** Reported
  from the field on [issue #11] with the command that works:
  `--spec-type draft-dflash,ngram-mod -md <head>`. Ours emitted
  `-md <head> --spec-type ngram-mod` — and naming *any* type makes llama.cpp's
  `spec_types_is_default()` false, so `common_speculative_types_from_gguf()` is
  never consulted. The head loaded, cost VRAM, and only ngram drafted; a sweep
  would have measured ngram and labelled it DFlash2.

  The draft's own type is now named whenever anything else will be, and merged
  into one flag. Doing that safely needed an exact DFlash-vs-DSpark distinction,
  so the GGUF reader carries its single pass one block further and reads tensor
  names — llama.cpp tells them apart by `markov_w1.weight` and now so do we.
  When the draft is alone, the inference is still left to llama.cpp.

- **An EAGLE3 draft head would have run as the wrong kind of drafter.** llama.cpp
  implements EAGLE3 in full — encoder, decoder, target-layer hidden-state
  extraction — but `common_speculative_types_from_gguf` still recognises only
  `dflash` and the MTP tensor, its own outstanding TODO. A head reporting
  `general.architecture = eagle3` therefore names no type llama.cpp can infer,
  and the fix below for plain drafts would have handed it `draft-simple`. Draft
  architectures now map to the type they need named, so eagle3 gets
  `--spec-type draft-eagle3`.

- **A plain `--draft-model` never speculated at all.** Found while closing the
  rest of [issue #19]. llama.cpp infers the speculative type from the draft GGUF,
  but only for heads that name themselves — architecture `dflash`, or an MTP
  head. An *ordinary* draft model (the classic setup: a small sibling of the
  target) tells it nothing, inference returns an empty list, the type stays at
  its default of `none`, and the draft is **loaded, charged to VRAM, and never
  used**. Nothing reported it: on a target with no MTP head of its own, nothing
  had requested speculation, so the row was not flagged `spec_off` either.
  `--spec-type draft-simple` is now named for exactly the drafts that cannot name
  themselves, and withheld from the ones that can.

- **The pasted command omitted the draft model entirely.** A `--draft-model`
  sweep printed a suggested `llama-server` line with no `-md` on it, so the
  command could not reproduce the row above it. The command is the deliverable;
  it now carries the draft and its type.

- **A supplied `--draft-model` was loaded and then ignored.** [Issue #19]. On a
  target carrying its own MTP head the tool emitted `--spec-type draft-mtp`
  unconditionally, so pointing `--draft-model` at a DFlash2 head produced
  `-md dflash.gguf --spec-type draft-mtp` — the draft loaded, MTP ran instead,
  and nothing said so. llama.cpp infers the type from the draft GGUF itself
  (`common_speculative_types_from_gguf`: architecture `dflash` → draft-dflash or
  draft-dspark, an `nextn.eh_proj` tensor → draft-mtp), so the fix is to stop
  pre-empting that inference: a draft model named on the command line decides.

- **The prompt-cache miss warning now says something measured, and something
  different per architecture.** [Issue #15]. Both earlier versions were wrong, in
  opposite directions: the first blamed an undersized context, the second claimed
  prefix reuse was "a property the model does not have". Measured: SWA models lose
  reuse past the window and `--swa-full` restores it; hybrid/recurrent models keep
  87% reuse at 4k tokens, 43% at 8k and 0% at 16k, unchanged by
  `--ctx-checkpoints` at 32 or 512. Depth is the variable in both cases and the
  architecture only decides whether a knob exists — so the warning branches on
  `{arch}.attention.sliding_window` vs `{arch}.ssm.state_size` and gives the lever
  that actually applies.

- **`swa_full` is now swept automatically on sliding-window models.** [Issue #15].
  Past the sliding window, a shared-prefix workload loses its prompt cache: on
  gemma-3-270m with a 15.7k-token prompt at 90% requested reuse, the *second* rep
  re-prefills the entire prompt (0% cache hit) because the window has scrolled
  past the prefix. `--swa-full` holds it at 90%. Measured against the
  alternatives, which do nothing: `--ctx-checkpoints` at 0, 32, 128 and 512, and
  `--cache-ram`. The flag is not free — attention loses the sliding-window
  shortcut, 388 t/s prefill against 4283 — but it replaces a 15.7k-token
  re-prefill with a 1.6k-token one, so which side wins is a measurement, not a
  default. Detected from `{arch}.attention.sliding_window` in the GGUF; server
  driver only, and skipped on models that attend globally, where llama.cpp
  disables the flag itself.

  **Hybrid/recurrent models (`qwen35` and friends) have no equivalent knob** — a
  recurrent state cannot be rolled back to an arbitrary prefix at all — so for
  that class prefix reuse remains unavailable and `cache_hit` is the thing to
  read before believing a reuse-shaped result.

- **A speculative knob that could not act still voted on its own main effect.**
  [Issue #16]. With `mtp` swept, half the rows carry `spec_n_max` and friends at
  levels that do nothing, and those rows were averaged into their effects — a
  knob credited with an effect computed from run-to-run noise, in the table the
  user reads to decide what matters. On a 20/30/40/50 gradient the inert rows
  halve the apparent range. `factor_level_means` already excluded `active_when`
  rows; it now excludes `gated_by` ones too, and the inert `--spec-draft-*` flags
  are no longer pasted into an `mtp=0` command.

  Staging the gate — the fix the issue proposed — was measured and rejected: the
  array is already L125 on the unconditional factors, so removing the four
  columns shrinks nothing and a screen-plus-tune split would cost 150 runs
  instead of 125.

- **An unmeasured config was averaged into the main effects as a zero.** A
  server-driver row whose reps never finished came back `tg=0.0` with status
  `OK`. `measured_ok` kept it out of the picks, but `factor_level_means` filtered
  on `status == "OK"` alone — the two disagreed about what counts as a
  measurement — so the row was counted as a measured zero and
  **halved the main effect of every factor level it touched**, which
  `refine_factors` then used to decide what the next `--iterate` pass sweeps. The
  picks stayed right while the analysis and the refinement quietly did not.
  `err_rate` did not catch it either: a deadline break is not an exception.

  Fixed at both ends. Such a row is now `SLOW` (budget deliberately tightened by
  `--min-tgs`) or `TIMEOUT` (it simply ran out), matching what the bench driver
  already did, with the reason recorded and printed — both routes covered, no rep
  finished or every surviving rep reported 0 t/s. And `factor_level_means` now
  uses `measured_ok`'s criterion rather than the status alone, which closes the
  disagreement at its source and repairs CSVs recorded before the fix, since the
  stored status cannot be recomputed but the number can be read.

- **The OOM pruner priced the wrong KV placement on every row.**
  `_fit_params_flags` emitted `--no-kv-offload` for `nkvo=0`, but `nkvo=1` is the
  level that puts the KV cache in system RAM — that is what both drivers emit for
  it, and `--no-kv-offload` is the same flag by another name (llama.cpp:
  `-kvo, --kv-offload, -nkvo, --no-kv-offload`). So the estimate was inverted
  against the run. Measured on gemma-4-31B at `-ngl 60 -c 65536 -ctk f16`:
  **32,666 MiB with the KV on the GPU against 26,513 with it in RAM** — a 6 GB
  error, in whichever direction the row sat.

  Consequences ran both ways: `nkvo=0` rows were under-priced and admitted, then
  OOM'd at launch; `nkvo=1` rows were over-priced and deleted as `SKIP_PRED`. The
  regression test asserted the inversion, so it locked the bug in. There is now a
  test that whenever a *driver* emits the flag the *estimator* does too, which is
  the invariant that was missing.

  Found by asking why `full_offload_fits` reported that a model fitted when
  `llama-fit-params` says it needs 32,666 MiB of a 32,240 budget — so the `ngl`
  grid was also being collapsed toward full offload on models where the
  layers-for-context trade is exactly what the sweep should be exploring.

- **On hybrid SSM models, three of five `ngl` levels aborted the server.**
  [Issue #18]. Mapped on `qwen35moe`: `-ngl` 0, 1, 2 run; **3 through n_layers all
  segfault**; `n_layers + 1` and above run. Note `-ngl n_layers` *dies* while `n_layers + 1`
  lives — at the layer count the output tensor takes the last slot and block 0 is
  left on the CPU, which is the straddle that crashes. The grid is now `[0, 99]`
  on models with `{arch}.ssm.state_size`, with the reason printed and the override
  named. The CPU anchor stays: if the model does not fit, the OOM pruner drops the
  `99` row and CPU-only really is the only placement that runs.

  **Superseded in two ways, 2026-09-02** — see *Fixed*, above. The
  `resolve_fused_ops` warning this entry originally named as the cause is not
  one: it appears identically, `set to disabled` and all, in launches that load
  fine. And the map is the *bare* map — `--spec-type draft-eagle3` or
  `draft-dflash` lifts the whole dead band on `Qwen3.6-27B-Q5_K_M`, generating
  correctly at `-ngl 5` and `-ngl 53`.

- **The `ngl` grid spent its slowest rows where the answer could not be.**
  [Issue #14]. `ngl_levels` spanned `0 .. n_layers` evenly and never asked
  whether the model fits, so a 40-layer model that fits entirely in VRAM still
  got `[0, 10, 20, 30, 40]` — four levels putting layers on the CPU, every one a
  near-certain loser, and `ngl=0` an order of magnitude slower than the rest of
  the design. When `llama-fit-params` says every layer fits at the *deepest*
  depth and *largest* KV type in the design, the levels now bias to the top
  quarter instead: `[0, 30, 33, 37, 40]`.

  `ngl=0` is kept at every level count and every verdict. Not for information —
  because the verdict can be wrong, and it is then the only row that can still
  produce a measurement instead of an OOM. `--no-oom-prune` restores the even
  span: an estimator not trusted to delete rows is not trusted to shape the grid.
  The header says when the bias fired and why.

### Added

- **`--task`: measure on the content you actually generate.** [Issue #11]'s
  reporter saw 31 t/s on real work where the sweep reported 45 — same model, same
  box, different work. Speculative decoding drafts *output* tokens, so its
  acceptance is a property of what the model generates, and without a task the
  generation is a continuation of filler prose. Every speculative number the tool
  has produced is therefore a statement about prose.

  `--task` appends an instruction to the tail of every prompt: your own text, or
  a preset (`code`, `code:sql`, `code:js`, `code:cpp`, `code:web`, `reasoning`,
  `roleplay`). Tail rather than head because the depth axis needs the filler to
  carry the length, because long-context-then-instruction is the shape agent and
  RAG traffic has, and because it leaves the shared prefix — and therefore
  `--prefix-reuse` — meaning what it meant. The tail counts *inside* the
  requested length, so a row labelled `n_depth=8192` still sends ~8192 tokens.

  Verified on gemma-3-270m, the weakest model available: after 2000 tokens of
  filler, `--task code:sql` produces SQL and `--task roleplay` produces
  narrative, where no task produces more filler. The task is recorded in the
  results CSV and printed in the report's scope line, since two sweeps with
  different tasks did not measure the same thing.

- **`ngram-cache` is reachable at last**, via `--ngram-type ngram-cache` or
  `--factor ngram=ngram-cache` — the last of llama.cpp's eleven `--spec-type`
  values the tool could not express. Not in the default screen, and that is a
  hard limit rather than a choice: the gate already holds five variants and five
  levels is the ceiling for the orthogonal arrays here (the vendored library has
  no six-level array, and run generation rejects one outright). Its context cache
  builds as it goes, so it needs no `-lcs`/`-lcd` file.

- **`repack`, `no_op_offload` and `no_host` are now swept** instead of silently
  inheriting llama.cpp's default on every run. Each is a documented behaviour
  switch whose answer depends on backend, CPU and model, and each was registered
  and reachable by hand but never explored. Free in runs: an L125 holds 31
  factors, so the design stays at 125. Gated on the binary advertising the flag,
  and safe to sweep blind because a level that turns out not to run is reported
  and dropped by the out-of-bounds detection rather than poisoning the design.

- **`spec_type` factor: sweep *which* speculative head, not MTP on/off.**
  [Issue #19]. Reported from the field — "DFlash2 seems a lot better than MTP" —
  on a model that has both, where the sweep could not put that question at all
  because `mtp` is a binary projection of a categorical axis. When more than one
  head is available the design now carries `spec_type` (`none`, `draft-mtp`, and
  the supplied draft's own architecture, e.g. `dflash`) and `mtp` steps aside;
  with only one head, nothing changes.

  The levels do not share a flag, which is why they are emitted by the caller:
  `draft-mtp` names a type (the head is inside the target model), a supplied
  draft head is `-md` and *no* type — llama.cpp reads it off the file, and
  telling dflash from dspark is a tensor-level distinction we deliberately do not
  attempt — and `none` is silence. `-md` is loaded only for the level that asks
  for it, so a row is not charged VRAM for a head it is not measuring.

- **Crashes are read as boundary measurements, not discarded.** A `SIGNAL` row
  says a parameter set is out of bounds, and the array already visited that
  region beside different partners — so a level that failed in *every* one of its
  rows is out of bounds rather than unlucky. Those levels are now named in the
  report (`OUT OF BOUNDS`, with the status tally that decided it) and dropped
  from the next `--iterate` pass, so refinement narrows toward what actually
  runs. Deliberately conservative: one failure is never enough, `SLOW` and
  `IMPLAUSIBLE` do not count (both had a working launch), a factor with every
  level dead is reported as nothing (that is the model failing, not the knob),
  and only levels *measured* dead are dropped — the neighbours `refine_numeric`
  invents are unknowns for the next pass to settle.


- **`prompt_tok` column: the tokens the prompt actually became**, next to the
  `n_depth` that was asked for. Different numbers on any tokenizer that is not
  4 chars/token, and the recommendation now rides on the measured one.

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
  can decline to reuse a prefix the client believes it shares — and when it does,
  a rep the tool believes is pure decode pays a full prefill instead. Nothing in
  the row could tell those apart. A large gap between requested and delivered is
  warned about rather than rejected: the numbers are real, they just measure a
  colder workload than the one being tuned for.

  The leading cause turned out not to be the one first guessed. On the reporter's
  model — `general.architecture = qwen35`, a **hybrid SSM/attention** model —
  llama.cpp cannot roll a recurrent state back to an arbitrary prefix, so it
  forces full prompt re-processing unless a context checkpoint covers the prefix.
  Delivered reuse was 0% at every depth and `--prefix-reuse 100` changed nothing.
  Prefix reuse is a property the model either has or does not; the warning now
  says so, and names an undersized context second rather than first.

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
[Issue #14]: https://github.com/bigattichouse/llama-optimize/issues/14
[Issue #16]: https://github.com/bigattichouse/llama-optimize/issues/16
[Issue #15]: https://github.com/bigattichouse/llama-optimize/issues/15
[Issue #18]: https://github.com/bigattichouse/llama-optimize/issues/18
