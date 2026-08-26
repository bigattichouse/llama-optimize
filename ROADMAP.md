# Roadmap

Improvement ideas, roughly ordered by expected value. Items get checked off (and
their design notes trimmed) as they land.

**Picking up mid-stream?** [`docs/NEXT-SESSION.md`](docs/NEXT-SESSION.md) is the
working handoff: what is in flight, what is blocked on what, and what has already
been verified so you do not re-check it.

## 1. Noise-aware picks — partially done

Landed: `--verify-picks` (default on, 2 extra reps; `--full`=3) re-measures the
pick candidates after the sweep and reports the **median** of all measurements,
with the observed spread printed on the pick (persisted to
`<results>.verify.json` so `--report-only` re-applies it). Motivated by a real
sweep where the same config measured 10.6 vs 7.7 tg t/s (~27% swing, thermal).

Remaining:

- llama-bench `-o json` already reports `stddev_ts` per test — capture it into
  the CSV as `pp_std`/`tg_std`; have the server driver keep per-rep samples and
  do the same.
- Report: flag a pick that is statistically tied with its runner-up (within
  ~2σ of the combined noise).
- Tie-breaking: among tied configs prefer more context, then lower measured
  VRAM, instead of whichever got the lucky rep.
- Use the recorded `temp_c` to flag rows measured well above the idle baseline.

## 2. Predictive OOM pruning

OOM rows are correctly scored 0, but each still costs a model load + timeout —
at L125 with big models that's 20+ minutes of known-doomed runs.

- `--vram` sampling already exists. Fit a rough VRAM footprint from
  `ngl`/`kv_type`/context on the first few completed rows.
- Skip combinations certain to exceed physical VRAM, recorded as `SKIP_PRED`
  (never silently absent), with an opt-out flag.
- Must be conservative: a wrongly-skipped viable config is worse than a wasted
  OOM run. Needs live-GPU validation.

## 3. Multi-GPU factors

No `-ts` (tensor-split), `-sm` (split-mode layer/row), or `--main-gpu` in the
FACTORS registry — the biggest untuned lever on 2+-card boxes.

- Detect devices via llama.cpp's own `--list-devices` (landed for issue #7),
  not the vendor tool — the orderings disagree on non-identical cards. Only
  add the factors when >1 (same "only sweep where it varies" pattern as `numa`).
- Sensible default levels: `sm=layer,row`; `ts` bracketing the VRAM ratio
  *widely* — a published mixed-Vega setup tunes to 1:4 where capacity says 1:2
  ([`docs/field-reports.md`](docs/field-reports.md), F3). Keep the
  single-device configuration reachable: on that box the second GPU lost.
- Full factor model and checklist: [`docs/multi-gpu-design.md`](docs/multi-gpu-design.md).
- Needs multi-GPU hardware to validate.

## 4. ~~Results-diff mode~~ — done

`--diff old.csv new.csv` compares two sweeps of the same factor space
(llama.cpp upgrade, driver update, quant swap): per-config tg deltas on the
factor columns both files share, status changes, and whether the old winner
still wins.

## 5. Time-to-first-token metric — partially done

Landed: every suggested command now prints a prefill-cost estimate — filling
the emitted `-c` at that config's measured `pp` speed, plus an 8k-prompt figure
(e.g. the 235k max-context command: ≈32 min to first token). Derived, not
measured.

Remaining (true measured TTFT):

- Report alongside `pp`/`tg` (timestamp of first streamed token, server driver).
- Not a new objective initially; could later back a `--score ttft`.

## 6. ~~CI for the selftest~~ — done

GitHub Action running the selftest on push/PR, plus a binding smoke test
(builds the submodule, exercises L25/L125 generation and the analyzer — the
paths the selftest deliberately skips).

## 7. ~~Speculative-decoding telemetry~~ — done

We swept six speculative factors while recording nothing about whether the
drafter was ever accepted. llama.cpp hands both counters to us in the same
`timings` block `ServerSession.measure` already reads for the throughput rate.

Landed: a `draft_acc` column (accepted/drafted over the measured reps), plus a
`spec_off` flag for a run that asked for speculation and drafted nothing — the
issue #8 failure class that `docs/CONSTRAINED-FACTORS.md` closes by construction,
now independently checked rather than assumed. It is a flag, never a status: the
measurement is real, it just isn't measuring what its factor column claims. Both
columns appear only where a draft is possible at all. Background:
[`docs/field-reports.md`](docs/field-reports.md), F1.

## 8. Draft-model tuning

We accept no `-md`, so for anyone who owns a draft model the entire draft-side
surface — `-ngld`, `-ctkd`/`-ctvd`, `-ncmoed`, the `draft-simple`/`draft-dflash`
spec types — is unaskable. Design and checklist:
[`docs/draft-model-design.md`](docs/draft-model-design.md).

## 9. Workload shape — now a correctness fix, not a feature

Every request a sweep issues is byte-identical, so any parameter whose payoff
depends on how requests *relate* measures zero by construction — prefix caching
above all. Shape is an input describing the user's traffic, not a factor to
optimize.

Measured 2026-08-26, this is worse than a gap. n-gram speculation keeps state
across requests, so a repeated prompt drives it to **100% draft acceptance and
2.3-3.4x the honest throughput**; on distinct prompts it never drafts at all.
Every ngram number this tool has produced was taken in that saturated regime:
ngram-vs-off is inflated, and the variant screen has been ranking configurations
at a shared ceiling where the differences it exists to resolve cannot appear.
Until `--prefix-reuse` lands, ngram results are upper bounds. Design, data and
checklist: [`docs/workload-shape-design.md`](docs/workload-shape-design.md).

## 10. Flag coverage — two live defects

A full audit of `llama-server`/`llama-bench` `--help` against `FACTORS`
([`docs/flag-coverage.md`](docs/flag-coverage.md)) turned up two things that are
wrong today rather than merely missing:

- `bench_command` emits `-mmp`, now deprecated in favour of `--load-mode`.
  llama.cpp does remove deprecated args (three removals are visible in the same
  help text), and when this one goes every bench run fails at argument parsing.
- `--parallel` is always passed explicitly, and `kv_unified` defaults to true
  *only* when slots are auto (`tools/server/server.cpp:151`). So every
  concurrency sweep has measured `kv_unified = false`, and the regime a default
  `llama-server` actually runs in has never been measured.

Plus ~12 uncovered perf-relevant knobs aimed at the partial-offload case
(`--load-mode`, `--repack`, `--op-offload`, `--no-host`, `-kvu`, `-sps`,
`--swa-full`, batch-phase CPU affinity, `--prio`), and an `--audit-flags` mode so
coverage stops rotting silently.

## 11. ~~Warn when the build cannot see the GPU~~ — done

A llama.cpp without a GPU backend reports plausible CPU numbers rather than
failing, so every `-ngl` level measures the same run and the sweep recommends
layers that never load. Hit for real: a stale build directory left `GGML_HIP=OFF`
despite `-DGGML_HIP=ON`, and the same model measured 115 t/s (CPU) vs 444 t/s
(ROCm). Now diagnosed from `detect_vram_mib`'s existing answer — a vendor tool
seeing a card llama.cpp cannot is the signal. Warns loudly, never refuses: CPU-only
sweeps are legitimate and get a one-line note instead. See
[`docs/measurement-validity.md`](docs/measurement-validity.md).

Also landed: the `backend` column, from `llama-bench -o json`, so the same fault
is visible after a sweep and to whoever receives the CSV.

## 12. Remaining knobs — registered, not yet swept

Fourteen llama.cpp knobs are now reachable via `--factor` but stay out of the
default design on purpose ([`docs/remaining-factors-design.md`](docs/remaining-factors-design.md),
R1). Two open questions would promote some of them: whether SWA is detectable
from GGUF metadata (`swa_full`), and whether `load_mode` should auto-sweep when
the model does not fit in VRAM — which is exactly where paging decides
throughput.

## 13. Concurrency and the unified KV cache

`--parallel` and `kv_unified` are coupled by llama.cpp and cannot ride an
orthogonal array as free columns. Worse than the audit said: single-stream
sweeps silently run 4 auto slots with unified KV, concurrency sweeps never see
unified at all, and `parallel = auto` is not expressible. Design:
[`docs/concurrency-kv-design.md`](docs/concurrency-kv-design.md).

## 14. Objectives we do not measure

From a Bayesian vLLM autotuner ([`docs/field-reports.md`](docs/field-reports.md),
F6), which runs NSGA-II over up to seven axes against our two. Measured TTFT is
already item 5; TPOT and goodput are cheap now that per-request timings are
recorded (item 7). **tokens/joule** is the genuinely new one, and interesting on a
card whose thermal ceiling is already the dominant noise source.

The same source also specifies the memory model item 2 has been blocked on:
weights + KV + state + graph overhead + activations + margin, against a usable
pool and a hard VRAM limit, deliberately first-order so it stays conservative.

## Small cleanups

- ~~`--merge-results` rows aren't deduplicated against the current pass~~ —
  done: a merged row is kept only if it beats every known measurement of that
  exact config, so the Pareto/all-runs tables don't repeat rows across passes
  (and the never-lose-an-earlier-best guarantee is preserved).
