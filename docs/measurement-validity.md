# Measurement validity — rejecting impossible numbers

General principles for deciding whether a measurement is *believable*, distinct
from whether it was *obtained*. The first consumer is the 1,000,000 t/s defect
([issue #3](https://github.com/bigattichouse/llama-optimize/issues/3)), but the
mechanism is meant to cover any future case where a driver hands us a number
that cannot be true.

## Defect

A sweep reported `tg=1000000.0 t/s` and crowned that config the winner, with
`pp=444.1` on the same row. Every downstream consumer accepted it: the Pareto
frontier, the main-effects analysis, the suggested `llama-server` command. The
report was not merely wrong, it was confidently wrong — the bogus row dominated
every real one, so the one number a user actually acts on was the one number
that never happened.

Verification did not catch it. `verify_picks` re-measured three times and
reported "median of 3 measurements (spread 0%)" — because the fault is
deterministic, repetition confirmed it instead of exposing it. **Reproducibility
is not validity.** A measurement that is wrong the same way every time passes
every consistency check we have.

## Where the number comes from

The reporter's CSV header carries `threads_batch`, which is `server_only` in
`FACTORS` and is only added to the design when `cfg.driver == "server"`. So this
was a **server-driver** sweep, and the number came from `llama-server`'s own
timings, not from `llama-bench`.

`llama-server` derives the rate as (`server-context.cpp:390`):

```cpp
timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;
```

`ServerSession.measure` read `predicted_per_second` straight out of the response
and averaged it. Nothing in the response can contradict it: the rate, the token
count and the elapsed time all come from the same counter, so if that counter is
wrong every field agrees with every other field.

Solving for the elapsed time at 1e6 t/s gives **1.0 µs/token for any
`n_decoded`** — 128 tokens in 0.128 ms, 64 in 0.064 ms. Real decode on the
reporter's hardware (Ryzen 5700X3D, 8 GB 2060 Super, 35B MoE at Q4_K_M) runs
~25 t/s, or 40 ms/token. The measurement is off by four orders of magnitude, in
the direction only "no work happened" produces.

That `pp=444.1` on the same row is plausible is the tell: prefill ran, decode did
not, and only decode's number is impossible.

`llama-bench` has the same shape of exposure — it divides the *nominal*
`n_prompt + n_gen` by measured nanoseconds (`llama-bench.cpp:1473`) — but its
modern versions `exit(1)` when a decode fails ("bench: handle decode errors",
May 2025), which our driver already reports as `ERROR`. Older builds did not,
so the bench path needs the general gate even though the reported defect came in
through the server.

## Principles

**P1 — Plausibility is separate from success.** `status == "OK"` currently means
"the process exited cleanly and a number parsed". It does not mean the number
can be true. These are different questions and need different checks; a driver
that exits 0 while producing nonsense is exactly the case that slips through.

**P2 — Prefer invariants over calibration.** A threshold tuned to one person's
hardware is wrong on everyone else's. Prefer bounds that follow from how
transformers work and hold on any machine, and reserve absolute constants for
backstops where no invariant is available.

**P3 — Reject conservatively; a false reject is worse than a false accept.** A
rejected row is excluded from the design, and we have already been bitten by
silently deleting valid configurations (the fit-cache collision in PR #4).
Throwing out a real measurement costs a run *and* biases the main effects.
Bounds are therefore set to catch the physically absurd, not the merely
surprising — an order of magnitude of slack is deliberate, not sloppiness.

**P4 — Rejection must be loud and attributable.** A discarded measurement gets
its own status, appears in the CSV, and is counted in the report. Silently
dropping rows would turn one visible defect into an invisible one.

## Invariants

**I1 — Decode cannot outrun prefill.** For one config on one machine,
`tg_tps <= pp_tps`. Prefill consumes many tokens per forward pass; decode
consumes one, re-reading the same weights each time, and is memory-bandwidth
bound. Their ratio is normally 10–100× in prefill's favour. Speculative decoding
and MTP raise `tg` by small integer factors, never above `pp`.

Applied with a 10× margin (`tg > 10 * pp` rejects) so that only the absurd is
caught, per P3. The reporter's row sits at a ratio of **2251×**.

Requires `pp_tps > 0`; when prefill was not measured, I1 does not apply and I2
carries the check alone.

**I2 — Absolute ceiling.** `tg_tps <= MAX_PLAUSIBLE_TPS * parallel`. A backstop
for when I1 is unavailable. Scaled by `--parallel` because the server driver
reports aggregate throughput across concurrent streams, which legitimately
multiplies with stream count.

**I3 — Rejection is total.** A row failing I1 or I2 gets `status =
"IMPLAUSIBLE"` and is excluded everywhere `status == "OK"` gates: the Pareto
frontier, the picks, the max-context probe's base config, and the analyzer's
scoring (where it lands as a 0, the same treatment as OOM — "failure is data").

**I5 — A server's self-reported rate is checked against an independent clock.**
The decisive check, and the only one that addresses the cause rather than the
symptom. Every field in a completion response comes from the server's own
counter, so a wrong counter is internally consistent and unfalsifiable from the
inside. Our wall-clock time around the request is the one number the server does
not supply — and a request cannot have produced tokens faster than it elapsed.
So `predicted_n / wall_seconds` is a hard upper bound on any rate the server may
claim, resting on no hardware assumption at all.

**The wall must be the decode wall.** The bound is only sound while the request
IS decode, and it stopped being that when profiles gained a shared-prefix
fraction (`4ffa97a`): at `prefix_reuse < 1.0` every rep sends a *different*
prompt and re-prefills its differing suffix inside the request being timed. A
server-reported `tg` covers decode alone, so comparing it against a whole-request
wall compares two different spans and rejects honest runs — issue #11, where
17.6s of a 23.5s request was prefill and an honest 43.1 t/s read as "4× faster
than the wall clock permits". Deep context makes it worse in both terms at once:
the prompt to re-prefill grows while the generation stays at `n_gen`.

So the prefill is subtracted before the ceiling is formed, from the same
response's `timings.prompt_ms`, falling back to `(1 − reuse) × prompt_len / pp`
priced off the warm request when the server does not report it.

**Credit from the server, capped.** `prompt_ms` comes off the same clock I5
exists to distrust, so it is not trusted — it is capped at 90% of the wall
(`WALL_PREFILL_CREDIT_MAX`). A server with broken counters can loosen its own
ceiling by at most 10×; issue #3's row overshot by ~1500×. Uncapped, a response
claiming `prompt_ms == wall` drives the denominator to zero and switches the
check off against precisely the fault it was written for. The client-side
fallback needs no cap for P1 — it can only under-credit — but is capped by the
same code path anyway.

Applied with a 2× margin on top, because what is left after the credit is still
not pure decode: it carries the HTTP round trip, the JSON transfer of a prompt
that can be 160 KB, and server-side tokenization — which `prompt_ms` does not
cover, since llama.cpp starts that clock *after* tokenizing. The bound uses the
*kindest* rep, so one slow round trip cannot condemn a run.

Only the single-stream path needs it: the concurrency path already computes its
rates from our wall clock, so it cannot exceed it by construction.

## A rejection must survive its own review

I5 zeroes `pp_tps` and `tg_tps` on the row it rejects — correctly, since anything
reading the numbers before checking status must not use them. But that destroys
the only evidence for deciding whether the rejection was right, and the reason
went to the console and nowhere else. Issue #11 took five round trips with a
reporter for want of numbers that were in the response all along.

Three things follow, and all three are now done:

- The reason travels **in the row** (`implausible` is a results-CSV column), and
  carries the breakdown — tokens, wall, credited prefill — plus the measured `pp`
  the zeroing discards.
- `--report-only` prints discards too, not just the live sweep. It is the one
  command a reporter can run without a GPU or a llama.cpp build.
- `cache_hit` records what the server **delivered**, next to `reuse`, which is
  only what the battery **asked for**. llama.cpp matches on tokens and can
  decline to reuse a prefix the client believes it shares; when it does, a rep
  the tool believes is pure decode pays a full prefill instead, and nothing in
  the row could tell those apart. A large gap between the two is warned about
  rather than rejected: the measurement is real, it just answers a colder
  workload than the one being tuned for.

- `prompt_tok` records the tokens the prompt **became**, next to the `n_depth`
  it was **asked for**. The server driver sizes prompts in characters, so those
  are different numbers on any tokenizer that is not 4 chars/token, and the
  recommended `-c` used to ride on the requested one — emitting a context deeper
  than anything the sweep had actually loaded (see `docs/constants-audit.md`
  C-B).

## A rejection should be as narrow as the fault

I5 originally compared the **mean** rate across reps against the **kindest**
rep's ceiling. Both halves of that are wrong for the same reason: they mix reps
together. Issue #11 ended on a config whose three reps reported ~600, 333,362 and
~620 t/s. The mean carried the outlier into the verdict, the entire row was
zeroed, and the surviving reason — built from an aggregate and one rep's clock —
could not say whether one rep had gone bad or all three had.

A server counter that breaks on one request has produced one bad rep. The other
reps are still a measurement, and throwing them away costs a configuration that
was measured fine.

So each rep is now screened against **its own** request's duration
(`screen_reps`), the survivors decide the number by **median** rather than mean,
and the row is IMPLAUSIBLE only when nothing survived. A partial rejection keeps
its numbers and is therefore not allowed to be silent: `rejected_reps` is a
column and the count is printed when it fires. This is also what the roadmap's
"keep per-rep samples" item was for.

Alongside it, a completion carrying an `error` object is raised rather than
returned. llama-server answers some failures with HTTP 200 and an error body,
and reading timings off a non-result is how zero elapsed time becomes an
infinite rate.

**I4 — The check runs where measurements are born.** Applied in the two places
that construct a result dict from measured numbers — `run_one` (bench driver)
and `measure_in_session` (server driver) — so every caller inherits it: the
sweep, the max-context probe, and pick verification alike. Validating in the
report instead would let the probe and the verifier keep making decisions on
numbers the report would later reject.

## A result is never a general fact

Every number this tool prints is conditioned on a quant, a model, a machine, an
operating point **and the conditions it was taken in** — and the same knob flips
answer across all of them.

A flash-attention comparison here read as a **33% loss** on Mistral-Small-24B
Q8_0 at 4k depth, and it was wrong. The `temp_c` column showed the two rows were
measured at 44 °C and 91 °C: a cold card against a hot one, with `--no-thermal-wait`
disabling the guard that exists for exactly this. Re-run with execution order
randomised and both rows hot, the same comparison came out at **1.03x** — no
difference worth reporting. The retraction is in `docs/constants-audit.md` C-A.

That is why `temp_c` is a column and not a log line, and why the thermal settle is
on by default. A between-run swing here has been measured at ~27%, which is larger
than most effects the sweep is looking for.

### The settle had three ways of silently not settling

All three were found on 2026-09-02 by reading `temp_c` on a comparison that had
the settle **enabled**: two rows at 44 °C and 87 °C, a 12% apparent effect, and
no warning anywhere. The guard was on and doing nothing.

1. **The plateau rule gave up after one slow poll.** "Cooled by less than 0.5 °C
   since the last poll" ended the wait — but at a 3 s poll that is 10 °C/min, a
   rate a hot card passes straight *through* on its way down. It now takes four
   consecutive stalled polls (`THERMAL_PLATEAU_POLLS`), so a card cooling
   steadily is waited out and only a genuine stall exits early.

2. **The 120 s cap was shorter than an MI50 takes to shed a run.** Raised to
   600 s and exposed as `--thermal-cap`, because it is a wall-clock/comparability
   trade and the tool is not entitled to make it silently. With the plateau rule
   fixed the cap is a backstop rather than the usual exit.

3. **The "idle baseline" was one instantaneous sample.** Taken while the card was
   still hot from the previous sweep, it printed `idle baseline 99°C — settle to
   ≤104°C`: a target nothing can ever exceed, so every subsequent run was
   recorded as thermally settled while running at 100 °C. `idle_baseline_c` now
   *measures* it — watching until the reading stops falling, which returns at
   once on an idle card and waits on a cooling one. No "too hot to be idle"
   threshold is involved, deliberately: that number is a property of the card,
   the cooler and the room.

The shared lesson is the one this section is about. A guard that reports nothing
when it fails is indistinguishable from a guard that works, and all three of
these failed silently for as long as they existed. A settle that gives up now
says so, naming the temperature it gave up at.

### A deep prompt of filler alone stops measuring decode

`tg=0.00` with `status=OK` on both deep rows of an `fa` sweep — `pp` fine,
`err_rate=0.0`, `rejected_reps=0`, `secs` 639 against a 1200 s budget. Every rep
returned successfully and reported nothing.

Bracketed rather than guessed, on `Qwen3.6-35B-A3B` (MI50 32GB / ROCm, llama.cpp
`6c84c7d5d`), reading the server's own counters:

| prompt | `prompt_n` | `predicted_n` | `stop_type` | rate |
|---|---|---|---|---|
| filler, shallow | 2,347 | 128 | `limit` | 27.1 t/s |
| filler, deep | 13,058 | **1** | **`eos`** | 0.00 |
| filler, deep **+ task block** | 15,401 (cached) | 128 | `limit` | 26.1 t/s |

**The model emits EOS as its first token.** `truncated = 0`, so it is not context
overflow, and the deadline is not involved. Given thousands of tokens of repeated
filler and no instruction, the model decides the document is finished — which is
a reasonable thing for it to do. The row was honest; there was genuinely no
decode to measure.

Two consequences:

- **`--task` is not only about content realism, it is what keeps deep rows
  measurable.** Ending the prompt with an instruction restores full generation at
  the same depth, in the same context. Sweeps predating `--task` are the ones
  exposed.
- **The diagnosis has to be specific, because the responses are opposite.** A
  config that is slow, or whose counter is broken, is a tuning problem. A model
  that has decided the prompt is over is a *prompt* problem, and no amount of
  retuning `ubatch` touches it. `ended_at_eos` separates them and names `--task`
  in the message.

Ruled out along the way, and worth recording so they are not re-suspected: the
per-config deadline (`secs` well inside budget, no rep cut short), and the tool's
deep prompt construction — the same sweep shape at `n_depth=16384` on
`gemma-3-270m-it-Q8_0`, CPU-only, measured 53.9 tg t/s.

### Warm is the default, because that is where inference runs

Fixing the settle raised a better question: *should the tool be cooling the card
at all?* A number taken from an idle card is a **burst** figure. Sustained serving
runs hot — on an MI50 the card reaches ~100 °C under load and throttles there, and
that throttled rate is the rate a deployment actually gets. Settling to idle
measures a state the workload never occupies.

So `--thermal-mode warm` is the default: before any measured rep, each config is
preheated **with its own workload** until the GPU stops heating, and measured at
that steady state.

The distinction that makes this work, and it is the whole distinction: **warm
means "at its own thermal steady state", not "whatever the last run left
behind".** Simply skipping the cooldown is what produces rows at 44 °C and 87 °C,
because then a config inherits its neighbour's heat and the measurement depends
on execution order. Preheating under the config's own load defines the state by
the config instead, which is why the rows stay comparable.

Two consequences worth stating plainly:

- **Configs plateau at different temperatures, and that is the honest answer.** A
  config that heats harder really does throttle harder in deployment. It is not
  noise to be removed.
- **A burst-fast config can lose a sustained comparison.** For a bursty agent
  workload where the card really is cool at request time, that ranking is wrong
  — which is why `--thermal-mode idle` still exists and still settles to the idle
  baseline.

The preheat reuses the *real* request rather than a cheaper stand-in, because
heat is a property of the workload: a lighter warm-up would plateau somewhere
else and hand back a steady state the measured reps do not share. If the preheat
cap arrives before the card is flat, the row says so — it is a burst number
wearing a sustained label, and that is exactly the kind of silent failure the
section above is about.

Warm mode also costs less wall-clock than it replaces: there is no idle baseline
to establish and no multi-minute cooldown between runs, only a preheat that ends
when the card is flat.

A finding printed without its scope reads as a property of llama.cpp. It then
gets adopted as a default and shipped to people whose hardware disagrees, which
is precisely the history of `FIXED_FA = 1`: measured once, justified in
`DESIGN.md` as "helps gfx906", pinned for everyone
([`constants-audit.md`](constants-audit.md) C-A).

So the report leads with the scope rather than leaving the reader to supply it:

```
RESULTS: 47/125 configs succeeded
For Mistral-Small-24B-Instruct-2501.Q8_0.gguf, on AMD Instinct MI60 / MI50
(ROCm 7.2.1, 32752 MiB), 8 physical cores, server driver, single profile — we find:
```

Each part of that is there because it changes the answer and is not implied by
the others:

- **The quant** rides in the model filename, which is why it is printed whole
  rather than prettified.
- **The card**, not the device slot. `ROCm0` says which index answered; it does
  not say what ran.
- **The backend and its version.** The same card under ROCm, Vulkan and CUDA is
  three different sets of kernels, and they change between backend releases —
  a CUDA reader should see at a glance that a number came from somewhere else,
  and a ROCm 6 reader that it came from ROCm 7. Read best-effort per backend
  (`/opt/rocm/.info/version`, `nvidia-smi`, `vulkaninfo`); an unreadable version
  is omitted rather than guessed.
- **The capacity.** An MI50 ships in 16 GB and 32 GB variants, so it is part of
  the identity rather than a detail beside it.

Rows already carry `tool_version` and `llama_build` for the same reason: a CSV
outlives the terminal it was printed in, and the question it has to answer later
is "what produced this, and on what".

## Nothing measured is not the same as measured zero

A server-driver config whose reps never finished came back `tg = 0.0` with status
`OK`. That looks harmless — `measured_ok` requires a positive score, so the picks
and the Pareto frontier already ignored it. But `factor_level_means` filters on
`status == "OK"` alone, so an **unmeasured row was averaged in as a zero**:

```
rows: ngl=99 -> 40.0 t/s (measured), ngl=99 -> 0.0 (never ran)
main effect reported for ngl=99: 20.0
```

Half the true value, and `refine_factors` reads the same means to decide what the
next `--iterate` pass narrows to. The picks stayed right while the analysis and
the refinement quietly did not — the worst shape a defect can have here, because
nothing in the output looks wrong.

Reproduced by tightening the budget (`--timeout 45` at depth 16384) so the reps
cannot finish: `status=OK tg=0.0 err_rate=0.5`. Note `err_rate` does not catch it
either — a deadline break is not an exception, so nothing was counted as failed.

Fixed at both ends, because the root cause was a disagreement rather than a
missing check: **`measured_ok` required a positive score and `factor_level_means`
did not.** The analysis now uses the same criterion the picks always used, so a
row that produced no number is excluded whatever its recorded status says — which
also repairs a CSV written before the fix, since the status is stored in the file
and cannot be recomputed while the number can be read. Verified on a hand-built
historical CSV: a level whose true mean was 7.06 read as 3.53 and now reads 7.06.

And the server driver names it the way the bench driver already did: **`SLOW`**
when the budget was deliberately tightened by `--min-tgs`, **`TIMEOUT`** when it
simply ran out, with the reason in `too_slow` and printed as it happens. Two ways
in, both covered: no rep finished, or every surviving rep reported 0 t/s. Being
non-`OK` is what keeps such a row out of the main effects, and `NO_RESULT_STATUS`
already counts `TIMEOUT` toward the out-of-bounds detection above, so a level that
can never finish gets reported as such rather than as a slow one.

## A crash is a measurement of a boundary

A `SIGNAL` row is not a gap in the data. It says a parameter set is **out of
bounds** — and the design already visited that region systematically, so the
information is there to be read rather than thrown away.

What makes it readable is the array's balance. Each level appears in several rows
beside *different* partners, so "every row at this level failed" is evidence about
the level. **One failure is not**, and inferring from it is exactly the
generalisation to refuse: a single crash in a Taguchi row implicates the whole
combination, not any one column. `dead_levels` therefore requires at least two
rows at a level before it will call it dead, and the report says which statuses
made up the verdict rather than only that there was one.

Three boundaries on what counts:

- **`SLOW` and `IMPLAUSIBLE` are excluded.** Both had a working launch. `SLOW` is
  a real measurement below a floor the user set; `IMPLAUSIBLE` is a number we
  refused. Neither says anything about whether the configuration *runs*.
  `SIGNAL`, `OOM`, `ERROR`, `TIMEOUT` and `CRASH` do.
- **A factor with every level dead is not reported.** That is the model or the box
  failing, not the knob, and narrowing it would hide the real fault.
- **Only levels actually measured dead are dropped** from the next `--iterate`
  pass. `refine_numeric` invents new neighbours around the winner; those are
  unknowns, and the next pass is what settles them. A dead level at `ngl=16` does
  not license deleting `8` and `24` — even where the dead band turns out to be
  contiguous, as it is on `qwen35` (issue #18).

The filtering happens on the refined **output**, not the input, because
`refine_numeric` spans the old endpoints and would otherwise put a known dead
level straight back.

This is the general form of the special case in `ngl_levels`: rather than knowing
in advance which levels abort on which architecture, the sweep finds the edge and
stops spending passes on it. Where a static rule and this disagree, the
measurement should win.

## Layering

I5 is causal and specific: it catches the defect at its source, with a real
reason attached ("the request's own duration allows at most 639 t/s"). I1 and I2
are the general backstop — they know nothing about servers or requests and
therefore still apply to the bench driver, to `--report-only` re-analysis, and to
whatever produces the next impossible number.

Neither layer is sufficient alone. I5 cannot see a bench-driver result; I1/I2
would accept a server row that is wrong by 5× rather than 2000×. Both are cheap.

## The build itself can be the invalid measurement

Everything above rejects a number that cannot be true. There is a quieter failure
one level up: a number that is entirely possible, correctly measured, and answers
a different question than the one asked.

A llama.cpp built without a GPU backend does not fail. It runs on the CPU and
reports plausible throughput. For a *tuning* tool that is the worst available
shape of wrong: every `-ngl` level measures the same CPU run, the sweep still
crowns a winner, main effects are computed over columns that are secretly equal,
and the recommended command claims layers on a GPU that was never touched.

Observed here on 2026-08-26, not hypothesised. Reconfiguring a stale build
directory after an upstream restructure left `GGML_HIP=OFF` despite
`-DGGML_HIP=ON` on the command line. The same 270M model then measured:

| build | backend | tg |
|---|---|---|
| stale cache, HIP silently off | CPU | 115 t/s |
| HIP actually compiled in | ROCm | 444 t/s |

3.9x, no error, no warning, and `llama-bench` names the backend in a column
nobody was reading.

**`gpu_visibility` diagnoses it from evidence already collected.**
`detect_vram_mib` asks llama.cpp first and falls back to `rocm-smi`/`nvidia-smi`,
so the *source* of the answer is the whole diagnosis and no second probe is
needed:

- llama.cpp listed a device → nothing to say
- a vendor tool sees a GPU that llama.cpp does not → **`blind`**: the build cannot
  reach the card this machine has
- nobody reports a GPU → `cpu-only`, a legitimate sweep
- the binary predates `--list-devices` → `unknown`; llama.cpp's silence is not
  evidence, so this must not be reported as a broken build

**It warns; it does not refuse.** CPU-only sweeps are a real use case, and
arguably the one where tuning matters most — `threads`, `ubatch`, `numa` and
affinity are the whole game when there is no GPU to hide behind. What must never
pass silently is a CPU-only *build* on a machine that has a GPU, which is a
mistake rather than a choice. The two are distinguishable and are reported
differently: an alarm for the first, a one-line note for the second saying which
factors cannot vary.

The warning names both causes it cannot separate — a missing backend and a GPU
hidden by `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` — because guessing between
them would send half of readers down the wrong path.

## The harness can create the effect it measures

A sibling of the section above, and the subtler of the two. There the *build*
answered a different question than the one asked; here the *request shape* does.

`ServerSession.measure` sends one identical prompt for the warm request and every
rep, deliberately: it keeps the prefill out of the measured window so a rep is
pure decode. That is sound for `-ngl` or `-ub`, which do not care what the tokens
say. It is not sound for any parameter that feeds on repetition — and n-gram
speculation is exactly that.

Measured 2026-08-26: on a repeated prompt, ngram reaches 100% draft acceptance and
better than doubles tg; on distinct prompts it never drafts at all. The full table
and consequences are in
[`workload-shape-design.md`](workload-shape-design.md). The harness was not
measuring ngram's value — it was manufacturing the conditions under which ngram
looks best, then reporting the result.

The lesson generalises past ngram: **a measurement is only valid for the request
shape it was taken under, and the harness's shape is a choice, not a neutral
default.** A factor whose payoff depends on how requests relate to each other
cannot be measured by a loop that sends the same request every time — it will
read as either free or worthless depending on which way the loop happens to
lean, and both readings will be reproducible.

There is no check that catches this class automatically the way `gpu_visibility`
catches a CPU-only build; it was caught because `draft_acc` recorded acceptance
and the number was implausibly perfect. That is the argument for recording
diagnostics you do not yet know you need.

**And it had a second layer.** Fixing the harness to send genuinely distinct
requests did not fix the measurement: acceptance stayed at 1.00 at every reuse
level, including 0%. The prompts differed *from each other* while each one was
built by tiling a short passage to length, and n-gram speculation feeds on
repetition wherever it occurs — including inside a single context. The generator
was manufacturing the effect it was being used to measure, one level below where
anyone was looking.

So: **"the requests differ" is not the same as "the text is varied."** A prompt
generator is part of the measurement apparatus and deserves the same suspicion as
a timer. Both layers were only visible because acceptance was recorded and the
number was too good.

## "OK" is not the same as "measured"

A run can complete cleanly and still generate nothing. `implausible_reason`
deliberately passes on `tg <= 0` — "not a measurement; other paths own it" — so
such a row keeps `status == "OK"` with a zero score. That is correct: rejecting it
as *implausible* would be the wrong verdict, since nothing impossible happened.

The gap is what happens downstream. `status == "OK"` is a fine filter where rows
are **counted**, and the wrong one where a row is **selected**. `pick_recommendations`
computes `longest` as `max(ok, key=(depth, score))` — depth first, score only as a
tie-break — so a zero-throughput row at the deepest depth wins outright and is
handed to the user as the max-context recommendation: a command that loads and
then produces nothing.

`measured_ok` is the shared filter for selection paths: OK **and** a score worth
acting on. `pareto_frontier` already applied it inline; `pick_recommendations` and
the max-context probe seed did not, and now do. Counting paths (the "N/M configs
succeeded" line) deliberately still use plain `status`, because a run that
completed did complete.

Found by reading another tool's methodology docs rather than by hitting it: a
Bayesian vLLM autotuner documents discarding zero-measurement configurations "to
prevent spurious Pareto dominance" ([`field-reports.md`](field-reports.md) F6).
Our Pareto was already guarded; our *recommendations* were not, which is the more
user-visible of the two.

## A partial failure is a measurement

The mirror of the section above. There a row was *too readily* believed; here one
was too readily thrown away.

A concurrent round used `ThreadPoolExecutor.map`, which re-raises the first
exception while iterating, so a single failed request out of `--parallel N` blew
up the round and the config landed as `ERROR`. That is the same verdict a model
that will not load receives, and the two are not the same thing: one config
cannot run at all, the other serves most of its traffic quickly and drops some.
A tuning tool that cannot tell them apart cannot warn about the second.

Failures are now counted (`err_rate`), and only an all-requests-failed config is
an `ERROR`.

**The throughput penalty is intrinsic and was left alone.** A round's tokens are
divided by the round's wall clock, and that clock covers the failed requests too —
so a config that drops a quarter of its work measures roughly a quarter slower
without any explicit penalty. Adding a score multiplier on top would double-count
the same fact. This is worth stating because the obvious "fix" is the wrong one:
the honest penalty was already there, hidden in the denominator.

What the number *cannot* express is the difference between "slower" and "faster
but lossy" — identical `eff_tps`, very different things to deploy. That
distinction is surfaced as a warning on the recommended command rather than
folded into the score, because folding it in would make the score mean two things
at once.

## Some numbers survive conditions the headline number does not

`temp_c` and the thermal settle exist because throughput drifts. What was not
obvious until it was measured is that **not every recorded number drifts with
it**, and that changes which one to trust when conditions are imperfect.

Measuring MTP acceptance across prefix-reuse levels on a 27B model, once forward
and once in reverse: acceptance reproduced to ±0.02 at every level, while the
same configuration's throughput differed by 78% between the two runs as the card
went from 92 °C to 96 °C. Throughput fell monotonically with *time* in both
directions — which, run once, looks exactly like an effect of the thing being
varied.

Two consequences worth keeping:

- **Prefer the invariant signal for the question it can answer.** "Is the drafter
  working, and how well?" is answerable from `draft_acc` under thermal conditions
  that would make a throughput comparison meaningless. `draft_acc` was added as a
  guard against silently-off speculation (F1); it turns out to be the more robust
  instrument for speculative questions generally.
- **A monotonic trend in a sequentially-run series is not evidence.** It took one
  reversed re-run to separate a real effect from thermal drift, and reversal is
  cheap. The sweep already randomises run order and settles between runs; ad-hoc
  measurements taken outside that machinery need the control done by hand.

## What this does not do

It does not explain *why* generation returned early on the reporter's
configuration. That is upstream behaviour, plausibly a config that cannot
actually run (35B MoE, `ngl=20`, `-ncmoe 40`, 8 GB VRAM, 8192-token depth)
failing in a way that neither sets an error nor prints anything `_OOM_PAT`
recognises. Diagnosing it needs the reporter's full CSV; the issue attachment is
truncated to a header and a single OOM row.

The gate is deliberately independent of that answer. Whatever makes a driver
emit an impossible number, the tool's obligation is the same: do not report it
as the best configuration.
