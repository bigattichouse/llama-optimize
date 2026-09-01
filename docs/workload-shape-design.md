# Workload shape — design & work log

Concrete plan for letting a sweep describe the *shape* of the traffic it is
tuning for, not just its size. The staging and gating principles it leans on live
in [`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md); the measurement contract it
must not break lives in [`measurement-validity.md`](measurement-validity.md).

## Origin

[`field-reports.md`](field-reports.md) F4. The largest single number in the whole
field-report set is a prefix cache: 381 tok/s reproducing document context against
124 tok/s otherwise. llama.cpp's analogues are `--cache-reuse N`
(`common/arg.cpp` ~3543) and `--cache-ram` (~1705), and neither is in `FACTORS`.

Adding them today would return a confident null, and the reason is the interesting
part.

## Defect

The tool measures exactly one request shape. `ServerSession.measure` sends the
**same prompt** every rep with `cache_prompt=true` — deliberately, so a
single-stream rep is pure decode with no re-prefill. The concurrency path sends
that same prompt to every slot. So the prefix is already 100% reused and
byte-identical across every request a sweep ever issues.

Under that traffic, `--cache-reuse` — which recovers *partial* prefix overlap
between *differing* prompts — has nothing to recover. It would measure zero, and
the zero would be an artifact of our generator rather than a property of the
knob.

**This is a class, not a knob.** `PROFILES` has three entries (`single`,
`agents`, `multi`) and they vary only `n_prompt`, `n_gen`, `ctx_floor` and
`parallel` — the *size* of a request and how many run at once. Nothing describes
how requests relate to each other. Any parameter whose payoff depends on that
relationship is invisible to the tool by construction, and prefix caching is
simply the one with the biggest number attached.

The tool's promise is "find the best parameters for your workload". It currently
delivers "for a workload of your size", which is a different and much weaker
claim.

## Shape is an input, not a factor

The distinction the rest of this design rests on.

A **factor** is something the user can change about how llama.cpp runs, and the
sweep's job is to pick its best level. A **shape** is something true about the
user's traffic, which they cannot choose and we must not optimize away. Prefix
reuse is the second kind: an agent stack with a 6k-token system prompt has ~90%
reuse whether or not that is convenient, and the right `--cache-reuse` for them
is the one that wins *at their reuse level*.

So shape joins `n_prompt`/`n_gen`/`parallel` as a request-profile input. Sweeping
it would be a category error — it would report "your workload should have more
prefix reuse", which is not advice.

**W-D1 — one shape dimension, described numerically.** `--prefix-reuse PCT`
(default 0, preserving today's behaviour where it matters): the fraction of each
prompt that is a stable prefix shared across requests, with the remainder
generated fresh per request. One number covers the real span — 0% is independent
requests, ~90% is an agent or app with a fixed system prompt, and the field
report's document-reproduction case is the high end.

**W-D2 — profiles set a default, the flag overrides it.** Same precedence the
tool already uses (`built-in < use-case < explicit flag`). The `agents` profile —
8192-token prompts, 32768 ctx floor — is where a long shared system prompt
actually lives, so it is the natural home for a non-zero default. Users whose
traffic does not look like any profile give the number directly, which is the
generic path and should stay first-class rather than an escape hatch.

**W-D3 — the shared prefix must be shared *content*, not a shared length.**
Generating `pct` tokens of fresh prose per request and calling it a prefix
measures nothing: llama.cpp matches on tokens. The prefix is generated once per
sweep and reused verbatim; only the suffix varies. `_realistic_prompt` already
builds varied prose for a reason (speculative acceptance realism) and the same
generator serves both halves.

## What this does to the measurement

The part that needs care, and the reason this is a design doc rather than a patch.

Today's single-stream reps measure **pure decode**: prefill happens once in the
warm request, and `cache_prompt=true` on an identical prompt means no rep pays
for it. With a varying suffix, every rep prefills that suffix. `tg` stops being a
clean decode number and starts including prefill work.

**W-I1 — a profile states its measurement semantics; nothing inherits them
silently.** At `--prefix-reuse 0` the current identical-prompt reuse is no longer
defensible either — it is why the shape axis was invisible in the first place. The
honest split is: the *shared prefix* is cached (that is the point), the *suffix*
is not, and `pp`/`tg` are reported against that contract explicitly.

**W-I2 — the validity checks are re-derived, not assumed to carry.**
`implausible_reason` rejects `tg > 10 × pp` on the reasoning that decode cannot
outrun batched prefill. That reasoning was formed under pure-decode reps. Under
partial reuse the measured `pp` covers only the uncached suffix while `tg` is
unchanged, which moves the ratio in the direction of the limit — a real
high-reuse config could approach a bound written for a different measurement.
The limit needs re-deriving against the new shape before it can be trusted to
reject only the absurd (P3: rejecting a real measurement silently deletes a
config from the design, the worse error).

**W-I3 — no inert columns.** `--cache-reuse` and `--cache-ram` enter the registry
gated on a non-zero `--prefix-reuse`, the same way draft factors gate on a draft
model ([`draft-model-design.md`](draft-model-design.md), DM1). At reuse 0 they
would be columns whose every level gives an identical run.

## Confirmed: every ngram measurement so far was taken in a regime real traffic does not reproduce

This started as an open question — n-gram speculation feeds on repetition, every
rep of every sweep has sent a byte-identical prompt, so *if* n-gram state survives
between requests, measured acceptance is inflated and the ngram screen has been
grading on a curve.

Measured on 2026-08-26 (gemma-3-270m-it-Q8_0, ROCm, `--spec-type ngram-mod`,
`temperature 0`, `cache_prompt: true` — the harness's own request shape). Three
identical requests in a row, then three distinct ones:

| request | `draft_n` | accepted | tg t/s |
|---|---|---|---|
| identical prompt, 1st | *absent* | — | 273 |
| identical prompt, 2nd | 53 | 53 (100%) | **601** |
| identical prompt, 3rd | 53 | 53 (100%) | **578** |
| distinct prompt A | *absent* | — | 176 |
| distinct prompt B | *absent* | — | 259 |
| distinct prompt C | *absent* | — | 236 |

State persists across requests, and the effect is not subtle. On a repeated
prompt the drafter reaches **100% acceptance** and better than doubles throughput.
On genuinely distinct prompts it **never drafts at all**.

**This maps exactly onto `ServerSession.measure`.** It issues a warm request and
then `reps` more, all with the same prompt. The warm request is row 1 above — cold
state, no speculation, and discarded. Every *measured* rep is row 2 or later, in
the saturated regime. So the numbers behind every ngram result this tool has
produced sit at a ceiling that a real workload reaches only when it re-sends
prompts verbatim.

Two consequences, and the second is worse than the first:

- **ngram-vs-off is inflated**, by something like 2.3-3.4x on this model. That is
  the comparison users actually act on.
- **ngram-variant-vs-variant is compressed.** Every variant saturates at 100%, so
  the screen that picks between `ngram-simple`, `ngram-mod`, `ngram-map-k` and
  `ngram-map-k4v` has been ranking configurations at a shared ceiling, where the
  differences it is trying to resolve cannot appear.

MTP is very likely unaffected — it drafts from the model's own NextN head rather
than from context n-grams — but that has not been measured and should not be
assumed.

**This turns `--prefix-reuse` from a capability into a correction.** The shape
input is no longer only "let users describe their traffic"; it is the mechanism
by which ngram gets measured honestly at all. Until it lands, ngram findings
should be treated as upper bounds, not estimates.

Worth noting how this was found: `draft_acc` ([`field-reports.md`](field-reports.md)
F1) had been recording for less than a day. The absent-`draft_n` signal — the one
built to catch a *config* that silently did not speculate — is what made a
*harness* that silently over-speculates visible in a single six-request run.

## Landed, and what it measured

`--prefix-reuse PCT` is implemented, along with the category-weighted battery and
achieved-reuse reporting. Profiles default it: 0 everywhere except `agents` at
90, whose name is itself a claim about traffic.

**These numbers were corrected twice before they were right, and the corrections
are the useful part** — each one was a defect in the measuring apparatus, not in
llama.cpp.

Final figures (gemma-3-270m-it-Q8_0 / ROCm, `--spec-type ngram-mod`, 512-token
prompts, run interleaved so thermal drift cannot masquerade as an effect):

| reuse | drafted | accepted | generated | acc | **cov** | tg t/s |
|---|---|---|---|---|---|---|
| 100% (identical requests) | 124 | 104 | 128 | 0.84 | **0.81** | **846** |
| 90% | 59 | 59 | 128 | 1.00 | **0.46** | 579 |
| 0% | 62 | 62 | 128 | 1.00 | **0.48** | 578 |

Identical requests inflate ngram throughput by **~1.46x** (846 vs 579), and
speculative coverage by **~1.75x**. Earlier drafts of this document said 2.3x;
that figure came from a contaminated generator and is withdrawn.

### Acceptance rate alone ranks configs backwards

The instrument needed fixing before the measurement meant anything.

`draft_acc` is accepted/drafted — draft *quality*. Read the table: the two
*slowest* configurations score a perfect **1.00**, and the fastest scores 0.84. A
drafter that is always right about the few tokens it dares to guess is not
helping much.

What tracks throughput is `accepted / generated` — the fraction of output tokens
that came free from speculation. 0.81 → 846 t/s, 0.46 → 579, 0.48 → 578. That is
now recorded as **`draft_cov`** alongside `draft_acc`; both are kept, because
quality and contribution are different questions and the pair distinguishes "the
drafter is wrong a lot" from "the drafter barely tries".

### MTP is not affected — measured, not assumed

The open question the plan was blocked on. Qwen3.8-27B-UD-Q6_K_XL,
`--spec-type draft-mtp --spec-draft-n-max 4`, 512-token prompts, 3 reps, run once
forward through the reuse levels and once in reverse:

| reuse | acceptance (fwd) | acceptance (rev) | tg (fwd) | tg (rev) |
|---|---|---|---|---|
| 100% | 0.78 | 0.78 | 32.7 | 18.4 |
| 90% | 0.70 | 0.70 | 30.4 | 17.4 |
| 50% | 0.68 | 0.68 | 29.2 | 22.7 |
| 0% | 0.72 | 0.74 | 27.3 | 28.2 |

**Acceptance is stable and reproducible to ±0.02**, and it moves ~8 points across
the whole reuse range (0.78 → 0.70). Compare n-gram over the same range: 1.00 →
0.31, sixty-nine points, with no drafting at all in between. MTP drafts from the
model's own NextN head rather than from cross-request n-gram state, and the
measurement bears that out.

**So MTP results stand and only ngram needs the advisory.** That was the word
"probably" being converted into a number, which is the whole reason the check
came first.

### The throughput column is thermal drift, and the reverse run is what proved it

Read the tg columns above. Forward, tg falls 32.7 → 27.3 as reuse decreases —
which looks exactly like a reuse effect. Reversed, it falls 28.2 → 18.4 as reuse
*increases*. It declines with **time** in both directions, not with reuse. GPU
temperature climbed 92 °C → 96 °C across the sequence, and sustained tg nearly
halved.

Two things follow.

**Acceptance is thermally invariant; throughput is not.** The same reuse level
gave 0.78 acceptance in both runs while its throughput differed by 78%. For any
speculative question, `draft_acc` is therefore the more robust signal — it
answers "is the drafter working?" without needing the thermal conditions to
match. This is a good reason to have recorded it beyond the issue-#8 guard it was
built for.

**The earlier tables here were taken back-to-back without settle.** Their
*absolute* throughput figures are only loosely comparable within a sequence. The
ngram finding survives that caveat because it is an acceptance result — 1.00 →
0.31 — and acceptance does not drift with temperature; the 2.3x throughput figure
should be read as indicative rather than precise.

The tool already randomises run order and settles thermally between runs for
exactly this reason. These ad-hoc measurements did neither, and a monotonic
confound appeared immediately. That is a point in favour of the machinery, not
against the finding.

### The generator was contaminated twice, at two different scales

Worth recording in full, because each layer looked like a finding until it was
measured.

**First layer — tiled text.** Prompts were built by repeating a short passage to
length, so requests differed *from each other* while each was internally
self-similar. Acceptance stayed pinned at 1.00 at every reuse level including 0%.

**Second layer — a corpus too small to fill a prompt.** After switching to
composition from a sentence pool, the pool held **14 sentences / 1,529
characters**. A 512-token prompt needs 2,048; the `agents` profile's 8,192-token
prompts need 32,768, so the entire corpus repeated about twenty times inside
every one of them. Acceptance went straight back to 1.00 at 0% reuse. The fix was
not a longer literal but a **combinatorial** one: sentences generated from
interchangeable fragments, ~14,400 distinct from a few lines of source.

`repeated_fraction` now measures the property directly — the share of a text's
8-grams that are repeats. Tiled text scores near 1.0; the current generator
scores 0.000 at 256 and 512 tokens and 0.094 at 8,192.

**The lesson, three times over: "the requests differ" is not "the text is
varied", and "the text is varied" is not "the corpus is large enough to stay
varied at this length."** A prompt generator is measuring apparatus. It gets a
measurable acceptance criterion, like any other instrument.



Worth recording because it was missed twice and only a measurement caught it.

The first battery built each prompt by tiling a short passage to length. Requests
then differed *from each other* while each one was heavily self-similar — and
n-gram speculation feeds on repetition wherever it finds it, including inside a
single context. Acceptance stayed pinned at **1.00 across every reuse level,
including 0%**, which looked like "reuse does not matter" and actually meant "the
prompts are still pathological".

`_fill` now composes from a shuffled sentence pool instead of repeating, so
repetition appears only at a long period. `_realistic_prompt` had the identical
defect — its docstring claimed "varied prose so speculative-decoding acceptance is
realistic" while it tiled `_CORPUS` — and now shares the same builder.

**The general lesson: "the requests differ" is not the same as "the text is
varied", and only the second one makes speculative measurement honest.** A
prompt generator is part of the measurement apparatus, and this one was quietly
manufacturing the effect it was used to measure.

### The default is still the unrealistic one, deliberately

`--prefix-reuse` defaults to 100 — today's behaviour, exactly. Shipping the
capability and silently redefining every measurement in the same release are
separable, and the second deserves its own decision with a GPU re-measurement of
the ngram screen behind it. The preamble now says so on every server run, and
[`../CHANGELOG.md`](../CHANGELOG.md) carries the advisory.

## Cost

Cheap in factors, which is unusual here. One input, two gated factors, no new
array columns unless the user opts into the shape. The cost is in the generator
and in re-deriving W-I2, not in run count.

`--cache-ram` is the exception and is deliberately last: it only does anything
when slots are evicted and restored, which needs *rotating distinct
conversations*, not one prefix with varying tails. That is a second shape
dimension and should not be smuggled into the first one.

## Prefix reuse is a property of the model, not only of the request

The battery can shape the request. It cannot make the server able to reuse the
result. Issue #11's reporter measured 0% delivered reuse against a requested 90%,
at every depth, and with `--prefix-reuse 100` — which should have made all four
requests byte-identical — it changed nothing.

The cause is the model. Their GGUF reports `general.architecture = qwen35` with
`qwen35.ssm.*` metadata: a **hybrid SSM/attention** model. llama.cpp cannot roll a
recurrent state back to an arbitrary prefix, so `cache_prompt` falls through to
context checkpoints, and when none covers the prefix the server does this
(`tools/server/server-context.cpp`, "forcing full prompt re-processing due to lack
of cache data (likely due to SWA or hybrid/recurrent memory)"):

```
pos_next = 0; n_past = 0;
```

Every rep then pays a full prefill. That is not a defect in the tool or in
llama.cpp — it is what the architecture costs — but it means the `agents` profile
measures a workload nobody asked for whenever the model is hybrid/recurrent or
SWA, and it does so silently unless `cache_hit` is being read.

### Measured: SWA is fixable, recurrent is not (issue #15)

The paragraph above lumped SWA and hybrid/recurrent together and guessed that
`--ctx-checkpoints` was the lever. Both halves were wrong. Measured on
gemma-3-270m (SWA, window 512) with the battery's own shape — 15.7k-token prompt,
90% requested reuse, warm plus two reps:

| server flags | warm | rep1 | rep2 |
|---|---|---|---|
| default (`-ctxcp 32`) | 0% | 90% | **0%** |
| `-ctxcp 0` | 0% | 90% | **0%** |
| `-ctxcp 128` | 0% | 90% | **0%** |
| `-ctxcp 512` | 0% | 90% | **0%** |
| `--cache-ram 8192` | 0% | 90% | **0%** |
| **`--swa-full`** | 0% | 90% | **90%** |

Three things fall out of that table.

**The failure is intermittent, not total.** rep1 reuses and rep2 does not. After
rep1, the sliding window has scrolled past the shared prefix, so rep2's
`pos_min` clears `pos_min_thold` and the checkpoint search decides the whole
prompt must be reprocessed. A `cache_hit` averaged over reps therefore lands
somewhere in the middle and looks like partial reuse rather than a switch that
flipped — which is how this hid.

**Depth decides whether it happens at all.** The same model at a 1,960-token
prompt reuses 90% on every rep. The first control run was too shallow to see it,
and it is exactly the deep-context profiles (`agents`) that cross the line.

**`--ctx-checkpoints` is not the lever, `--swa-full` is.** Checkpoints made no
difference at 0, 32, 128 or 512, and neither did `--cache-ram`. `--swa-full`
fixes it outright. It is not free — the reused reps run at 388 t/s prefill
against 4283 t/s without it, because attention no longer takes the sliding-window
shortcut — but it replaces a 15.7k-token re-prefill with a 1.6k-token one, worth
roughly 3-6x on this workload. A trade with a real cost on both sides is a thing
to **measure**, so `swa_full` is now swept automatically whenever
`{arch}.attention.sliding_window` is present in the GGUF (gemma3, gemma4,
muse-glimmer here; absent on qwen35).

**Hybrid/recurrent models have no such knob.** There is no `--swa-full` for a
recurrent state: it cannot be rolled back to an arbitrary prefix at all, which is
what the source says and what the reporter's 0%-at-every-depth shows. For that
class, prefix reuse is genuinely unavailable, and the honest thing is to say so
rather than sweep a knob that cannot help.

Consequences:

- The miss warning names this cause first. Its previous advice ("try a smaller
  `--n-depth`") is the *second* cause, and was wrong here: reuse was 0% at depth
  8192 with ~6k tokens of `n_ctx` to spare.
- It also explains the reporter's earlier result that `--prefix-reuse 100` did not
  clear the rejection. It was never going to: the delivered reuse was 0 either
  way, so the reps re-prefilled and the wall clock stayed mostly prefill.

## Open questions

1. Is `--prefix-reuse` per-request-shape enough, or does arrival pattern (bursty
   vs steady) belong in the same axis? Bursts change queueing, not cache hits —
   probably a separate concern, and one `--parallel` half-covers already.
2. Should the max-context probe use the shaped prompt or the plain one? It
   answers "what context loads", which is a memory question — plain is probably
   right, but the two must not disagree about what `n_prompt` means.
3. Does `--cache-reuse` interact with `n_depth`? Both move how much of the
   context is already resident; if their effects are not separable the pair may
   need the constrained-factor treatment
   ([`CONSTRAINED-FACTORS.md`](CONSTRAINED-FACTORS.md)) rather than two
   independent columns.

## Checklist

- [x] `--prefix-reuse PCT` input on `Config` (defaults to 100 = historical
      behaviour; per-profile defaults are the follow-up, W-D2)
- [x] Report the reuse fraction actually **achieved** — `achieved_reuse` measures
      the prompts we built rather than echoing the request, and lands in the CSV
      as `reuse` ([`field-reports.md`](field-reports.md) F6)
- [x] Category-weighted battery (40% short-QA / 20% reasoning / 20% code /
      15% RAG / 5% long-context). Prompt *text* only — per-category output
      lengths would redefine what a single tg number means and are still open
- [x] Prompt generator: one sweep-stable prefix + per-request fresh suffix —
      and composed from shuffled sentences, because tiling a passage made every
      prompt self-similar enough to keep acceptance at 100% (W-D3)
- [x] `ServerSession.measure`: distinct suffix per rep and per concurrent slot
- [ ] Re-derive `TG_OVER_PP_LIMIT` under partial reuse, or scope it to the shape
      it was written for (W-I2) — **before** any factor lands, since a wrong
      limit deletes configs silently
- [ ] `--cache-reuse` registry entry, gated on non-zero reuse (W-I3)
- [ ] `--cache-ram` and a rotating-conversation shape — separate, later
- [x] Check whether n-gram state survives across requests — **it does**, and the
      distortion is 2.3-3.4x (table above)
- [ ] Flip the default to per-profile realistic reuse, and re-measure the ngram
      screen — the remaining half of this work, and the part that needs a GPU
- [ ] `--selftest`: generator produces a byte-identical prefix and distinct
      suffixes; the gate keeps cache factors out of the design at reuse 0; the
      revised validity limit accepts a plausible high-reuse row and still rejects
      issue #3's 1e6 t/s — all stdlib-only, no GPU

## Testing without a long sweep

The generator, the gating and the validity limit are all pure functions and belong
in `--selftest` — which is where the real risk is, since W-I2 is a change to what
the tool will *reject*, and a too-tight limit fails silently by deleting rows.

The one thing selftest cannot answer is whether `--cache-reuse` actually moves at
a given reuse level, and that is a single short server sweep on any model rather
than a full design — worth running before the factor is enabled by default, so
the tool never ships a knob it has not seen do anything.
