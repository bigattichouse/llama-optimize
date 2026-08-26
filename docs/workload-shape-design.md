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

## Cost

Cheap in factors, which is unusual here. One input, two gated factors, no new
array columns unless the user opts into the shape. The cost is in the generator
and in re-deriving W-I2, not in run count.

`--cache-ram` is the exception and is deliberately last: it only does anything
when slots are evicted and restored, which needs *rotating distinct
conversations*, not one prefix with varying tails. That is a second shape
dimension and should not be smuggled into the first one.

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

- [ ] `--prefix-reuse PCT` input on `Config`, defaulted per profile (W-D2)
- [ ] Prompt generator: one sweep-stable prefix + per-request fresh suffix,
      sharing `_realistic_prompt`'s prose (W-D3)
- [ ] `ServerSession.measure`: distinct suffix per rep and per concurrent slot,
      cache on, with the pp/tg contract stated at the call site (W-I1)
- [ ] Re-derive `TG_OVER_PP_LIMIT` under partial reuse, or scope it to the shape
      it was written for (W-I2) — **before** any factor lands, since a wrong
      limit deletes configs silently
- [ ] `--cache-reuse` registry entry, gated on non-zero reuse (W-I3)
- [ ] `--cache-ram` and a rotating-conversation shape — separate, later
- [x] Check whether n-gram state survives across requests — **it does**, and the
      distortion is 2.3-3.4x (table above)
- [ ] Re-measure the ngram screen under distinct prompts once `--prefix-reuse`
      lands, and say plainly in the report that earlier ngram numbers were upper
      bounds
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
