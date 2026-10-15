# Load timing, load estimates, and streamed requests — design & work log

How long a server takes to come up, what it is doing while it does, and what a
request is doing before it returns. Principles live in [`DESIGN.md`](DESIGN.md);
this is the concrete model and checklist, the same split
[`sweep-cost-design.md`](sweep-cost-design.md) uses.

## Origin

2026-10-14. Timing the user's everyday config (`llama-server.sh`:
`Qwen3.6-27B-UD-Q6_K_XL`, `-ngl 99`, MTP n-max 2, q8_0 KV, `-b 1024 -ub 256`)
through the tool's own server path, MI50 32GB, llama.cpp b11639:

| depth | load | config | result |
|---|---|---|---|
| 8192 | **212 s** (cold) | 101 s | OK, tg 17.8, pp 192, MTP acceptance 0.88 |
| 32768 | 5 s (page cache) | 439 s | OK, tg 9.8, pp 94, acceptance 0.92 |

The user's question: "those settings do work, and work well in general" — so a
config that loads in 212 s against a 180 s `--server-start-timeout` is a working
config the sweep would file as failed. And from outside, nothing distinguished
"still reading 24 GB off disk" from "hung" from "generating slowly": a request
was a black box until it returned.

Three things follow, and they are separable.

## L1 — record what a load spent its time on

The session's stderr is already drained line by line (C5). Each line is now
stamped on arrival with our own clock, and the load is split at llama.cpp's own
stage markers:

| phase | from | to |
|---|---|---|
| `start` | process spawn | `load_tensors: loading model tensors` |
| `weights` | that | `llama_context: constructing` |
| `ready` | that | `/health` answers |

`weights` is reading and uploading the model; `ready` is the context, KV
allocation and warmup. The `… model buffer size = N MiB` lines say how much of
the model landed on each device, which is the upload half of `weights`.

Our clock rather than llama.cpp's line timestamps: they are relative to its own
start in a format (`M.SS.mmm.uuu`) that is not documented as stable, and
arrival time is what the start deadline is measured in anyway.

## L2 — estimate a load before paying for it

```
load_s ≈ uncached_mib / disk_mib_s + model_mib / upload_mib_s + fixed_s
```

- `uncached_mib` — the part of the GGUF not in the page cache, from
  `fincore` (util-linux). The 212 s → 5 s pair above is this term: the same
  24.2 GB, cold then cached. No `fincore`: assume nothing is cached (the
  safe direction for a timeout).
- `disk_mib_s`, `upload_mib_s`, `fixed_s` — **learned per box**, from loads the
  tool already performs: a load with a large uncached share prices the disk, a
  fully cached one prices upload + fixed. Kept in a small JSON next to the
  results (`load-rates.json`) and updated as a running estimate; priors are
  deliberately pessimistic until measured.

This is a guess and is labelled as one. What it is for:

- **The start timeout is derived, not fixed.** `max(--server-start-timeout,
  3 × estimate)`, so a cold 26 GB model gets the time it needs and a 270M one
  still fails fast. The flag stays the floor and the explicit override.
- **The sweep's ETA can include loads**, which `estimate_secs_per_run` prices
  today with `6 + 2.5 × GiB` — a constant that cannot know whether the file is
  cached, i.e. whether the answer is 5 s or 212 s.

## L3 — stream requests, so a request is not a black box

`/completion` with `"stream": true` returns server-sent events, one per token,
with the final event carrying the same `timings` a non-streamed request does.
Reading them as they arrive gives, from our own clock:

- **when prefill ended** — the first token's arrival;
- **how far generation got** — tokens so far, when a request is cut off;
- **the decode rate** — tokens over time since the first token, measured by
  us, not reported by the server.

That last one is the real fix for the wall-clock cross-check (I5/C5): the check
exists because the server's own counter could not be trusted, and a client-side
rate between first and last token needs no prefill credit and no overhead
credit at all.

And it delivers what `--tgs-timeout` was always read as: after that many
seconds of decode, judge the average rate against `--min-tgs` and stop the
request if it is below — regardless of `--n-gen`.

## L4 — something to watch while it runs

User, while a 1h+ timing run sat silent: "if we ran in screen could we possibly
watch it instead of just waiting". Two plain files beside the results CSV:

- `<results>.live` — one line, rewritten in place (atomic, ≤ 4×/s): the
  config, the step (`warm-up request`, `preheat`, `rep 2/3`, `server`), the
  phase (`loading: weights`, `prefill`, `decoding`), tokens so far, live rate,
  elapsed. `watch -n1 cat …live`. The TTY ticker shows the same fields.
- `<results>.server.log` — the server's own log, copied as the drain reads it.
  `tail -f`.

Files rather than a socket: every terminal, `screen` and `tmux` can already
watch a file, and nothing yet exists that would read a socket.

**And a disk probe instead of a disk prior.** "a quick … load test … to
benchmark when the system loads": on a box with no learned disk rate and a
model that is not in the page cache, 1 GiB of the model is read at startup with
`O_DIRECT` (page cache bypassed). Measured here: 128 MiB/s in 8 s, predicting
~194 s to read the 24.2 GiB model cold, against a measured cold load of 212 s.
A few MB would measure latency and readahead, not throughput. RAM→GPU upload is
not probed — Python has no portable way to time it without a GPU library — and
is learned from the first real load instead; on this box it is the small term
(the same model loads in 5 s once cached).

## L5 — wait on progress, not on a clock

User: "when you launch the load … have that run in screen or some other job
system so the software can peek on the output instead of having to wait until
the end". The drain thread (C5) already reads the server as it writes — the
same thing screen does with a pty — so the peeking exists; what was missing
was *acting* on it. Two fixed clocks became progress-aware:

**Loads.** `--server-start-timeout` (now `max(flag, 3 × estimate)`) is a soft
deadline. Past it, a load still showing progress — a new log line, or more of
the GGUF in page cache (`fincore`, every 5 s) — keeps waiting; it is abandoned
after `LOAD_STALL_S` (90 s) with no progress, or at `LOAD_HARD_CAP_FACTOR` (4×)
the deadline. The live view shows the weights stage as a percentage read.

**Prefill.** `/completion` with `return_progress` streams prompt-progress
events (`processed`, `total`, `time_ms`) once per batch, so prefill is no
longer silent. Measured motivation, the user's Qwen3.6-27B config:

| depth | config time | result under the old fixed `--timeout 1200` |
|---|---|---|
| 65536 | 900 s | OK |
| 131072 | 1204 s | **TIMEOUT** — prefill alone needs ~1870 s at ~70 t/s |

That row was working the whole time. Now: prefill may run past `--timeout`
while it is visibly advancing (same stall and cap rules), and decode then gets
`--timeout` of its own from the first token — so a trickling decoder is still
bounded and a slow prefill does not eat the decode's time. When the warm
request's prefill itself overran the config deadline, the reps (cheap: the
prompt is cached) get their own budget rather than a TIMEOUT for a config that
just did the hard part. `--min-pps` is judged during prefill the way
`--min-tgs` is during decode.

### Open: the sweep's 64k depth cap

`DEFAULT_MAX_DEPTH = 65536` caps the sweep's depth grid regardless of the
model's native context, from before prefill was observable ("deep contexts on
CPU prefill very slowly"). The user: "a long context is one of the things I
would expect users to want to test". The max-context probe already goes to the
native ceiling, and `scripts/time-config.py` ignores the cap. Lifting it for
the grid is a cost decision, not a correctness one: on Qwen3.6-27B a 131k row
is ~34 min, and an L125 puts 25 rows at its top depth.

## Invariants

- **LS1 — a slow load is not a failed config.** A server that is still loading
  when its deadline passes is `TIMEOUT`, and the deadline scales with what
  there is to load.
- **LS2 — every number here is measured on this box or labelled a guess.**
  Rates are learned from this machine's own loads; until then the estimate
  says it is a prior.
- **LS3 — streaming changes how a request is read, never what it asks for.**
  The same prompt, `n_predict`, sampling and cache flags; the server's final
  `timings` still recorded, so a streamed row is comparable to an old one.

## Checklist

- [x] L1: stage markers timed on arrival; `load_phases` per session; printed per launch
- [x] L2: `fincore` cached share; `load-rates.json` learned rates; estimate
- [x] L2: start deadline = `max(flag, 3 × estimate)`
- [x] L2: `O_DIRECT` disk probe seeds the disk rate on a box with none learned
- [x] L3: streamed `_completion`; first-token time; client-side decode rate,
      which the wall-clock cross-check now uses in place of its credits
- [x] L3: the warm request judged on streamed decode after `--tgs-timeout`
- [x] L4: `<results>.live` and `<results>.server.log`; ticker shows the step
- [x] L5: progress-aware load wait (log lines + page cache), stall + hard cap
- [x] L5: prefill progress events; soft `--timeout` during prefill; `--min-pps` live
- [x] Sweep ETA prices loads from measured rates, cold read shown separately
- [x] Prefill ETA in the live view; early stop when even the optimistic
      projection misses the cap
- [x] `scripts/time-config.py`: one config, model-agnostic depth grid
- [ ] Decide the sweep's 64k depth cap (above)
- [ ] Measured on hardware: cold vs cached load of a large model, and a
      streamed vs non-streamed rate on the same config
