# Flag coverage — what the tool can and cannot explore

A triaged inventory of every `llama-server` and `llama-bench` parameter against
the `FACTORS` registry, so "can this tool find my best config?" has an auditable
answer instead of an optimistic one.

Produced 2026-08-26 against llama.cpp `4d19b2876` (ROCm 7.2.1 / gfx906 build) by
diffing `--help` output against the flags the registry and the command builders
emit. All 37 flags the registry emits were re-validated as still accepted by that
build; none has been removed. The method matters more than the snapshot — see
[Keeping this honest](#keeping-this-honest).

## Why coverage is the product

Every other design doc here asks "is this knob worth sweeping?". This one asks
the inverse and more important question: **what can a user not even ask us?**
A knob we sweep badly shows up as a disappointing number. A knob we cannot
express is invisible — the user never learns the question existed, and the tool
reports a confident optimum from a space that excluded their answer.

`--factor` rejects names absent from the registry, so registry coverage *is* user
reachability. There is no escape hatch.

## Two live defects

**C1 — FIXED (0.2.0-dev). We emitted a deprecated flag on every bench run.** `bench_command` emits
`-mmp 1` unconditionally (`FIXED_MMAP`), and `llama-bench --help` now says
`-mmp, --mmap <0|1> (DEPRECATED IN FAVOUR OF --load-mode)`. The server path is
the same family: `--no-mmap`/`--mlock` are deprecated in favour of `-lm`.

This is not cosmetic. llama.cpp **does** remove deprecated arguments — three
removals are visible in the same help text we parse (`--draft`, `--draft-min`,
`--spec-ngram-size-n`, all now "the argument has been removed"). The 221-commit
pull to `4d19b2876` added two more data points on the same trend line: upstream
is actively migrating internal usage off these flags
(`0177dcc73 common: migrate the deprecated --mmap/--no-mmap to --load-mode`), and
removed another CLI argument outright in the same batch
(`9fee29e94 arg: remove -no-cnv from cli`). When `-mmp`
follows them, every bench run fails at argument parsing, which is the whole
sweep, not one row. `common/arg.cpp:884` also warns that `-lm` and
`--mmap`/`--mlock`/`--direct-io` must not be combined — "only the last flag on
the command line will take effect" — so a user passing `-lm` today gets a silent
conflict with a flag we inserted for them.

**C2 — `--parallel` silently pins `kv_unified`, and we always pass it.**
`-kvu/--kv-unified` documents its default as "enabled if number of slots is
auto". The mechanism is `tools/server/server.cpp:153`: `kv_unified = true` is set
**only** when `n_parallel < 0`. The struct default is `false`
(`common/common.h:563`).

`build_server_args` passes `--parallel` explicitly whenever concurrency is
non-trivial, and `parallel` is itself a factor. So slots are never auto in the
runs where it matters, and **every concurrency sweep we have ever run measured
`kv_unified = false`** — while a user who starts `llama-server` without
`--parallel` gets unified KV and 4 slots. We have never measured the regime the
default lands in, and cannot: there is no factor for it.

This is the [`CONSTRAINED-FACTORS.md`](CONSTRAINED-FACTORS.md) shape again — one
factor silently determining another's value — and it deserves the same treatment
rather than a note.

## Genuinely uncovered, perf-relevant

Ranked by expected effect on this project's reference hardware (partial offload,
model spilling to system RAM), which is where several of these bite hardest.

| Flag | What it does | Why it matters here |
|---|---|---|
| `-lm`, `--load-mode` | `auto`/`none`/`mmap`/`mlock`/`mmap+mlock`/`dio` | Supersedes the mmap/mlock/dio knobs we *fix*. On partial offload the CPU-resident half is exactly what pages, and `mlock` vs `mmap` is a throughput decision, not a preference. Also the fix for C1 |
| `--repack` / `-nr` | weight repacking → `no_extra_bufts` | Repacks weights into CPU-optimized buffer layouts. Only pays for CPU-resident tensors — i.e. precisely the layers `-ngl` leaves behind |
| `-kvu` / `--kv-unified` | one KV buffer shared across sequences | See C2. Changes both memory footprint and multi-slot behaviour at `--parallel > 1` |
| `--op-offload` / `-nopo` | offload host tensor ops to device (default on) | Decides whether CPU-resident tensors' operations ship to the GPU. A partial-offload-specific lever, and in both drivers |
| `--no-host` | bypass host buffer, allowing extra buffers | Pairs with `--repack` (both steer buffer-type selection); in both drivers |
| `-sps`, `--slot-prompt-similarity` | how closely a prompt must match a slot to reuse it | The server-side prefix-reuse routing knob. Belongs with [`workload-shape-design.md`](workload-shape-design.md) — it is unmeasurable for the same reason `--cache-reuse` is |
| `--swa-full` | full-size SWA cache | Memory/speed trade on sliding-window models (Gemma and friends), invisible on others — a conditional factor by nature |
| `-bs`, `--backend-sampling` | offload sampling to the backend | Experimental, default off. Sampling cost is a fixed per-token tax, so it matters most exactly where tg is already high |
| `--prio`, `--prio-batch` | process/thread priority | Real on a contended box — which, per this project's own notes, is the normal case, not the exception |
| `-Cb`, `-Crb`, `--cpu-strict-batch`, `--poll-batch` | batch-phase CPU affinity twins | We sweep `threads_batch` (`-tb`) but none of its affinity siblings. An internal inconsistency: we already believe the batch phase deserves separate CPU tuning |
| `-ctxcp`, `-cms` | context checkpoints, min step | Recompute cost on context shift — bites at the high-depth end where throughput already sags |
| RoPE/YaRN tail | `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`, `--yarn-attn-factor`, `--yarn-beta-fast/slow` | We sweep `rope_scaling` and `yarn_factor` and stop. An incomplete family reads as a deliberate scope, and isn't |

Multimodal is a whole uncovered workload rather than a flag: `--mmproj-offload`
is a genuine placement decision (the projector competes for the same VRAM),
`-mmdev`/`--mmproj-device` (new in `4d19b2876`) makes that placement explicit
per-device, and
`--image-min-tokens`/`--image-max-tokens`/`--mtmd-batch-max-tokens` set the
request shape for image traffic. Embedding/rerank serving (`--pooling`,
`--embeddings`, `--rerank`) is likewise a real workload the tool has no profile
for. Both are out of scope *today*, but as absent workloads, not absent knobs —
noting them here so that stays a decision.

## Covered by designs already written

- **Multi-GPU** — `-sm`, `-ts`, `-mg`, `-dev`:
  [`multi-gpu-design.md`](multi-gpu-design.md)
- **Draft model** — `-md` and the ~12-flag `--spec-draft-*` mirror:
  [`draft-model-design.md`](draft-model-design.md)
- **Workload shape** — `--cache-reuse`, `-cram`, `--cache-prompt`,
  `--cache-idle-slots`, and now `-sps`:
  [`workload-shape-design.md`](workload-shape-design.md)

## Correctly excluded, with the reason

Recorded so the exclusion is a decision rather than an oversight, and so this
list can be an allow-list the audit checks against.

- **Sampling** (`--temp`, `--top-p`, `--min-p`, mirostat, DRY, XTC, penalties,
  grammar/JSON-schema): throughput is independent of sampling, and
  [`DESIGN.md`](DESIGN.md) already keeps the recommended values for the emitted
  command. **Caveat:** `--reasoning-budget`, `--reasoning-effort` (new in
  `4d19b2876`) and `--ignore-eos` change how many tokens are actually generated,
  so they perturb measured t/s without being tuning knobs — they are hazards, not
  factors.
- **Transport and API** (`--host`, `--port`, `--api-key`, CORS, SSL,
  `--api-prefix`, web UI, MCP/agent/tools): no effect on decode.
  `--threads-http` is the marginal case — it could matter at high concurrency,
  but the HTTP layer is not what saturates first.
- **Logging/introspection** (`--log-*`, `--metrics`, `--props`, `--slots`,
  `--verbose`): `--perf` and `--metrics` add measurable overhead and are
  correctly left off rather than swept.
- **Model acquisition** (`-hf`, `-mu`, `-dr`, `--offline`, the `--*-default`
  presets, `--check-tensors`): upstream of tuning.
- **Adapters** (`--lora*`, `--control-vector*`): change the model, not its
  placement.
- **Deprecated aliases** (`-dt/--defrag-thold`, `--mlock`, `-dio`, `-mmp`):
  superseded — see C1.
- **Measurement hazards, not knobs**: `--no-warmup` (we do our own),
  `--sleep-idle-seconds` (a server that sleeps between our runs would corrupt
  timings), `--slot-save-path`.

## Keeping this honest

A hand-written inventory is accurate the day it is written and rots from then on.
llama.cpp gained `-lm`, `--repack`, `-kvu`, `-nopo` and `--no-host` without us
noticing; the next batch will arrive the same way.

The rate is now measured rather than assumed. Refreshing this audit across one
routine pull — 221 commits, `1d2869c6e` to `4d19b2876`, thirteen days — moved:

| | |
|---|---|
| flags added | 2 (`-mmdev`/`--mmproj-device`, `--reasoning-effort`) |
| flags removed | 0 (but `-no-cnv` went from the CLI in the same batch) |
| cited line numbers that drifted | 4 of 12 re-checked |

Nothing in this document's conclusions changed, which is the point: the cost of
*checking* was a scripted diff, and the cost of *not* checking would have been
discovering a removed `-mmp` when a user's sweep died at argument parsing. Line
numbers drifted at a third the rate of a single pull, so citations here name the
symbol as well as the line wherever the symbol is stable enough to grep.

The fix is to make coverage checkable: parse `llama-server --help` and
`llama-bench --help`, diff against `FACTORS` plus the emitted-fixed flags, and
report anything in neither that list nor the explicit exclusion allow-list above.
Unknown flags then surface as a prompt to triage rather than as silent gaps. The
`--help` parser already exists in another form — `supports_flag` and
`_help_cache` are the capability probe, and this is the same input read for a
different question.

Two properties it must have to be worth anything: the exclusion list lives in
code next to the registry (not in this file, which cannot be executed), and an
unrecognised flag is *reported*, never auto-added — a factor nobody reasoned
about is how inert columns get created.

## Checklist

- [x] **C1** — `load_mode_args` emits `--load-mode`, probed via `supports_flag`,
      falling back to the per-driver legacy spellings on older builds
- [ ] `load_mode` as a *factor* (mmap / mlock / mmap+mlock / dio), now that
      emitting it is settled — the partial-offload case is where it should pay
- [x] **C2** — designed: [`concurrency-kv-design.md`](concurrency-kv-design.md).
      `--parallel` and `kv_unified` are a constrained pair, and the audit
      understated it — single-stream sweeps run 4 auto slots with unified KV,
      concurrency sweeps never see unified at all
- [x] `--repack`, `--op-offload`, `--no-host`, `--swa-full`, `--backend-sampling`,
      `--prio`/`--prio-batch`, the batch-phase CPU affinity twins, and
      `-ctxcp`/`-cms` — all registered and reachable via `--factor`, none
      auto-swept ([`remaining-factors-design.md`](remaining-factors-design.md), R1)
- [x] `--load-mode` as a factor, with the fixed-emission interlock
- [ ] `-sps` folded into [`workload-shape-design.md`](workload-shape-design.md)
- [ ] Auto-sweep `swa_full` if SWA turns out to be detectable from GGUF metadata
- [ ] Decide the RoPE/YaRN tail: complete the family or state the scope
- [ ] `--audit-flags` mode + exclusion allow-list in code, with `--selftest`
      coverage over captured `--help` text (no binary, no GPU)
