# Multi-GPU tuning — design & work log

Concrete plan for sweeping llama.cpp's multi-GPU placement knobs. The
*principles* — why gated parameters cannot ride a flat orthogonal array, and the
staged mechanism that fixes it — live in
[`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md), which already names this case
as a future consumer. This file is the multi-GPU-specific factor model and task
checklist, the same split ngram uses in
[`ngram-design.md`](ngram-design.md).

## Origin

[Issue #5](https://github.com/bigattichouse/llama-optimize/issues/5): two
non-identical GPUs, and llama.cpp's automatic fit splits layers roughly evenly
even though one card is faster. The ask is exactly what a measured sweep answers
and a heuristic cannot — the optimal split depends on the two cards' relative
bandwidth, the model, and the context depth, all at once.

## Defect

`FACTORS` has no entry for any placement knob, so `-sm`, `-ts` and `-mg` are
invisible to the tool: they are neither swept nor settable. `--factor` rejects
names absent from the registry, so there is no user-side workaround either — a
multi-GPU owner cannot ask the question at all today.

Two prerequisites are already fixed. `detect_vram_mib` read only the first card
(2edff6c), which both understated the machine and mis-scoped the OOM pruner's
limit; its parsing now returns per-device figures, which is also what `-ts` level
generation needs. And the `--list-devices` parser argued for below is now built
and used for capacity — see the checklist, and note that issue #7 got there
first, on one GPU.

## Ask llama.cpp for the devices, not the vendor tool

`-ts` proportions are positional: the *i*-th value belongs to llama.cpp's *i*-th
device. So the device list we derive them from must be llama.cpp's, and
`rocm-smi`/`nvidia-smi` are the wrong source for it.

**The trap is specific to this issue's hardware.** `nvidia-smi` enumerates by PCI
bus ID. The CUDA runtime's default ordering is *fastest-first*, unless
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. On two **identical** cards the two orders
agree and nothing goes wrong; on two **different** cards — precisely issue #5 —
index 1 in `nvidia-smi` can be index 0 to llama.cpp. Generating `-ts` from
smi-ordered capacities would then hand the larger share to the wrong GPU, and the
result would look like a plain bad measurement rather than a mix-up.

`--list-devices` avoids the whole class of problem:

```
Available devices:
  ROCm0: AMD Instinct MI50/MI60 (32752 MiB, 32730 MiB free)
```

Issue #5's reporter supplied theirs, which is the mixed pair this is all about:

```
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24117 MiB, 6672 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (11909 MiB, 9837 MiB free)
```

Three things that data settles:

- The CUDA and ROCm lines are the **same shape**, so one parser genuinely
  covers both — the assumption this design rests on, now confirmed rather than
  assumed.
- A VRAM-proportional starting split is roughly **2:1** (24117 : 11909), which
  is a long way from the even split llama.cpp picks by default. There is real
  headroom here, which is what makes the sweep worth running.
- **`free` here was a snapshot, not the resting state.** The 6672-of-24117 line
  on CUDA0 read like a display or another workload permanently holding two
  thirds of the card, and an earlier draft of this file concluded from it that
  level generation must use *free* rather than total. The reporter has since
  [corrected that](https://github.com/bigattichouse/llama-optimize/issues/5#issuecomment-5471917817):
  a model was loaded when they captured it, and at rest roughly 1 GiB is used per
  card. So `ts_levels()` derives from **total**, which is the stable property of
  the hardware; free VRAM stays what `headroom_warning` made it — a warning about
  the instant the sweep starts, not an input to the design. Worth keeping as a
  reminder that a single reading of a device list describes a moment, not a box.

It also demonstrates the ordering hazard concretely: the 3090 is the larger and
faster card, so getting the order backwards would hand two thirds of the model to
the 3060.

It is the authoritative order (the one `-ts` and `-dev` index into), it carries
per-device total **and free** VRAM, and it is vendor-neutral — `ROCm0`, `CUDA0`,
`Vulkan0` all parse identically, so one parser replaces the two vendor paths and
NVIDIA boxes stop being the untested branch. It also costs one cheap subprocess
call on a binary we have already resolved.

The smi path stays for `vram_used_mib` sampling during a run, which
`--list-devices` cannot do (it exits immediately). For *capacity and identity*,
llama.cpp is the source of truth.

**M0 — Device order comes from `--list-devices`.** No placement factor is
generated from smi-derived ordering.

## The factor model

| Gate `-sm` | meaning | conditional children |
|---|---|---|
| `none` | one GPU only | `-mg` (which GPU) |
| `layer` | split by layer (default) | `-ts` (proportions) |
| `row` | split by row | `-ts`, `-mg` (KV/intermediate home) |
| `tensor` | split by tensor | `-ts` |

This is the same *mode selector plus mode-specific knobs* shape as ngram, so it
needs `active_when`/`flag_for` **data**, not new mechanism. `plan_stages` already
orders a screen stage (gate at defaults) ahead of per-value tuning stages.

Two things are genuinely new and are where the work is:

**N1 — `-ts` is a vector, not a scalar.** Every existing factor takes a scalar
level; `-ts` takes proportions per device (`3,1`). It enters the registry as a
categorical whose levels are pre-rendered strings, with the level *set* generated
from detected hardware rather than fixed — the same way `ngl_levels` and
`cpu_offload_levels` derive from the model and box.

Proposed generation for N devices with detected capacities `c_i`:

- `1,1,…` — even, the baseline llama.cpp already picks
- `c_0,c_1,…` — proportional to VRAM, the obvious prior
- two or three interpolations either side of proportional, spanning **well
  past** it — field evidence has the optimum at 1:4 where capacity said 1:2
  ([`field-reports.md`](field-reports.md), F3)

Levels must be **normalised and deduplicated** (`2,2` and `1,1` are the same
split), or the OA wastes rows measuring one configuration twice and the main
effect for `-ts` is computed over columns that are secretly equal.

**N2 — the gate interacts with the memory model.** Under `-sm none` the usable
capacity is one card, not the summed total that `detect_vram_mib` now returns.
Any consumer of `hw["vram"]` that runs per-config — the OOM pruner especially —
needs per-device capacity plus the active split to compute a real limit, or it
will repeat the mis-scoping bug in a new form: right total, wrong distribution.
This is the part to design carefully; a wrong verdict here silently deletes
configurations.

The single-pool assumption is not hypothetical — it is in the code today.
`parse_fit_print` sums llama-fit-params' per-device lines into one scalar, and
`predict_fits` compares that scalar against summed total VRAM. For "does a 35B
fit across 36 GiB" that is the right question and it answers it correctly. For
issue #5's 24 GiB + 12 GiB pair it is the wrong one: a config can clear the
summed total and still overflow the 3060. The per-device breakdown needed to fix
it is already in the text being parsed and is discarded a line before it would be
useful, so the fix is a change of return type rather than of source — but every
caller then has to say *which* device it is asking about, which is why this is
the risky checklist item and not a small one.

## Invariants

- **M1 — No inert columns.** `-ts` is emitted only in rows whose `-sm` splits,
  and `-mg` only under `none`/`row`. Inherited from I2/I3 in
  `CONDITIONAL-FACTORS.md`; `validate_factor_registry` should reject a
  multi-GPU registry that violates it.
- **M2 — Single-GPU boxes see nothing.** With one device detected, no placement
  factor enters the design and no stage is planned. Multi-GPU support must cost
  a single-GPU user exactly zero runs. Mirrors how `numa` is only swept on a
  machine with multiple NUMA nodes.
- **M3 — Levels are distinct after normalisation.** Per N1.
- **M4 — Capacity is per-device.** No consumer compares an all-device footprint
  against one card's capacity, or vice versa. Per N2.

## Cost

`-sm` has 3–4 levels and `-ts` perhaps 5. Flat, that is ~20 extra rows of an OA
mostly measuring inert flags. Staged, it is one screen stage over `-sm` at
default `-ts`, then a single tuning stage over `-ts` for the winning mode — the
same shape that took ngram from 9 knobs to 3 collapsed ones and L125 back to L25.

## Open questions

1. Is `-sm tensor` worth a level, or too new to be widely available? Gate on a
   `--help` probe like other capability checks, or just include it and let a
   failed row score 0.
2. Should `-ts` levels be generated from detected VRAM or measured bandwidth?
   VRAM is what we can see; bandwidth is what actually determines the optimum,
   and the two disagree exactly in issue #5's case (one card faster, not bigger).
   A cheap per-device bandwidth probe may be the more honest input — and since
   the sweep measures anyway, a VRAM-proportional prior plus interpolations
   either side lets the measurement settle it without us guessing.
   A mixed Vega64+MI50 setup in [`field-reports.md`](field-reports.md) (F3)
   is evidence for the bandwidth side: its tuned split favours the faster
   card twice as hard as capacity alone would justify.
3. Does `-mg` deserve sweeping under `row`, or is it a tie-breaker better left
   pinned?

## Checklist

- [x] `--list-devices` parser (`parse_list_devices`/`list_devices`), returning
      per-device id, name, total and free MiB in llama.cpp's order. Landed ahead
      of the rest because [issue #7](https://github.com/bigattichouse/llama-optimize/issues/7)
      needed it for a *single*-GPU reason: on an APU the vendor tool reports the
      2 GiB VRAM carve-out, not the ~30 GiB of GTT the model actually runs in,
      and the OOM pruner skipped nearly the whole sweep. `detect_vram_mib` now
      asks llama.cpp first and falls back to smi, which satisfies **M0** for
      capacity. Ordering is carried but nothing consumes it yet.
- [ ] Per-device capacities recorded in `hw` (parsing already returns them)
- [ ] `ts_levels()` derives from per-device **total** VRAM; free VRAM is a
      start-of-run warning only (see the correction above)
- [ ] `M2` guard: placement factors only when >1 device is detected
- [ ] `-sm` registry entry (gate) + `-ts`/`-mg` with `active_when`
- [ ] `ts_levels()` generator + normalisation/dedup, with tests
- [ ] `validate_factor_registry` coverage for M1
- [ ] `parse_fit_print` returns per-device figures instead of a sum, and
      `predict_fits` compares each against that device's capacity — prerequisite
      for `-ts`, since a split changes the distribution and not the total
- [ ] OOM pruner made split-aware (N2/M4) — the risky one
- [ ] Report: show the chosen split and per-device footprint
- [ ] Validate end-to-end on a real two-GPU box (we do not have one — issue #5's
      reporter is the natural tester)

## Testing without the hardware

Development happens on a single AMD card, so the NVIDIA paths are the ones that
rot untested — the reason `--list-devices` parsing is worth more than two vendor
branches. Three tiers, in increasing cost:

1. **Parser tests on captured output** (free, in `--selftest`). Real
   `--list-devices` and `nvidia-smi` text pasted verbatim, single- and
   multi-device, is enough to pin ordering, per-device capacity and the
   normalise/dedup rules. Everything above except the actual split behaviour is
   testable this way.
2. **A single NVIDIA box** confirms the CUDA branch parses real output and that
   `-sm none` plus `-mg` behave — worth doing before asking anyone else to run
   it, since it catches vendor-format surprises that captured strings cannot.
   `--selftest` needs no GPU; a `--quick --no-probe` sweep on a small model
   exercises the real path.
3. **Two non-identical GPUs** is the only way to test the thing that matters —
   the split itself, and the device-ordering trap above, which by construction
   cannot reproduce on one card or on two identical ones.

Tier 2 is worth doing early even on old hardware: an old NVIDIA card still
enumerates through the CUDA branch, so it validates parsing and flag emission
even if its measured throughput is uninteresting.
