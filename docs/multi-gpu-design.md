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

One prerequisite is already fixed: `detect_vram_mib` read only the first card
(2edff6c), which both understated the machine and mis-scoped the OOM pruner's
limit. Its parsing now returns per-device figures, which is also what `-ts` level
generation needs.

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
`ncmoe_levels` derive from the model and box.

Proposed generation for N devices with detected capacities `c_i`:

- `1,1,…` — even, the baseline llama.cpp already picks
- `c_0,c_1,…` — proportional to VRAM, the obvious prior
- two or three interpolations either side of proportional

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
3. Does `-mg` deserve sweeping under `row`, or is it a tie-breaker better left
   pinned?

## Checklist

- [ ] Per-device capacities recorded in `hw` (parsing already returns them)
- [ ] `M2` guard: placement factors only when >1 device is detected
- [ ] `-sm` registry entry (gate) + `-ts`/`-mg` with `active_when`
- [ ] `ts_levels()` generator + normalisation/dedup, with tests
- [ ] `validate_factor_registry` coverage for M1
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
