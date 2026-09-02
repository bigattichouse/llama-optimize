# Community sweeps

A place to post sweep results together with the machine that produced them, so
the next person can look before they spend four hours measuring.

## Why this directory exists

Every number this tool prints is conditioned on a quant, a model, a card, a
backend *and its version*, and the thermal state it was taken in. That is not
pedantry — it is the difference between a useful result and a misleading one:

- Flash attention measured **33% slower** on Mistral-Small-24B Q8_0 on an MI50,
  and the same box has seen it *help* on other models.
- Partial `-ngl` **segfaults** on hybrid SSM models under ROCm — but only bare;
  `--spec-type draft-eagle3` lifts the whole dead band.
- Prefix reuse collapses past a depth that varies by architecture.
- An EAGLE3 head drafted at 47% acceptance and bought **1.00x** — on one head,
  against one target, on one card.

Each of those would be settled in a week by three people with different cards,
and cannot be settled on any single box. That is what this directory is for.

## Posting a sweep

Every sweep writes a fingerprint beside its results:

```
llama-optimize.py --run model.gguf --results mysweep.csv
# -> mysweep.csv
# -> mysweep.csv.fingerprint.json
```

Then:

```
gzip -k mysweep.csv
```

and open a PR adding both files under `community/<card>/<model>/`, or attach
them to an issue. `--fingerprint` prints the same JSON without running a sweep,
if you want to look at it first.

## What the fingerprint contains, and what it does not

It records **hardware and software identity**: CPU model, core counts and RAM;
each GPU with its VRAM, backend and backend version; OS and architecture; the
llama.cpp build; the model's *basename*, quant and architecture flags; and how
the sweep was run.

It contains **no paths, no hostname and no username**, by construction and by
test — a model path leaks a directory layout and usually a login name, while the
basename and quant are what actually condition the result. Nothing leaves your
machine automatically; the tool prints an instruction, never uploads. The file is
plain JSON and short enough to read in full before you post it.

## Reading someone else's

Nobody's box is identical to yours, so the question is never equality but
distance. Roughly, in decreasing order of how much it tells you:

| | |
|---|---|
| same card, same backend major version | directly usable |
| same card, different backend major (ROCm 6 vs 7) | usable, but kernels changed — re-confirm the winner |
| same architecture family, different card | the *shape* transfers (which knobs matter), the numbers do not |
| different architecture | a hint about what to sweep, nothing more |

`schema` is versioned so these files stay readable as the tool changes. Adding a
field does not bump it; changing or removing one does.
