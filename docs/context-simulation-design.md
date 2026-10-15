# Context simulation — design & work log

Measuring a config at a deep context without paying for the prefill that
builds it. Principles live in [`DESIGN.md`](DESIGN.md); this is the concrete
model, the evidence, and the checklist.

## Origin

2026-10-15. The user's everyday config (`Qwen3.6-27B-UD-Q6_K_XL`, MTP, q8_0
KV, MI50) needs ~34 minutes of prefill to reach 131k tokens
([`load-and-stream-design.md`](load-and-stream-design.md)) — and a sweep pays
that per row. User: "any way to simulate the KV/attention fill instead of
actually walking through it?", then "if we don't care about it being nonsense,
load a short one, then modify the file to be the larger equiv full of repeated
or random info", and "a setting for 'test context with generation' would be an
alternative with this simulation/calculation as the default, a sort of dry-run".
Goal stated: "whip through morris/sobol/taguchi and get very close to real
numbers".

## Two questions that were one

| question | needs a filled cache? | how |
|---|---|---|
| does depth N **fit**? | **no** | llama.cpp allocates the whole KV for `-c N` at load. A server that starts with `-c N` holds N. Seconds. |
| decode **speed** at depth N | yes — attention reads N cells | restore a saved (or synthesized) slot |
| prefill **speed** at depth N | yes, by definition | only a real prefill measures it |

The max-context probe answered the first question with a full prefill and
measurement per step. That is the load-only test with 30 minutes added.

## Mechanism: llama-server slot save/restore

`--slot-save-path DIR`, then `POST /slots/0?action=save|restore`
`{"filename": ...}`. The file (GGSQ v4) is self-describing:

```
u32 magic 'ggsq', u32 version, u32 n
i32 tokens[n]   — llama-server's block: [-1, version, count, ids..., n_media]
memory sections, in order:
  KV section (one; two for iSWA: full then SWA):
    u32 n_stream; per stream: u32 cell_count, then
      meta  per cell: i32 pos, u32 n_seq, [12-byte ext], i32 seq[n_seq]
            ext = (x, y, token) for M-RoPE / PLE models; (pos, pos, tok) for text
      data  u32 v_trans, u32 n_layer, u32 n_rot_k, u32 n_rot_v,
            per layer K: i32 type, u64 row, row x cells
            per layer V: same (or transposed: i32 type, u32 el, u32 n_embd, ...)
  recurrent section (hybrid models) — fixed size, independent of depth
  appended context checkpoints (llama-server)
```

Read: `src/llama-context.cpp state_seq_save_file`,
`src/llama-kv-cache.cpp state_write*`, `tools/server/server-common.cpp
server_tokens::serialize`.

**Synthesis** (`expand`): N cells with positions 0..N-1, the source's K/V rows,
ext and token ids repeated cyclically, the recurrent tail copied unchanged.
The content is nonsense by design — decode cost depends on how many cells
attention reads, not what they hold. 3.2 GB written in 2.1 s.

## Evidence

All on `Qwen3.6-27B-UD-Q6_K_XL` (hybrid: 16 attention layers + 48 recurrent,
M-RoPE ext), MI50 32GB, llama.cpp b11639, f16 KV, `-ngl 99`, no speculation.
Saved from one config (`-ub 512 -t 8`), restored into another (`-ub 2048 -t 4`).

| comparison | time to depth | decode | gap |
|---|---|---|---|
| restored vs real prefill, 20,690 tok, **no thermal control** | 0.4 s vs 189 s | 7.3 vs 4.8 t/s | +51% |
| same, both preheated 90 s | 0.5 s vs 182 s | 7.2 vs 6.2 t/s | +17% |
| same, both preheated 240 s, measured at 94 / 95 °C | 0.4 s vs 182 s | 6.85 vs 6.80 t/s | **+0.7%** |
| **synthesized 41,380 (from the 20,690 slot) vs real prefill of the same token ids**, 95 / 95 °C | **0.8 s vs 417 s** | 6.65 vs 6.70 t/s | **−0.8%** |

A restore into a different KV type (q8_0 server, f16 slot) is refused with
HTTP 400 — the safe failure.

**The first two rows are the lesson, not the result.** The +51% and +17% were
the card's temperature: the real path had just run the GPU flat out for three
minutes, the restored one had not. On this MI50 the cool-vs-hot swing is ~1.8×.
Any comparison here must match temperatures, and record them.

### The three open checks (2026-10-15)

Same protocol: 240 s preheat on both paths, median of three 64-token decodes,
temperature recorded at measurement.

| check | model | synthesized / restored | real prefill | gap |
|---|---|---|---|---|
| **MTP** (`draft-mtp`, n-max 2), 20,690 tok | Qwen3.6-27B | 13.72 t/s, draft 117/141, 96 °C | 12.32 t/s, draft 117/141, 97 °C | **+11.4% ✗** |
| **iSWA**, expanded 4,115 → 31,000 | gemma-3-270m | 267.1 t/s, 96 °C | 270.2 t/s, 96 °C | −1.1% ✓ |
| **split KV, slot 1** (`--parallel 2`), 31,000 | gemma-3-270m | 268.4 t/s, 95 °C | 265.3 t/s, 96 °C | +1.2% ✓ |

**MTP fails, and the source says why.** llama-server keeps a separate draft
context per slot (`ctx_dft`) and the slot-save path writes `ctx_tgt` only
(`server-context.cpp`, `llama_state_seq_save_file(ctx_tgt, ...)`). After a
restore the draft has no history, so each draft step attends over nothing and
is cheaper: same acceptance (117/141 both ways — it predicts as well), ~11%
faster. A dry run would overstate every speculative config at depth by about
that much on this model. The server's in-RAM prompt cache (`--cache-ram`)
*does* save draft state (`cur->data.drft`), so a llama.cpp change to include
it in slot files looks small; until then speculative rows need a real prefill
or an explicit "optimistic" label.

**iSWA works once the window is handled.** gemma's file has two KV sections:
the full layers (4,115 cells, positions 0..4,114) and the sliding window
(512 cells stored as a ring). Expansion grows the first to N and keeps the
window at 512, moved to N-512..N-1. The 15.7 MB tail is appended context
checkpoints, copied unchanged.

**Split KV needs the target's layout.** A file saved from a one-slot server
has one stream; a `--parallel 2` server without `--kv-unified` has one stream
per slot and refused it (`n_stream mismatch`). Writing the file with the
target's stream count — the saved slot's stream filled, the others empty, as
llama.cpp itself writes it — restores and decodes within 1.2%. With
`--kv-unified` the original file restores into any slot.

Also noted, not chased: the split-KV run's "real" side reported a near-zero
prefill time (the one-slot gemma run took 4 s); its decode still ran over the
full 31k (`cache_n=30999`), which is the quantity compared.

## Plan

- **Default — "dry run":** fit by load-only test; decode at depth from a
  synthesized slot, built per run from one short real prefill per
  (model, KV type, flash-attn, swa_full) and expanded per depth; rows say
  prefill-at-depth was not measured.
- **`--real-context`:** today's behaviour — real prefill, which also measures
  prefill speed at depth.

## Open — must be settled before it is the default

- [x] **Speculative decoding (MTP / draft):** +11.4% — not faithful; the draft
      context is not in slot files. Real prefill for speculative rows, or a
      llama.cpp change to save `ctx_dft` (see above).
- [x] **iSWA models** (gemma): −1.1% with the window kept at its size.
- [x] **Concurrency rows:** +1.2% with the target's stream layout written.
- [x] The expander kept as `scripts/kvslot.py` (`info`, `expand`).
- [ ] Disk: a 131k f16 synthesized slot for the 27B is ~8.6 GB; delete as used.
- [ ] Format drift: GGSQ version is checked; an unknown version must fall back
      to real prefill, not guess.
