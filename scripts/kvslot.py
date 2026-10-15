#!/usr/bin/env python3
"""Read and expand llama.cpp slot-save files (GGSQ v4).

Prototype for docs/context-simulation-design.md: a short real slot, expanded
to depth N, restored into llama-server, decodes within ~1% of a real prefill
(except under speculative decoding — see the doc). Not yet wired into the
sweep.

    scripts/kvslot.py info  SLOT.bin
    scripts/kvslot.py expand SRC.bin DST.bin N [--streams S --stream K]


Layout (llama-context.cpp state_seq_save_file, llama-kv-cache.cpp state_write):
  u32 magic 'GGSQ', u32 version, u32 n_tok, i32 tokens[n_tok]
  then memory sections in order: one KV section per KV cache (two for iSWA:
  base then SWA), and for hybrid models an opaque recurrent section last.
  KV section: u32 n_stream; per stream: u32 cell_count; if > 0:
    meta  cell_count x (i32 pos, u32 n_seq, [12-byte ext], i32 seq[n_seq])
    data  u32 v_trans, u32 n_layer, u32 n_rot_k, u32 n_rot_v,
          n_layer x (i32 k_type, u64 k_row, k_row*cell_count bytes)
          then V: non-transposed  n_layer' x (i32 type, u64 row, row*cells)
                  transposed      n_layer' x (i32 type, u32 el, u32 n_embd,
                                              n_embd*el*cells bytes)
"""
import struct

MAGIC = 0x67677371      # 'ggsq'


class Reader:
    def __init__(self, b, off=0):
        self.b, self.o = b, off

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.o)[0]
        self.o += 8
        return v

    def take(self, n):
        if self.o + n > len(self.b):
            raise ValueError("past end")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v


def parse_kv(b, off, ext):
    """One KV section at `off`, or raise. Returns (section dict, end offset)."""
    r = Reader(b, off)
    n_stream = r.u32()
    if not 1 <= n_stream <= 64:
        raise ValueError(f"n_stream {n_stream}")
    streams = []
    for _ in range(n_stream):
        cells = r.u32()
        if cells > 1 << 24:
            raise ValueError(f"cell_count {cells}")
        st = {"cells": cells, "meta": [], "k": [], "v": [], "v_trans": 0}
        if cells:
            for _ in range(cells):
                pos, n_seq = r.i32(), r.u32()
                if n_seq > 64:
                    raise ValueError("n_seq")
                ex = r.take(ext)
                seqs = [r.i32() for _ in range(n_seq)]
                st["meta"].append((pos, ex, seqs))
            v_trans, n_layer = r.u32(), r.u32()
            r.u32(), r.u32()                       # n_rot_k, n_rot_v
            st["rot"] = b[r.o - 8:r.o]
            if v_trans not in (0, 1) or not 1 <= n_layer <= 512:
                raise ValueError(f"v_trans {v_trans} n_layer {n_layer}")
            st["v_trans"] = v_trans
            for _ in range(n_layer):
                t, row = r.i32(), r.u64()
                if not 0 <= t < 64 or not 0 < row < 1 << 24:
                    raise ValueError("k header")
                st["k"].append((t, row, r.take(row * cells)))
            # V layers: as many as follow and parse (layers without V skip)
            for _ in range(n_layer):
                save = r.o
                try:
                    t = r.i32()
                    if not 0 <= t < 64:
                        raise ValueError
                    if v_trans:
                        el, n_embd = r.u32(), r.u32()
                        if not 0 < el <= 8 or not 0 < n_embd < 1 << 20:
                            raise ValueError
                        st["v"].append((t, el, n_embd, r.take(el * n_embd * cells)))
                    else:
                        row = r.u64()
                        if not 0 < row < 1 << 24:
                            raise ValueError
                        st["v"].append((t, row, r.take(row * cells)))
                except (ValueError, struct.error):
                    r.o = save
                    break
        streams.append(st)
    return {"streams": streams, "ext": ext}, r.o


def parse(path):
    b = open(path, "rb").read()
    r = Reader(b)
    if r.u32() != MAGIC:
        raise ValueError("not a slot file")
    version, n_tok = r.u32(), r.u32()
    tokens = struct.unpack_from(f"<{n_tok}i", b, r.o)
    off = r.o + 4 * n_tok
    sections, ext_used = [], None
    while off < len(b):
        for ext in ([ext_used] if ext_used is not None else [0, 12]):
            try:
                sec, end = parse_kv(b, off, ext)
                cells = sec["streams"][0]["cells"]
                # a KV section's first cell is position 0 or a window's start;
                # its tokens must agree with the header's count
                if cells and cells > n_tok:
                    raise ValueError("more cells than tokens")
                sections.append(sec)
                off, ext_used = end, ext
                break
            except (ValueError, struct.error, IndexError):
                continue
        else:
            break                                  # opaque tail (recurrent)
    return {"version": version, "tokens": list(tokens), "sections": sections,
            "tail": b[off:], "size": len(b)}


def _info(path):
    p = parse(path)
    print(f"version {p['version']}, {len(p['tokens'])} tokens, "
          f"{len(p['sections'])} KV section(s), tail {len(p['tail'])} bytes, "
          f"file {p['size']} bytes")
    for i, s in enumerate(p["sections"]):
        st = s["streams"][0]
        pos = [m[0] for m in st["meta"]]
        print(f"  section {i}: ext {s['ext']}B, cells {st['cells']}, "
              f"pos {pos[0] if pos else '-'}..{pos[-1] if pos else '-'}, "
              f"K layers {len(st['k'])} row {st['k'][0][1] if st['k'] else '-'}B, "
              f"V layers {len(st['v'])}, v_trans {st['v_trans']}")


def server_tokens(header_tokens):
    """The real token ids inside llama-server's serialized block:
    [-1 (LLAMA_TOKEN_NULL), version, count, tokens..., n_media(=0)]."""
    if header_tokens and header_tokens[0] == -1:
        n = header_tokens[2]
        return list(header_tokens[3:3 + n])
    return list(header_tokens)                     # older plain token list


def expand(src, dst, n, n_stream=None, stream=0):
    """Write a slot file holding `n` cells, built by repeating `src`'s cells.

    The contents are nonsense by design: decode cost at depth depends on how
    many cells attention reads, not on what they say. Positions run 0..n-1 so
    the cache is a valid one; K/V rows, the M-RoPE extension and the token ids
    repeat the source cyclically. A recurrent tail (hybrid models; plus any
    appended context checkpoints) is copied unchanged — its size does not
    depend on depth.

    n_stream / stream: the TARGET server's KV layout. A server with several
    slots and split (non-unified) KV has one stream per slot and refuses a file
    with another count ("n_stream mismatch"); llama.cpp itself writes the
    saved sequence's stream filled and the others empty, so that is what this
    writes. Default: the source file's layout."""
    p = parse(src)
    toks = server_tokens(p["tokens"])
    if not p["sections"] or not toks:
        raise ValueError("nothing to expand")
    new_toks = [toks[i % len(toks)] for i in range(n)]
    with open(dst, "wb") as f:
        w = f.write
        packed = [-1, 1, n] + new_toks + [0]
        w(struct.pack("<III", MAGIC, p["version"], len(packed)))
        w(struct.pack(f"<{len(packed)}i", *packed))
        for sec in p["sections"]:
            ext = sec["ext"]
            src_st = next((x for x in sec["streams"] if x["cells"]),
                          sec["streams"][0])
            streams = (sec["streams"] if n_stream is None else
                       [src_st if i == stream else {"cells": 0}
                        for i in range(n_stream)])
            w(struct.pack("<I", len(streams)))
            for st in streams:
                src_n = st["cells"]
                if not src_n:
                    w(struct.pack("<I", 0))
                    continue
                # A full-history section grows to n cells at 0..n-1. A window
                # (iSWA's SWA cache holds only the last n_swa positions) keeps
                # its size and moves to the END of the new history.
                window = src_n < len(toks)
                cells = min(src_n, n) if window else n
                first = n - cells
                w(struct.pack("<I", cells))
                for i in range(first, n):
                    w(struct.pack("<iI", i, 1))
                    if ext:
                        w(struct.pack("<iii", i, i, new_toks[i]))
                    w(struct.pack("<i", 0))
                w(struct.pack("<II", st["v_trans"], len(st["k"])))
                w(st["rot"])
                reps, rem = divmod(cells, src_n)
                for t, row, data in st["k"]:
                    w(struct.pack("<iQ", t, row))
                    for _ in range(reps):
                        w(data)
                    w(data[:rem * row])
                for v in st["v"]:
                    if st["v_trans"]:
                        t, el, n_embd, data = v
                        w(struct.pack("<iII", t, el, n_embd))
                        stride = el * src_n
                        for j in range(n_embd):
                            col = data[j * stride:(j + 1) * stride]
                            for _ in range(reps):
                                w(col)
                            w(col[:rem * el])
                    else:
                        t, row, data = v
                        w(struct.pack("<iQ", t, row))
                        for _ in range(reps):
                            w(data)
                        w(data[:rem * row])
        w(p["tail"])
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info")
    a.add_argument("slot")
    b = sub.add_parser("expand")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("n", type=int)
    b.add_argument("--streams", type=int, default=None,
                   help="target server's KV stream count (split KV: slots)")
    b.add_argument("--stream", type=int, default=0,
                   help="which stream (slot) holds the expanded cells")
    args = ap.parse_args()
    if args.cmd == "info":
        _info(args.slot)
    else:
        expand(args.src, args.dst, args.n, args.streams, args.stream)
        _info(args.dst)
