#!/usr/bin/env python3
"""
llama-optimize - find good llama.cpp command-line parameters for a given GGUF model
on this machine, using a Taguchi orthogonal-array sweep over llama-bench.

Usage:
    llama-optimize.py MODEL.gguf                 # plan only: print the matrix + commands
    llama-optimize.py MODEL.gguf --run           # actually run the benchmark sweep
    llama-optimize.py MODEL.gguf --run --array L125   # bigger sweep

The tool auto-detects CPU cores, VRAM, and the model's layer count to choose
sensible factor levels, runs the sweep (one llama-bench invocation per Taguchi
run), then reports the fastest / longest-context / balanced configurations as
ready-to-paste llama-server command lines.

Nothing touches the GPU unless --run is given.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import struct
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT.parent
# The Taguchi/Morris/Sobol suite is vendored as the `robust` git submodule at
# ./robust (its internal layout is nested, so locate the python binding by
# search rather than a fixed path). Older checkouts used ./taguchi; fall back to
# it so the tool keeps working before a `git submodule sync`.
SUBMODULE_DIR = next((p for p in (PROJECT_ROOT / "robust", PROJECT_ROOT / "taguchi")
                      if p.exists()), PROJECT_ROOT / "robust")


def resolve_binary(name: str, explicit: Path | None, hint: Path | None) -> Path:
    """Locate a llama.cpp binary. Search order: explicit path, --llama-cpp hint
    (root / build/bin / bin), $LLAMA_CPP, $PATH, then the sibling-workspace
    default. Returns the first existing match, else a best-guess path (whose
    non-existence is reported later)."""
    if explicit is not None:
        return explicit
    roots = []
    if hint is not None:
        roots.append(hint)
    env = os.environ.get("LLAMA_CPP")
    if env:
        roots.append(Path(env))
    cands: list[Path] = []
    for r in roots:
        cands += [r / name, r / "build" / "bin" / name, r / "bin" / name]
    on_path = shutil.which(name)
    if on_path:
        cands.append(Path(on_path))
    default = WORKSPACE / "llama.cpp" / "build" / "bin" / name
    cands.append(default)
    for c in cands:
        if c.exists():
            return c
    return default  # doesn't exist; caller validates and errors clearly


_help_cache: dict[str, str] = {}


def binary_help(binary: Path) -> str:
    """Cached `--help` text for a llama.cpp binary; "" if it cannot be run."""
    key = str(binary)
    if key not in _help_cache:
        try:
            out = subprocess.run([key, "--help"], capture_output=True,
                                 text=True, timeout=60)
            _help_cache[key] = (out.stdout or "") + "\n" + (out.stderr or "")
        except (OSError, subprocess.TimeoutExpired):
            _help_cache[key] = ""
    return _help_cache[key]


def supports_flag(binary: Path, flag: str) -> bool:
    """Does this binary's --help advertise exactly `flag`?

    Matched with a boundary so `--fit` does not match `--fit-target`: llama-bench
    has the -target/-ctx variants but no `--fit` of its own, and a substring test
    would emit a flag it rejects. llama.cpp exits non-zero on an unknown
    argument, so a wrong answer here fails every run rather than degrading."""
    return re.search(re.escape(flag) + r"(?![\w-])", binary_help(binary)) is not None


def preflight(binary: Path, timeout: int = 60):
    """Confirm the binary actually runs (not just exists) — catches a wrong build
    or missing GPU libraries. Returns (ok, reason)."""
    try:
        out = subprocess.run([str(binary), "--help"], capture_output=True,
                             text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "not found / not executable"
    except subprocess.TimeoutExpired:
        return False, "hung running --help"
    except OSError as e:
        return False, f"failed to execute ({e})"
    if out.returncode != 0:
        tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
        return False, "exited nonzero — " + " ".join(tail)
    return True, ""


def find_taguchi_binding() -> Path:
    """Return the dir to add to sys.path so `import taguchi` works."""
    hits = sorted(
        SUBMODULE_DIR.glob("**/bindings/python/taguchi/__init__.py"),
        key=lambda p: len(p.parts),
    )
    if not hits:
        raise SystemExit(
            f"taguchi python binding not found under {SUBMODULE_DIR}.\n"
            f"Run:  git submodule update --init  &&  make -C {SUBMODULE_DIR}\n"
            "(builds libtaguchi.so + the morris binary; see the README's Setup section)."
        )
    return hits[0].parents[1]  # .../bindings/python


def find_robust_binary(name: str) -> Path:
    p = SUBMODULE_DIR / "build" / "bin" / name
    if p.exists():
        return p
    hits = list(SUBMODULE_DIR.glob(f"**/bin/{name}"))
    return hits[0] if hits else p


def prepare_taguchi_cli():
    """Point the python binding at the taguchi CLI we actually built.

    The binding locates the CLI by walking paths relative to its own package
    dir, which assumed the old layout (taguchi/build/taguchi, produced by a
    vendored sub-make). Upstream now builds taguchi as a normal peer into
    build/bin/, so those relative guesses all miss and the binding raises
    "Could not find taguchi CLI" — even though it is sitting right there.
    Rather than depend on the submodule's internal layout, hand it the path:
    TAGUCHI_CLI_PATH is the enhanced binding's highest-priority source, and
    PATH covers the plain binding, which only falls back to shutil.which."""
    cli = find_robust_binary("taguchi")
    if not cli.exists():
        return
    os.environ.setdefault("TAGUCHI_CLI_PATH", str(cli))
    bindir = str(cli.parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")

# Fixed parameters (see design notes): flash-attn is a precondition for KV-quant
# and a near-certain win on gfx906; mmap on is the sane default; batch fixed to
# avoid invalid batch<ubatch combinations.
# Stamped into every results CSV so a file found later can be traced to what
# produced it. A tuning tool's output outlives the run: when a fix changes what a
# past measurement MEANS (see CHANGELOG "Affects existing results"), the only way
# to answer "is this CSV affected?" is for the CSV to say what made it.
__version__ = "0.2.0"

FIXED_FA = 1
FIXED_MMAP = 1
FIXED_BATCH = 2048

# llama-bench measurement shape per config
BENCH_N_PROMPT = 512
BENCH_N_GEN = 128
BENCH_REPS = 3

# Workload profiles. Each sets the representative request shape (prompt + gen
# tokens) that the sweep measures at and scores by, a usable-context floor, and
# the driver. Objective = effective throughput for that request shape:
#   (P + G) / (P/pp_tps + G/tg_tps)   -- combines prefill and decode as the
# workload actually experiences them. "multi" needs the server driver (real
# concurrency), which llama-bench cannot do.
# `prefix_reuse` is part of the request SHAPE and so lives here with the sizes.
# Default 0.0 — assume nothing is shared unless the workload says otherwise. That
# is a statement, not a guess: an identical-request default silently inflates
# n-gram speculation (1.00 acceptance vs 0.31 at a realistic 90%), and the
# failure modes are asymmetric. Overstating speculation is invisible and ships a
# config that will not deliver; understating it is visible and recoverable.
# `agents` is the exception because the name is itself a claim about traffic —
# long tool-use prompts behind a fixed system preamble. It is an estimate to
# override, not a measurement (docs/constants-audit.md).
PROFILES = {
    "single": {"n_prompt": 512,  "n_gen": 256, "ctx_floor": 8192,  "driver": "bench",
               "prefix_reuse": 0.0},
    "agents": {"n_prompt": 8192, "n_gen": 256, "ctx_floor": 32768, "driver": "bench",
               "prefix_reuse": 0.9},
    "multi":  {"n_prompt": 1024, "n_gen": 256, "ctx_floor": 8192,  "driver": "server",
               "parallel": 4, "prefix_reuse": 0.0},
}

# Use-cases are high-level "runbooks": a friendly name that expands into a bundle
# of lower-level flags (driver + request profile + concurrency). Precedence is
# built-in defaults < use-case < explicit flags, so `--use-case agents --parallel 2`
# keeps the agents runbook but forces 2 streams. Each entry maps to a base profile
# (request shape / objective) plus the driver and concurrency that fit the workload.
USE_CASES = {
    # name          driver     profile    parallel   what it's for
    "app":        {"driver": "bench",  "profile": "single", "parallel": 1},
    #   general llama-based app / embedded llama.cpp — raw single-stream throughput
    "single":     {"driver": "server", "profile": "single", "parallel": 1},
    #   llama-server for one user/worker — real generation incl. MTP, one slot
    "agents":     {"driver": "server", "profile": "agents", "parallel": 4},
    #   several autonomous agents — long tool-use prompts, concurrent slots
    "multi-user": {"driver": "server", "profile": "multi",  "parallel": 8},
    #   many concurrent chat users — short prompts, high concurrency
}


# ---------------------------------------------------------------------------
# Hardware / model auto-detection
# ---------------------------------------------------------------------------
def detect_logical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def detect_physical_cores() -> int:
    """Count unique (physical id, core id) pairs from /proc/cpuinfo."""
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif line.strip() == "":
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    # Fallback: assume 2 threads/core
    return max(1, detect_logical_cores() // 2)


def detect_numa_nodes() -> int:
    """Number of NUMA nodes (1 when undetectable / not Linux)."""
    try:
        return max(1, len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))))
    except OSError:
        return 1


def parse_nvidia_vram(stdout: str) -> list[int]:
    """Per-GPU MiB from `nvidia-smi --query-gpu=memory.* --format=csv,noheader,
    nounits` — one line per device, already in MiB."""
    out = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(float(line)))
        except ValueError:
            continue
    return out


def parse_rocm_vram(payload: str, want: str) -> list[int]:
    """Per-GPU MiB from `rocm-smi --showmeminfo vram --json` (bytes), where
    `want` is "total" or "used". One entry per card key."""
    out = []
    for card in json.loads(payload).values():
        if not isinstance(card, dict):
            continue
        for k, v in card.items():
            kl = k.lower()
            # rocm-smi spells these "VRAM Total Memory (B)" and "VRAM Total
            # Used Memory (B)" — the used key contains "total" as well, so a
            # bare substring test for total matches both and the answer would
            # depend on key order.
            if "vram" not in kl:
                continue
            hit = ("used" in kl) if want == "used" else ("total" in kl and "used" not in kl)
            if hit:
                try:
                    out.append(int(v) // (1024 * 1024))
                except (TypeError, ValueError):
                    pass
                break
    return out


# `llama-server --list-devices` prints one line per device, same shape for
# every backend:
#     ROCm0: AMD Radeon 780M Graphics (30438 MiB, 43542 MiB free)
#     CUDA0: NVIDIA GeForce RTX 3090 (24117 MiB, 6672 MiB free)
_DEVICE_LINE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*?\d+)\s*:\s*(.+?)\s*"
    r"\(\s*(\d+)\s*MiB\s*,\s*(\d+)\s*MiB\s+free\s*\)\s*$")


def parse_list_devices(text: str) -> list[dict]:
    """Per-device capacity from `llama-server --list-devices`, in llama.cpp's
    own device order. Returns [{id, name, total_mib, free_mib}, ...].

    One parser covers every backend because the line shape is identical for
    ROCm, CUDA and Vulkan — which is also why this replaces two vendor-specific
    smi paths rather than adding a third.

    CPU devices are dropped: some builds list them, their "VRAM" is system RAM,
    and summing that into a GPU budget would overstate the machine by more than
    the bug this function exists to fix.
    """
    out = []
    for line in text.splitlines():
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        dev_id, name, total, free = m.groups()
        if dev_id.upper().startswith("CPU"):
            continue
        out.append({"id": dev_id, "name": name,
                    "total_mib": int(total), "free_mib": int(free)})
    return out


def list_devices(binary: Path | None) -> list[dict]:
    """Ask llama.cpp itself what devices it has. [] if the binary is missing,
    too old for --list-devices, or prints nothing we recognise.

    Output is read from stdout and stderr together: the device table goes to
    stdout, but backend init chatter lands on stderr and some builds interleave
    them. The line pattern is specific enough that combining is safe."""
    if binary is None or not binary.exists():
        return []
    try:
        out = subprocess.run([str(binary), "--list-devices"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    # Deliberately not gated on returncode: a build that does not know the flag
    # prints no device lines, which the parser reports as [] anyway.
    return parse_list_devices((out.stdout or "") + "\n" + (out.stderr or ""))


def detect_vram_mib(*binaries: Path | None) -> tuple[int, str] | None:
    """Best-effort total usable VRAM in MiB across ALL devices, with the source
    it came from. None if nothing could be read.

    Asks llama.cpp first (`llama-server --list-devices`) and only falls back to
    rocm-smi/nvidia-smi. llama.cpp is the right authority for the same reason it
    is for device *order* (docs/multi-gpu-design.md): the number that matters is
    what the inference process can actually allocate, which is not always what
    the vendor tool calls VRAM.

    Issue #7 is the case in point. On an APU (Radeon 780M) the model lives in
    GTT, host memory the GPU maps; `rocm-smi --showmeminfo vram` reports the
    2 GiB carve-out and nothing else, while llama.cpp reports the ~30 GiB it can
    really use. Reading 2048 MiB there made the OOM pruner skip nearly every
    configuration in the sweep without running one of them.

    Summed, not first-card: both smi tools print one line/key per device, and
    reading only the first silently understates a multi-GPU box. That number
    feeds the OOM pruner, which sums usage over every GPU line llama-fit-params
    reports — comparing an all-device footprint against one card's capacity
    prunes configurations that would have fit (issue #5).

    Several binaries may be offered and the first that answers wins: both
    llama-server and llama-bench implement --list-devices identically, and a
    bench-driver user has no reason to have built the server."""
    for binary in binaries:
        devs = list_devices(binary)
        total = sum(d["total_mib"] for d in devs)
        if total > 0:   # a device list that adds to zero is not an answer
            return total, "llama.cpp: " + ", ".join(d["id"] for d in devs)
    # AMD / ROCm
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            per_gpu = parse_rocm_vram(out.stdout, "total")
            if per_gpu:
                return sum(per_gpu), "rocm-smi"
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    # NVIDIA / CUDA
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            per_gpu = parse_nvidia_vram(out.stdout)
            if per_gpu:
                return sum(per_gpu), "nvidia-smi"
    except (OSError, ValueError):
        pass
    return None


def vram_used_mib() -> int | None:
    """Currently-used VRAM in MiB across all GPUs (AMD then NVIDIA); None if
    unavailable. Summed for the same reason as detect_vram_mib: a peak-usage
    sample on a split model is only meaningful against every device it lives
    on."""
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            per_gpu = parse_rocm_vram(out.stdout, "used")
            if per_gpu:
                return sum(per_gpu)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            per_gpu = parse_nvidia_vram(out.stdout)
            if per_gpu:
                return sum(per_gpu)
    except (OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# GPU visibility. A llama.cpp built without a GPU backend does not fail — it
# runs everything on the CPU and reports perfectly plausible numbers. For a
# TUNING tool that is the worst shape of wrong: every -ngl level measures the
# same CPU run, the sweep still crowns a winner, and the recommended command
# claims layers on a GPU that was never touched. Observed for real: the same
# 270M model measured 115 t/s tg on a silently CPU-only build and 444 t/s once
# the HIP backend was actually compiled in — a 3.9x error with no error message.
#
# CPU-only sweeps are legitimate and worth doing (threads, ubatch, numa, affinity
# all matter there). What must never pass silently is a CPU-only *build* on a
# machine that HAS a GPU, which is a mistake rather than a choice.
# ---------------------------------------------------------------------------
# Factors that can only do something when llama.cpp can see a GPU. On a CPU-only
# build every level of these is the same run.
GPU_ONLY_FACTORS = ("ngl", "ncmoe", "ncffn", "nkvo", "ot", "ffn_place")


def gpu_visibility(vram, vram_src: str, can_ask: bool) -> str:
    """Which of four worlds we are in, read off what detect_vram_mib returned.

    That function already asks llama.cpp first and falls back to the vendor tool,
    so its *source* string carries the whole diagnosis — no second probe needed:

    - "gpu"      llama.cpp itself listed a device; nothing to say
    - "blind"    a vendor tool sees a GPU that llama.cpp does not. The build has
                 no GPU backend on a machine that has one
    - "cpu-only" nothing reports a GPU: a real CPU box, a legitimate sweep
    - "unknown"  the binary predates --list-devices, so llama.cpp's silence is
                 not evidence and the two answers cannot be compared
    """
    if vram and vram_src.startswith("llama.cpp"):
        return "gpu"
    if not can_ask:
        return "unknown"
    return "blind" if vram else "cpu-only"


# Below this fraction of VRAM free, a sweep is likely to abort rather than
# measure. Deliberately generous: the point is to catch "the card is full",
# not to second-guess a tight but workable fit. A model needing more than half
# a free card is a normal thing to tune.
LOW_VRAM_FREE_FRAC = 0.25


def vram_headroom(devs: list[dict]) -> tuple[int, int]:
    """(free, total) MiB summed over devices; (0, 0) if unknown."""
    if not devs:
        return 0, 0
    return (sum(d.get("free_mib", 0) for d in devs),
            sum(d.get("total_mib", 0) for d in devs))


def headroom_warning(devs: list[dict]) -> str | None:
    """Why this box is too full to sweep on, or None.

    The OOM pruner reasons about TOTAL VRAM (`detect_vram_mib`), which is the
    right basis for "can this model ever fit on this card" and the wrong one for
    "can it fit right now". On a shared box the two diverge completely, and the
    failure is expensive and confusing: the pruner passes a config, the run
    aborts inside the CUDA allocator, and the sweep spends its whole budget
    discovering that another process owns the card. Even `-ngl 0` fails, because
    llama.cpp still op-offloads matmuls to the GPU backend, so "just run on the
    CPU" is not the escape it appears to be — the device has to be hidden.

    Free VRAM is not used to PRUNE, only to warn. It is a reading at one instant
    from a number the vendor tool can misreport (issue #7's APU counts GTT), and
    whatever holds the card now may release it before the sweep reaches the rows
    that need it. Refusing to run on it would trade a wasted sweep for a sweep
    that never starts."""
    free, total = vram_headroom(devs)
    if total <= 0 or free >= LOW_VRAM_FREE_FRAC * total:
        return None
    per_dev = ", ".join(f"{d['id']} {d.get('free_mib', 0)}/{d.get('total_mib', 0)} MiB"
                        for d in devs)
    return (f"only {free} of {total} MiB VRAM is free ({per_dev}) — "
            f"another process is using this GPU")


def warn_vram_headroom(devs: list[dict]) -> None:
    """Say so before the sweep, not after it has burned its budget."""
    why = headroom_warning(devs)
    if not why:
        return
    print()
    print("!" * 70)
    print(f"!! {why}.")
    print("!!")
    print("!! Runs are likely to ABORT rather than measure. The OOM pruner")
    print("!! compares against TOTAL VRAM, so it will not save you here, and")
    print("!! --no-oom-prune makes it worse rather than better.")
    print("!!")
    print("!! -ngl 0 is not a workaround: llama.cpp still offloads matmul to the")
    print("!! GPU backend, so CPU-only rows allocate VRAM too and abort with them.")
    print("!! To tune on the CPU while the card is busy, HIDE the device:")
    print("!!     HIP_VISIBLE_DEVICES=  (ROCm)   CUDA_VISIBLE_DEVICES=  (NVIDIA)")
    print("!!")
    print("!! Otherwise: wait for the card, or tune on the machine you will serve on.")
    print("!" * 70)
    print()


def warn_gpu_visibility(verdict: str, vram_src: str, factors) -> None:
    """Say so, loudly, when the build cannot see the GPU it is being tuned for."""
    inert = [n for n in GPU_ONLY_FACTORS if n in factors]
    if verdict == "blind":
        print()
        print("!" * 70)
        print("!! llama.cpp reports NO GPU devices, but " + (vram_src or "the vendor tool")
              + " sees one.")
        print("!!")
        print("!! Every run will execute on the CPU, and the numbers will look")
        print("!! entirely plausible while being wrong for your hardware — the same")
        print("!! model measured 115 t/s on a CPU-only build and 444 t/s once the GPU")
        print("!! backend was really there. 3.9x, with no error at any point.")
        if inert:
            print("!!")
            print("!! " + ", ".join(inert) + " cannot do anything here: every level is the same")
            print("!! CPU run. The sweep would spend rows telling identical configurations")
            print("!! apart, then recommend GPU layers that never load.")
        print("!!")
        print("!! Two causes, and this check cannot tell them apart:")
        print("!!  1. The build has no GPU backend. A STALE BUILD DIRECTORY DOES THIS")
        print("!!     SILENTLY — reconfiguring over an old cache can leave GGML_HIP=OFF")
        print("!!     even when -DGGML_HIP=ON was passed. Check build/CMakeCache.txt,")
        print("!!     or configure a clean build directory.")
        print("!!  2. The GPU is hidden from it — check HIP_VISIBLE_DEVICES /")
        print("!!     CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES in this shell.")
        print("!!")
        print("!! Either way: fix it, then confirm `llama-bench --list-devices` lists")
        print("!! your card before sweeping. If you meant to tune for CPU, this is")
        print("!! still worth doing — but drop the GPU factors so the rows aren't wasted.")
        print("!" * 70)
        print()
    elif verdict == "cpu-only" and inert:
        print(f"CPU-only   : no GPU detected — {', '.join(inert)} cannot vary here "
              "(every level is the same run);")
        print("             consider --factor to drop them and spend the rows on "
              "threads/ubatch/numa instead")


_build_cache: dict = {}


def llama_build(*binaries) -> str:
    """llama.cpp's own build identity, e.g. "build 10636 (4d19b2876)", or "".

    Stamped alongside our version because a result depends on BOTH: half the
    findings in this project's changelog are about llama.cpp behaviour, and
    `--diff` exists precisely to compare sweeps across llama.cpp upgrades.

    Only llama-server answers `--version` (llama-bench rejects it and prints its
    build in the run output instead), so several binaries may be offered and the
    first that answers wins — the same pattern detect_vram_mib uses."""
    for binary in binaries:
        if binary is None or not Path(binary).exists():
            continue
        key = str(binary)
        if key in _build_cache:
            if _build_cache[key]:
                return _build_cache[key]
            continue
        try:
            out = subprocess.run([str(binary), "--version"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            _build_cache[key] = ""
            continue
        got = parse_llama_version((out.stdout or "") + "\n" + (out.stderr or ""))
        _build_cache[key] = got
        if got:
            return got
    return ""


# llama.cpp prints: "version: 0.3.0-dev (build 10636, commit 4d19b2876)"
_VERSION_LINE = re.compile(
    r"\bbuild\s+(\d+)\s*,\s*commit\s+([0-9a-f]{7,40})\b")


def parse_llama_version(text: str) -> str:
    """"build 10636 (4d19b2876)" from a --version banner, or "".

    Matches on the build/commit pair rather than the leading "version:" token so
    a binary that reorders or reworks that line keeps working; a build number
    with no commit is not enough to identify what ran, so both are required."""
    m = _VERSION_LINE.search(text or "")
    return f"build {m.group(1)} ({m.group(2)})" if m else ""


# ---------------------------------------------------------------------------
# Model loading mode. `-mmp`/`--mmap`/`--no-mmap`/`--mlock`/`-dio` are all
# DEPRECATED in favour of `--load-mode`, and llama.cpp does remove deprecated
# arguments rather than carrying them forever — `--draft`, `--draft-min` and
# `--spec-ngram-size-n` are already gone. Emitting `-mmp` on every bench command
# is therefore a latent whole-sweep failure: when it goes, argument parsing
# fails and no row runs. common/arg.cpp also warns that the old and new spellings
# must not be combined ("only the last flag takes effect"), so a user who passes
# `--load-mode` today silently conflicts with the flag we insert for them.
#
# The old spellings stay as the fallback: a build predating `--load-mode` is a
# perfectly good build, and probing is what lets both work.
# ---------------------------------------------------------------------------
def load_mode_args(binary, driver: str,
                   mmap_on: bool = bool(FIXED_MMAP)) -> list[str]:
    """The flags that pin model loading, preferring `--load-mode`.

    `mmap` and `none` are the exact translations of the old pair: `-mmp 1`
    memory-maps, `--no-mmap`/`-mmp 0` asks for no special loading. `auto` is
    llama.cpp's own default and is deliberately NOT used — the point of pinning
    this is that every row loads the model the same way, and `auto` is free to
    decide differently per device.

    The legacy spellings are not symmetric and never were: llama-bench takes
    `-mmp 0|1`, while llama-server has only `--no-mmap` and mmaps by default, so
    "mmap on" is the absence of a flag there. `--load-mode` is what makes the two
    drivers finally say the same thing."""
    if supports_flag(binary, "--load-mode"):
        return ["--load-mode", "mmap" if mmap_on else "none"]
    if driver == "bench":
        return ["-mmp", "1" if mmap_on else "0"]
    return [] if mmap_on else ["--no-mmap"]


def gpu_temp_c() -> float | None:
    """Best-effort GPU temperature in °C (AMD rocm-smi then NVIDIA nvidia-smi);
    None if no sensor is readable. Returns the hottest sensor reported."""
    try:
        out = subprocess.run(["rocm-smi", "--showtemp", "--json"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            data = json.loads(out.stdout)
            edge, any_t = [], []
            for card in data.values():
                for k, v in card.items():
                    if "temp" not in k.lower():
                        continue
                    try:
                        t = float(v)
                    except (TypeError, ValueError):
                        continue
                    any_t.append(t)
                    if "edge" in k.lower() or "junction" in k.lower():
                        edge.append(t)
            temps = edge or any_t
            if temps:
                return max(temps)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return max(float(x) for x in out.stdout.strip().splitlines())
    except (OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Predictive OOM pruning: use llama-fit-params --fit-print to estimate VRAM
# before running a config — skip configs that can't fit, saving a model load
# + timeout per doomed row (ROADMAP item 2).
# ---------------------------------------------------------------------------
# Footprint flags recent enough that a llama-fit-params build in the wild may
# predate them. Keyed on the FLAG rather than the factor, because a factor's
# level can pick its own flag (`ffn_place` emits -ot for some levels and -ncffn
# for others), so only the emitted flags say what the estimator is really being
# asked for. Adding a placement flag is one entry here and nothing else.
#
# The estimator and the driver are separate binaries and are gated separately,
# so the mismatch is reachable in practice: --llama-cpp can point at a fresh
# build while an older llama-fit-params is found on $PATH or left from a
# previous checkout. Observed in the other direction too, inside one build tree
# — llama-fit-params carrying -ncffn while llama-bench did not.
FIT_RECENT_FLAGS = {"-ncmoe", "-ncffn"}


def fit_blind_flags(cfg: "Config", f: dict, driver: str) -> list[str]:
    """Flags this row's estimate needs that llama-fit-params cannot parse.

    Dropping such a flag and estimating anyway is the tempting move and the
    wrong one. These flags all exist to MOVE WEIGHTS OFF THE GPU, so an
    estimator blind to one reports the un-offloaded footprint — the largest
    configuration in the row. On the machine that needed the offload in the
    first place that overshoots VRAM, every level of the factor is predicted
    OOM, and the whole factor is silently pruned out of the sweep. A wrong
    estimate deletes rows; a missing estimate merely runs them (P3)."""
    emitted = set(_fit_params_flags(cfg, f, driver)) & FIT_RECENT_FLAGS
    return sorted(fl for fl in emitted if not supports_flag(cfg.fit_params, fl))


def _fit_params_flags(cfg: "Config", f: dict, driver: str) -> list[str]:
    """Build llama-fit-params args from the VRAM-relevant factors in a run row.
    Only the factors that affect memory footprint are forwarded; the rest of the
    config (threads, poll, batch, spec knobs) is irrelevant to the estimate.

    Every footprint factor present is emitted unconditionally, including ones
    the estimator may be too old to accept. Whether the binary can be asked at
    all is `fit_blind_flags`' business, decided once in `predict_fits`;
    keeping it out of here leaves this a pure "what would we pass" function, so
    the cache key it feeds still separates rows that differ only in a gated
    factor."""
    flags = []
    # GPU layers
    if "ngl" in f:
        flags += ["-ngl", f["ngl"]]
    # Context size (n_depth for bench, but fit-params uses -c for both)
    ctx = f.get("n_depth", "0")
    flags += ["-c", ctx]
    # KV cache type
    if "kv_type" in f:
        flags += ["-ctk", f["kv_type"], "-ctv", f["kv_type"]]
    # KV offload. `nkvo=1` is the level that puts the KV cache in SYSTEM RAM --
    # that is what the drivers emit for it (server: a bare `-nkvo`; bench:
    # `-nkvo 1`), and `--no-kv-offload` is the same flag by another name
    # (llama.cpp: `-kvo, --kv-offload, -nkvo, --no-kv-offload`). This was
    # inverted, so the estimator priced the OPPOSITE placement from the one the
    # row would run. Measured on gemma-4-31B at -ngl 60 -c 65536 -ctk f16:
    # 32666 MiB with the KV on the GPU against 26513 with `--no-kv-offload`, a
    # 6 GB error in whichever direction the row happened to sit — under-pricing
    # nkvo=0 rows into an OOM at launch, and over-pricing nkvo=1 rows into
    # SKIP_PRED. Found while asking why `full_offload_fits` said a model fitted
    # that llama-fit-params says needs 32666 of a 32240 budget.
    if f.get("nkvo", "0") == "1":
        flags.append("--no-kv-offload")
    # Tensor placement overrides (ot factor — only when present and not "none")
    if "ot" in f:
        pat = OT_PATTERNS.get(f["ot"], "")
        if pat:
            flags += ["-ot", pat]
    # MoE expert offload
    if "ncmoe" in f:
        flags += ["-ncmoe", f["ncmoe"]]
    # Dense FFN offload (llama.cpp #26622, tag b10645)
    if "ncffn" in f:
        flags += ["-ncffn", f["ncffn"]]
    # A second resident model changes the footprint more than any factor here,
    # and llama-fit-params has no way to be told about it ("invalid argument:
    # -md"). Emitted anyway so fit_blind_flags sees it and turns pruning OFF for
    # these rows: an estimate that silently ignores a whole model is the
    # confidently-wrong kind (DM3, revised — see draft-model-design.md).
    # Dense FFN placement: the level carries its own flag (-ot or -ncffn), and
    # both spellings are ones fit-params accepts, so this needs no special case
    # beyond asking the level what it emits.
    if "ffn_place" in f:
        flags += ffn_place_args(f["ffn_place"])
    return flags


# Track whether we've already warned about a disabled prune (once per sweep).
_oom_prune_warned = False
# Same, for the narrower "estimator is too old for this factor" case: reported
# once per distinct factor set so a partially-pruned sweep is legible rather
# than silent, without one line per row.
_fit_blind_warned: set = set()

# Cache for predict_fits: memoizing collapses the ~125 subprocess calls of an
# L125 sweep to the handful of distinct footprints it actually contains.
#
# The key is the *estimator's own argv*, not a hand-picked subset of factors.
# Anything that changes the estimate necessarily changes the flags, so the key
# cannot fall out of step with _fit_params_flags: adding a VRAM-relevant factor
# there extends the key for free. Listing factors separately was the bug this
# replaces — ncmoe and ot were forwarded as flags but missing from the key, so
# configs differing only in expert offload collided and inherited each other's
# verdict. A pruned row never runs, so a wrong verdict silently deletes a valid
# config from the sweep — on MoE models, where -ncmoe is the biggest VRAM lever
# we have, exactly the configs the feature exists to sort out.
_fit_cache: dict[tuple, bool | None] = {}


def _file_mib(path) -> int:
    """Size of a GGUF on disk, in MiB. 0 if it cannot be read."""
    try:
        return int(Path(path).stat().st_size / (1024 * 1024))
    except OSError:
        return 0


def resident_extra_mib(cfg: "Config", f: dict) -> int:
    """VRAM held by artifacts `llama-fit-params` cannot be told about.

    It rejects `-md` and `--mmproj` outright, so a setup that loads a draft
    model or a projector is priced for the text model alone. Rather than stand
    pruning down entirely, add what we can measure ourselves: weights on disk
    are a hard LOWER bound on weights in VRAM.

    A lower bound is the only safe direction. The estimate is compared against a
    ceiling, so understating it prunes FEWER rows — it admits some configs that
    will not fit, costing time, and can never delete one that would have. An
    overestimate would do the opposite, which is the failure this project keeps
    finding (issues #5, #7). Compute buffers and the draft model's own KV cache
    are therefore deliberately not modelled.

    Residency is per-row, not fixed: a projector with `--mmproj-offload 0` is on
    the CPU and costs nothing, and a draft model at `-ngld 0` likewise."""
    extra = 0
    mmproj = getattr(cfg, "mmproj", None)
    if mmproj and str(f.get("mmproj_offload", "1")) != "0":
        extra += _file_mib(mmproj)          # projector offloaded (llama.cpp default)
    draft = getattr(cfg, "draft_model", None)
    if draft:
        ngld = f.get("spec_draft_ngl")
        d_layers = draft_layer_count(draft)
        try:
            ngld_n = None if ngld is None else int(ngld)
        except (TypeError, ValueError):
            ngld_n = None
        if ngld is None:
            frac = 1.0          # not swept: llama.cpp puts the whole drafter on GPU
        elif ngld_n == 0:
            frac = 0.0          # explicitly CPU-resident, whatever its geometry
        elif d_layers and ngld_n is not None:
            frac = min(1.0, ngld_n / d_layers)
        else:
            # placement is swept but the drafter's geometry is unreadable. Do
            # not guess upward: over-counting prunes rows that would have fit,
            # which is the expensive direction. Fall back to pricing the text
            # model alone for this row.
            frac = 0.0
        extra += int(_file_mib(draft) * frac)
    return extra


def _fit_cache_key(cfg: "Config", f: dict, driver: str) -> tuple:
    """Identity of a fit estimate: the model, the exact args we'd pass, and the
    footprint of anything fit-params cannot be told about — which varies per row
    with the projector's and drafter's placement, so it has to reach the key."""
    return (str(cfg.model), tuple(_fit_params_flags(cfg, f, driver)),
            resident_extra_mib(cfg, f))


def parse_fit_print(stdout: str) -> int:
    """Total GPU MiB from `llama-fit-params --fit-print on`.

    Each line is "DEVICE model context compute" in MiB, e.g.
    "ROCm0 271 27 513". The Host line is system RAM, not VRAM, and is excluded —
    counting it would inflate the footprint and prune configs that fit."""
    gpu_used = 0
    for line in stdout.strip().splitlines():
        if not line.strip() or line.startswith("Host"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                gpu_used += sum(int(p) for p in parts[1:4])
            except ValueError:
                continue                   # not a device line; ignore
    return gpu_used


def predict_fits(cfg: "Config", f: dict, driver: str) -> bool | None:
    """Estimate whether factor config `f` fits in GPU VRAM using
    llama-fit-params --fit-print. Returns True (fits), False (would OOM), or
    None (could not determine — skip pruning for safety). Only considers GPU
    devices; when there are none, always returns True (CPU-only fits).

    Both sides of the comparison now come from llama.cpp: the footprint from
    llama-fit-params' per-device lines, the ceiling from --list-devices. They
    used to disagree about what "GPU memory" means — issue #7's APU counted GTT
    on one side and a 2 GiB VRAM carve-out on the other, and every row lost."""
    global _oom_prune_warned, _fit_cache
    total_vram = cfg.hw.get("vram")
    if total_vram is None or not cfg.fit_params.exists():
        return None
    # The estimator cannot see one of this row's placement factors, so it cannot
    # answer the question being asked. Say so and run the row (P3) rather than
    # act on a footprint that ignores the offload the row is about.
    blind = fit_blind_flags(cfg, f, driver)
    if blind:
        names = ", ".join(blind)
        if names not in _fit_blind_warned:
            _fit_blind_warned.add(names)
            print(f"OOM prune: llama-fit-params ({cfg.fit_params.name}) predates "
                  f"{names} — pruning is OFF for rows needing it; they will all "
                  "run. Update llama.cpp to restore it.", file=sys.stderr)
        return None
    key = _fit_cache_key(cfg, f, driver)
    if key in _fit_cache:
        return _fit_cache[key]
    try:
        args = [str(cfg.fit_params), "-m", str(cfg.model),
                "--fit-print", "on"]
        args += _fit_params_flags(cfg, f, driver)
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            # Indeterminate is a real answer and worth remembering: a binary
            # that rejects these flags rejects them every time, and without
            # caching we would respawn it once per row to learn that again.
            _fit_cache[key] = None
            return None
        gpu_used = parse_fit_print(proc.stdout)
        # fit-params priced the text model only; add the artifacts it cannot see
        gpu_used += resident_extra_mib(cfg, f)
        limit = total_vram - cfg.fit_headroom_mib
        result = gpu_used < limit
        _fit_cache[key] = result
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        if not _oom_prune_warned:
            print("OOM prune: unable to estimate — running all configs anyway "
                  "(conservative; --no-oom-prune silences this)", file=sys.stderr)
            _oom_prune_warned = True
        return None


# Thermal "wait and watch": between runs, block until the GPU falls back to
# within THERMAL_BAND_C of the idle baseline so each config is measured from a
# comparable thermal state (the MI50 throttles ~1.8× cool-vs-hot, a swing bigger
# than most factor effects — see docs/DESIGN.md). Capped so it can never hang.
THERMAL_BAND_C = 5.0
THERMAL_CAP_S = 120.0


def wait_until_cool(baseline_c: float | None, band: float = THERMAL_BAND_C,
                    cap_s: float = THERMAL_CAP_S, poll_s: float = 3.0) -> None:
    """Watch GPU temperature; return once it is within `band` °C of the idle
    `baseline_c`, or it plateaus (cooling stalls), or `cap_s` elapses. No-op
    when there's no baseline or no readable sensor. A *rising* temperature is
    not a plateau: right after a run the sensor often keeps climbing for a few
    seconds (heat soak), and bailing then would exit at the hottest moment."""
    if baseline_c is None:
        return
    target, t0, prev = baseline_c + band, time.time(), None
    tty = sys.stdout.isatty()
    while time.time() - t0 < cap_s:
        t = gpu_temp_c()
        if t is None:
            break
        settled = t <= target or (prev is not None and 0 <= prev - t < 0.5)
        if tty:
            print(f"\r  thermal: {t:>3.0f}°C  (settle ≤{target:.0f}°C)"
                  f"{'  ok' if settled else '  cooling…'}   ", end="", flush=True)
        if settled:
            break
        prev = t
        time.sleep(poll_s)
    if tty:
        print()


class VRAMSampler:
    """Polls used VRAM in a background thread, tracking the peak during a run.
    (rocm-smi is slow, so polling is coarse; captures the config's footprint.)"""

    def __init__(self, interval: float = 1.0):
        self.peak = 0
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def poll():
            while not self._stop.wait(self.interval):
                v = vram_used_mib()
                if v:
                    self.peak = max(self.peak, v)
        # one immediate sample so short runs still get a reading
        v = vram_used_mib()
        if v:
            self.peak = v
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        return False


# --- Minimal GGUF metadata reader (header only, no tensor data) -------------
_GGUF_MAGIC = b"GGUF"
# value type ids
_GT_UINT8, _GT_INT8, _GT_UINT16, _GT_INT16, _GT_UINT32, _GT_INT32 = 0, 1, 2, 3, 4, 5
_GT_FLOAT32, _GT_BOOL, _GT_STRING, _GT_ARRAY = 6, 7, 8, 9
_GT_UINT64, _GT_INT64, _GT_FLOAT64 = 10, 11, 12
_GT_FMT = {
    _GT_UINT8: "<B", _GT_INT8: "<b", _GT_UINT16: "<H", _GT_INT16: "<h",
    _GT_UINT32: "<I", _GT_INT32: "<i", _GT_FLOAT32: "<f", _GT_BOOL: "<?",
    _GT_UINT64: "<Q", _GT_INT64: "<q", _GT_FLOAT64: "<d",
}


class _GGUFReader:
    def __init__(self, f):
        self.f = f

    def _read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected EOF reading GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self._read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self._read(8))[0]

    def string(self):
        n = self.u64()
        return self._read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _GT_FMT:
            fmt = _GT_FMT[vtype]
            return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
        if vtype == _GT_STRING:
            return self.string()
        if vtype == _GT_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path: Path) -> dict:
    """Parse GGUF metadata key/values only. Returns {} on any failure."""
    try:
        with open(path, "rb") as f:
            r = _GGUFReader(f)
            if r._read(4) != _GGUF_MAGIC:
                return {}
            version = r.u32()
            if version < 2:
                return {}
            _tensor_count = r.u64()
            kv_count = r.u64()
            meta = {}
            for _ in range(kv_count):
                key = r.string()
                vtype = r.u32()
                meta[key] = r.value(vtype)
            return meta
    except (OSError, EOFError, ValueError, struct.error):
        return {}


def _meta_int(meta: dict, suffix: str) -> int | None:
    for k, v in meta.items():
        if k.endswith(suffix):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def model_block_count(meta: dict) -> int | None:
    return _meta_int(meta, ".block_count")


def model_expert_count(meta: dict) -> int:
    """Number of MoE experts; 0 (or missing) means a dense model."""
    return _meta_int(meta, ".expert_count") or 0


def model_context_length(meta: dict) -> int | None:
    """Native max context the model was trained for."""
    return _meta_int(meta, ".context_length")


def model_hw(meta: dict) -> dict:
    """The model-derived half of `cfg.hw`, straight from GGUF metadata.

    Extracted so the path from a metadata key to a swept factor can be tested at
    all. Every entry here gates a factor — `n_experts` picks `ncmoe` vs
    `ffn_place`, `n_nextn` the speculative family, `n_swa` the `swa_full` pair —
    so a key that silently reads 0 deletes a whole column from the design, and
    the deletion looks exactly like a model that does not have the feature.
    That was untestable while it lived inline in `main` (issue #15)."""
    return {"n_layers": model_block_count(meta),
            "n_experts": model_expert_count(meta),
            "n_ctx_train": model_context_length(meta),
            "n_nextn": model_nextn_layers(meta),
            "n_swa": model_swa_window(meta),
            "ssm_state": model_ssm_state(meta)}


def draft_head_kind(path) -> str | None:
    """The kind of speculative head a draft GGUF is, from its own architecture.

    `dflash` covers both DFlash2 and DSpark — llama.cpp tells them apart by a
    `markov_w1.weight` tensor, which is a tensor-level question this metadata
    reader cannot answer and does not need to: the level emits `-md` alone and
    lets `common_speculative_types_from_gguf` make that call. Any other
    architecture is an ordinary draft model.

    Measured on `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`: `general.architecture =
    dflash`, `dflash.block_count = 5`, `dflash.target_layers =
    [6, 20, 34, 48, 62]`."""
    if not path:
        return None
    try:
        arch = read_gguf_metadata(Path(path)).get("general.architecture")
    except (OSError, ValueError):
        return None
    return str(arch) if arch else None


def spec_type_levels(cfg: "Config") -> list[str]:
    """The speculative heads this model can actually be measured with.

    `none` always; `draft-mtp` when the target ships a NextN head; and the
    supplied draft model's own kind when there is one. Two or fewer levels means
    there is no choice to sweep and the older `mtp` on/off factor covers it —
    this exists for the case the field report raised (issue #19), where a model
    has an embedded MTP head AND a DFlash2 head is available, and the question
    "which one is faster" could not be put to the sweep at all."""
    levels = ["none"]
    if cfg.driver == "server" and cfg.hw.get("n_nextn", 0) > 0 and cfg.emit_mtp:
        levels.append("draft-mtp")
    kind = draft_head_kind(getattr(cfg, "draft_model", None))
    if kind and kind not in levels:
        levels.append(kind)
    return levels


def spec_type_args(cfg: "Config", level: str) -> list[str]:
    """Server flags for one `spec_type` level.

    `none` emits nothing. `draft-mtp` names the type, since the head is inside
    the target model and there is nothing to point at. Any other level is a
    supplied draft model: emit `-md` and let llama.cpp read the type off it,
    rather than pre-empting an inference it does better (issue #19)."""
    if level == "none":
        return []
    if level == "draft-mtp":
        return ["--spec-type", "draft-mtp"]
    return draft_model_args(cfg)


def draft_self_describes(path) -> bool:
    """Whether llama.cpp can work out this draft's speculative type on its own.

    `common_speculative_types_from_gguf` recognises two things: architecture
    `dflash` (DFlash2, or DSpark when the Markov head is there), and an ordinary
    architecture carrying `blk.N.nextn.eh_proj.weight` — an MTP head. The second
    is a tensor and this reader parses key/values only, but the standalone MTP
    sidecars carry `{arch}.nextn_predict_layers` in their KV as well, which is the
    same signal one layer up.

    Anything else is an ordinary model, and llama.cpp infers NOTHING from it —
    which is the whole reason this function exists."""
    if not path:
        return False
    meta = read_gguf_metadata(Path(path))
    if not meta:
        return False
    return (str(meta.get("general.architecture", "")) == "dflash"
            or model_nextn_layers(meta) > 0)


def draft_model_args(cfg: "Config") -> list[str]:
    """`-md`, plus the type when llama.cpp cannot infer one.

    A plain draft model — the classic speculative setup, a small sibling of the
    target — tells llama.cpp nothing about itself. Inference returns an empty
    list, the type stays at its default of `none` (`common/arg.cpp`
    `spec_types_is_default`), and the draft is **loaded, charged to VRAM, and
    never used**. Nothing said so: the row is not flagged `spec_off` either,
    because on a target without its own MTP head nothing had requested
    speculation in the first place.

    So `--spec-type draft-simple` is named for exactly the drafts that cannot
    name themselves, and withheld from the ones that can — overriding a working
    inference is the bug this file already fixed once (issue #19)."""
    dm = getattr(cfg, "draft_model", None)
    if not dm:
        return []
    args = ["-md", str(dm)]
    if not draft_self_describes(dm):
        args += ["--spec-type", "draft-simple"]
    return args


def draft_decides_spec_type(cfg: "Config") -> bool:
    """Whether an explicitly supplied draft model should pick the speculative
    type, rather than the target model's own MTP metadata.

    llama.cpp already infers the type from the draft
    (`common_speculative_types_from_gguf`): architecture `dflash` means
    draft-dflash, or draft-dspark when the Markov head is present; an ordinary
    architecture carrying `blk.N.nextn.eh_proj.weight` means draft-mtp. We do not
    reproduce that inference and must not pre-empt it.

    Before this, a target shipping its own MTP head got `--spec-type draft-mtp`
    unconditionally — so pointing `--draft-model` at a DFlash2 head produced
    `-md dflash.gguf --spec-type draft-mtp`, which loaded the draft and then ran
    MTP anyway, silently measuring the wrong thing (issue #19). A draft model the
    user named on the command line is better evidence of intent than metadata the
    target happens to carry."""
    return bool(getattr(cfg, "draft_model", None))


def model_ssm_state(meta: dict) -> int:
    """SSM state size for a hybrid/recurrent model; 0 for a plain transformer.

    `{arch}.ssm.state_size` — present on `qwen35`/`qwen35moe`, absent on gemma3
    and ordinary transformers. Recurrent memory cannot be rolled back to an
    arbitrary prefix past a certain depth, and unlike SWA there is no flag that
    changes that, so the only honest advice is about depth (issue #15)."""
    return _meta_int(meta, ".ssm.state_size") or 0


def model_swa_window(meta: dict) -> int:
    """Sliding-window size for a SWA model; 0 when the model attends globally.

    `{arch}.attention.sliding_window` — present and non-zero on gemma3, gemma4
    and muse-glimmer here; absent on qwen35 and ordinary transformers. Detectable
    is the whole point: `swa_full` is a knob that decides whether a shared-prefix
    workload gets prefix reuse at all, and a factor nobody knows to set by hand
    is one nobody sets (docs/flag-coverage.md, issue #15)."""
    return _meta_int(meta, ".attention.sliding_window") or 0


def model_nextn_layers(meta: dict) -> int:
    """MTP (multi-token-prediction / NextN) head layers; 0 means no MTP head.
    Present in e.g. Unsloth Dynamic quants that support draft-mtp speculative
    decoding in llama-server."""
    return _meta_int(meta, ".nextn_predict_layers") or 0


# ---------------------------------------------------------------------------
# Factor-level generation
# ---------------------------------------------------------------------------
def thin_to(values: list, n: int) -> list:
    """`values` reduced to at most n entries, endpoints kept.

    For level sets that are a fixed list rather than a computed span (ubatch,
    kv_type): the endpoints are the interesting extremes, so thin the middle."""
    if len(values) <= n or n < 2:
        return list(values[:max(1, n)]) if n < 2 else list(values)
    keep_idx = {round(i * (len(values) - 1) / (n - 1)) for i in range(n)}
    return [v for i, v in enumerate(values) if i in keep_idx]


def draft_layer_count(path: "Path | None") -> int | None:
    """Layer count of the draft GGUF, or None if it cannot be read.

    Separate from the target's `n_layers` because they are different models and
    routinely different sizes; `-ngld` levels generated from the target's count
    would ask the drafter to offload layers it does not have (D5)."""
    if not path:
        return None
    try:
        return _meta_int(read_gguf_metadata(Path(path)), ".block_count")
    except (OSError, ValueError, struct.error):
        return None


def n_levels_span(lo: int, hi: int, n: int = 5) -> list[int]:
    """`n` roughly evenly spaced distinct integer levels in [lo, hi].

    The level COUNT is the sweep's cost dial, which is why this is a parameter
    rather than the 5 it used to hard-code. `choose_array` sizes on the widest
    factor, so trimming one knob to three levels buys nothing while another
    still has five — the array stays L125 either way. Every auto-generated
    numeric factor has to narrow together or none of them does (--levels)."""
    n = max(2, int(n))
    if hi <= lo:
        return [lo]
    raw = [round(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    out = sorted(set(raw))
    # pad toward hi if collisions removed levels
    i = lo
    while len(out) < n and i <= hi:
        if i not in out:
            out.append(i)
        i += 1
    return sorted(out)[:n]


def five_levels_span(lo: int, hi: int) -> list[int]:
    """Back-compat alias: the default width."""
    return n_levels_span(lo, hi, 5)


def thread_levels(phys: int, logical: int, levels: int = 5) -> list[int]:
    cand = {
        max(1, phys // 2),
        max(1, phys * 3 // 4),
        phys,
        (phys + logical + 1) // 2,
        logical,
    }
    lv = sorted(c for c in cand if c >= 1)
    # ensure exactly `levels` distinct values where possible
    n = 1
    while len(lv) < levels and phys + n <= logical:
        lv = sorted(set(lv) | {phys + n})
        n += 1
    if len(lv) > levels:                 # thin from the middle, keep endpoints
        keep = {lv[0], lv[-1]}
        inner = [x for x in lv if x not in keep]
        step = max(1, len(inner) // max(1, levels - 2))
        lv = sorted(keep | set(inner[::step]))[:levels]
    return lv


def ngl_levels(n_layers: int | None, levels: int = 5,
               fits: bool | None = None, recurrent: bool = False) -> list[int]:
    """`levels` values for -ngl, spanning 0 (pure CPU) to all layers.

    `recurrent=True` — the model has SSM/recurrent memory — collapses the grid to
    `[0, 99]`, which on this class are the only two USEFUL placements that run.
    Mapped on `qwen35moe` (`Qwen3.6-35B-A3B`, 40 blocks, `-ncmoe 40`):

        -ngl   0   1   2   3   5  10  20  30  38  39  40  41  99
              ok  ok  ok   X   X   X   X   X   X   X   X  ok  ok

    The crash band is **3 .. n_layers**, and every death is a segfault right after
    `resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta
    Net (chunked) is assigned to device ROCm0`. It is not "any split": 1 and 2
    layers on the GPU are fine — presumably too few for the fused op to be placed
    on ROCm at all — they are simply useless placements, ~CPU speed on a 40-layer
    model.

    **`-ngl n_layers` dies and `n_layers + 1` lives**, which is why the top level
    is 99 and not `n_layers`: at 40 the output tensor takes the last slot and
    block 0 is left on the CPU, which is exactly the straddle that crashes.
    llama.cpp clamps 99 to the real count.

    **Evidence and its limits.** Two models, one architecture family, one backend
    (ROCm): `qwen35moe` as mapped above, plus `qwen35`
    (`Qwen3.6-27B-Q5_K_M`, dense, 64 blocks) dying at `-ngl 5` and `-ngl 40`. The
    gate is `ssm_state > 0`, which is BROADER — Mamba, Jamba, RWKV and Falcon-H1
    all match and none was tested. Applied anyway because the errors are not
    symmetric: wrong here costs three grid levels that
    `--factor ngl=0,16,32,48,64` hands straight back; wrong the other way costs
    three fifths of every sweep on rows that cannot produce a number.

    `fits=True` — the model provably fits in VRAM at the most demanding cell of
    this design — biases the levels toward full offload: the reason to put layers
    on the CPU is memory pressure, and the fit test says there is none, so an even
    span spends its slowest rows where the answer cannot be (issue #14).

    The bias is to the top QUARTER of the range, not to `top-1, top-2`: on a
    40-layer model those differ by ~7% of the model's compute against a measured
    run-to-run spread of 5-27%, so clustering would resolve a difference below the
    noise floor — the same waste in a new place
    (docs/sweep-cost-design.md).

    `ngl=0` survives every level count and every verdict. It is not there for
    information; it is there because the verdict can be WRONG, and it is then the
    only row that can still produce a measurement instead of an OOM."""
    top = n_layers if n_layers else 99
    # Always include 0 (pure CPU) and the top (all layers). 99 = "all" is safe
    # since llama.cpp clamps to the real layer count.
    if recurrent:
        # A conservative DEFAULT, not a verdict: 0 and 99 are the two placements
        # measured to work here, and under --run `probe_loadable_ngl` asks this
        # box which levels actually load and widens the grid back if they do. The
        # collapse only stands where the probe cannot run -- plan-only, or a
        # driver without a server to launch.
        return [0, 99]
    if fits and levels >= 3 and top >= 4:
        # anchor + a span across the top quarter, widened when a quarter is too
        # narrow to hold `levels - 1` distinct values -- otherwise a small model
        # would silently lose levels to deduplication
        lo = max(0, top - max(levels - 1, top // 4))
        return sorted({0} | set(n_levels_span(lo, top, levels - 1)))[:levels]
    mids = n_levels_span(0, top, levels)
    lv = sorted(set([0] + mids + [top]))
    # trim to `levels`, keeping endpoints
    if len(lv) > levels:
        keep = {0, top}
        inner = [x for x in lv if x not in keep]
        step = max(1, len(inner) // max(1, levels - 2))
        lv = sorted(keep | set(inner[::step]))[:levels]
    return lv


def probe_loadable_ngl(cfg: "Config", candidates: list, n_ctx: int,
                       timeout: int) -> tuple[list, list]:
    """Which `-ngl` levels actually load on THIS box. Returns (loadable, dead).

    Replaces a verdict with a measurement. The failure this exists for was found
    on one architecture family and one backend — `qwen35`/`qwen35moe` on ROCm,
    where `-ngl` in `[3, n_layers]` segfaults llama-server — and baking that in
    would delete the layers-for-context axis on hardware where the fused op is
    fine. Which levels load is a property of the model, the backend and the
    build, so it is asked here rather than assumed (issue #18).

    Each candidate is a server launch with a deliberately small context, so the
    cost is dominated by reading the weights (usually already in page cache after
    the first) and nothing is measured — this only asks whether the process
    stands up. Levels that fail are reported, never silently dropped: if the
    answer is surprising, that is the finding."""
    loadable, dead = [], []
    for lv in candidates:
        session = ServerSession(cfg, {**{k: v[0] for k, v in cfg.factors.items()
                                         if v}, "ngl": str(lv)}, n_ctx, timeout)
        try:
            (loadable if getattr(session, "ok", False) else dead).append(lv)
        finally:
            session.close()
    return loadable, dead


def full_offload_fits(cfg: "Config", depths: list, kv_levels: list,
                      n_layers: int | None) -> bool | None:
    """Does every layer fit in VRAM at the most demanding cell of this design?

    True / False / None, where None means "could not tell" and callers must keep
    whatever they would have done anyway. Asked of `predict_fits`, so the grid and
    the OOM pruner cannot disagree about what fits.

    The probe row is deliberately the WORST case the design contains, not a
    typical one: the deepest depth (levels are generated once for the whole design
    while n_depth varies per row, so testing at depth 0 would be optimistic
    exactly where OOM is likeliest), the largest KV type, KV resident on the GPU,
    and no tensor offload. Biasing only when even that cell fits errs toward the
    even span, which is the safe direction (issue #14)."""
    if not cfg.oom_prune or not n_layers:
        return None
    probe = {"ngl": str(n_layers),
             "n_depth": str(max(int(d) for d in depths) if depths else 0),
             # kv_type levels are ordered best-quality first, and best quality is
             # also the largest cache
             "kv_type": str(kv_levels[0]) if kv_levels else "f16",
             "nkvo": "0"}
    return predict_fits(cfg, probe, cfg.driver)


# Don't probe deeper than this by default, even if the model's native context is
# larger — deep contexts on CPU (low -ngl) prefill very slowly and burn memory.
DEFAULT_MAX_DEPTH = 65536
DEFAULT_KV_LEVELS = ["f16", "q8_0", "q5_1", "q4_1", "q4_0"]
DEFAULT_UBATCH_LEVELS = [128, 256, 512, 1024, 2048]

# KV cache types ordered best-quality -> lossiest. q8_0 is near-lossless; below it
# quality degrades and errors compound over context. --min-kv floors this.
KV_QUALITY = ["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]


def kv_at_or_above(levels: list, floor: str) -> list:
    """Keep only KV types at least as high-quality as `floor` (order in
    KV_QUALITY). floor 'any'/'none'/'' disables the filter."""
    if not floor or floor.lower() in ("any", "none"):
        return list(levels)
    fi = KV_QUALITY.index(floor) if floor in KV_QUALITY else len(KV_QUALITY) - 1
    kept = [l for l in levels
            if l not in KV_QUALITY or KV_QUALITY.index(l) <= fi]
    return kept or [floor]  # never empty


def depth_levels(n_ctx_train: int | None, override_max: int | None = None,
                 floor: int = 0, levels: int = 5) -> list[int]:
    """Five n_depth levels spanning floor..min(native ctx, cap). Adaptive so we
    never test beyond the model's native context.  The floor raises the minimum
    depth so the sweep only tests context sizes the user actually needs (--ctx-floor)."""
    top = min(n_ctx_train or DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH)
    if override_max is not None:
        top = min(top, override_max)
    if floor >= top:
        return [top]
    return n_levels_span(floor, top, levels)


def cpu_offload_levels(n_layers: int | None, levels: int = 5) -> list[int]:
    """Levels for a first-N-layers-on-CPU knob (-ncmoe, -ncffn): 0..n_layers."""
    return n_levels_span(0, n_layers if n_layers else 64, levels)


@dataclass
class Config:
    model: Path
    llama_bench: Path
    array: str
    ctx_floor: int
    llama_server: Path = field(
        default_factory=lambda: WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-server")
    reps: int = BENCH_REPS
    n_prompt: int = BENCH_N_PROMPT
    n_gen: int = BENCH_N_GEN
    max_depth: int | None = None  # cap n_depth levels (memory/time budget)
    emit_mtp: bool = True         # add draft-mtp flags to server cmd if supported
    ngram: bool = False           # enable ngram self-speculative decoding sweep
    ngram_type: str | None = None # pin the ngram variant (tuning stage / --ngram-type)
    ngram_keep: int = 2           # variants carried from screen into tuning (top-K)
    spec_draft_n_max: int = 2     # --spec-draft-n-max for MTP
    profile: str = "single"       # workload profile (see PROFILES)
    driver: str = "bench"         # "bench" (llama-bench) or "server" (llama-server)
    parallel: int = 1             # concurrent request streams (server driver)
    server_start_timeout: int = 180  # max seconds to wait for llama-server to load
    measure_vram: bool = False       # sample peak VRAM used during each run
    oom_prune: bool = True           # skip configs predicted to OOM via llama-fit-params
    fit_headroom_mib: int = 512      # safety margin under detected VRAM for OOM pruning
    fit_params: Path = field(
        default_factory=lambda: WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-fit-params")
    score: str = "tg"             # objective: "tg" (decode only) or "eff" (blend pp+tg)
    factors: dict = field(default_factory=dict)
    hw: dict = field(default_factory=dict)
    prefix_reuse: float = 1.0   # workload SHAPE: shared-prefix fraction (W-D1)
    env_factor_names: set = field(default_factory=set)  # factors that set env vars
    # Throughput floors: a config below them is a real measurement the user has
    # said they do not want, so waiting for it to finish buys nothing (T1).
    draft_model: Path | None = None   # -md: a SECOND model, an input not a factor (D1)
    mmproj: Path | None = None        # --mmproj: a THIRD resident artifact, likewise an input
    levels: int = 5             # level count for auto-generated numeric factors
    # set by build_factors: the ngl grid was biased toward full offload because
    # the model provably fits. Recorded so the header can say so (issue #14).
    ngl_biased: bool = False
    # set by build_factors: the model has recurrent memory, so partial -ngl is
    # not a placement that runs (issue #18)
    ngl_recurrent: bool = False
    # the un-collapsed ngl span, kept so a run-time probe can restore levels the
    # conservative default dropped (issue #18)
    ngl_candidates: list = field(default_factory=list)
    min_tgs: float = 0.0        # abandon a config generating slower than this
    min_pps: float = 0.0        # ...or prefilling slower than this
    slow_grace: int = 60        # never judge a config on less than this many seconds


def effective_tps(n_prompt: int, n_gen: int, pp: float, tg: float) -> float:
    """Throughput for a representative request of n_prompt prompt + n_gen gen
    tokens: total tokens / (prefill time + decode time)."""
    if pp <= 0 or tg <= 0:
        return 0.0
    return (n_prompt + n_gen) / (n_prompt / pp + n_gen / tg)


def objective_tps(cfg: Config, pp: float, tg: float) -> float:
    """The score a run contributes to fits/stats/picks (stored as eff_tps).
    Default (--score tg): pure generation speed — pp is still measured and
    reported, but a huge prefill number can't make a slow generator outrank a
    fast one. --score eff blends both into effective request throughput."""
    if cfg.score == "eff":
        return effective_tps(cfg.n_prompt, cfg.n_gen, pp, tg)
    return tg if tg > 0 else 0.0


def build_factors(cfg: Config):
    reg_errors = validate_factor_registry()
    assert not reg_errors, "FACTORS registry invalid: " + "; ".join(reg_errors)
    phys = cfg.hw["phys"]
    logical = cfg.hw["logical"]
    n_layers = cfg.hw.get("n_layers")
    # ONE dial for every auto-generated numeric factor. Narrowing them
    # individually is a no-op: choose_array sizes on the widest, so an L125
    # stays an L125 until the last five-level column is gone (--levels).
    nlv = max(2, int(getattr(cfg, "levels", 5)))
    depths = depth_levels(cfg.hw.get("n_ctx_train"), cfg.max_depth,
                          floor=cfg.ctx_floor, levels=nlv)
    kv_lv = list(DEFAULT_KV_LEVELS)[:nlv]
    # Asked once, at the deepest depth and largest KV in this design: when every
    # layer provably fits, an even 0..n_layers span spends its slowest rows where
    # the answer cannot be (issue #14, docs/sweep-cost-design.md). `--min-kv` is
    # applied to kv_type after this and only ever drops the LOSSIER levels, so
    # f16 -- the largest cache here -- survives it. The one exception is
    # `--min-kv f32`, whose cache is larger still and would make this probe
    # optimistic; the ngl=0 anchor and the OOM pruner are what bound that.
    fits = full_offload_fits(cfg, depths, kv_lv, n_layers)
    cfg.ngl_biased = bool(fits)          # reported in the header, not inferred
    # Recurrent models crash on any partial offload, so the grid has two usable
    # points and there is no sense generating the three that abort (issue #18).
    cfg.ngl_recurrent = int(cfg.hw.get("ssm_state", 0) or 0) > 0
    # What the grid WOULD be without that collapse. `probe_loadable_ngl` asks the
    # box which of these actually load and restores the ones that do, so the
    # conservative default never becomes a ceiling on hardware it was not
    # measured on (issue #18).
    cfg.ngl_candidates = [str(x) for x in ngl_levels(n_layers, nlv, fits)]
    factors = {
        "ngl": [str(x) for x in ngl_levels(n_layers, nlv, fits,
                                           recurrent=cfg.ngl_recurrent)],
        "n_depth": [str(x) for x in depths],
        "threads": [str(x) for x in thread_levels(phys, logical, nlv)],
        "kv_type": list(DEFAULT_KV_LEVELS)[:nlv],
        "ubatch": [str(x) for x in thin_to(DEFAULT_UBATCH_LEVELS, nlv)],
        # KV offload (-nkvo) is the VRAM-vs-bandwidth lever: keeping the KV
        # cache in system RAM frees VRAM for layers at a PCIe cost that only a
        # measurement can price on a given box. (fa stays fixed: flash-attn is
        # a precondition for quantized KV, so sweeping it would just fail every
        # fa=0 × KV-quant row — sweep it via --factor fa=0,1 --min-kv f16.)
        "nkvo": ["0", "1"],
        # An L125 fits 31 factors at the same 125 runs, so the remaining clean
        # knobs ride along free. batch is swept as a MULTIPLE of ubatch, not an
        # absolute: -b >= -ub holds in every row by construction, so no clamp is
        # needed and the low-batch regime (which an absolute floor of 2048 used
        # to hide entirely) is reachable again. 1x/4x/16x over ubatch's
        # 128..2048 spans -b 128..32768.
        "poll": thin_to(["0", "50", "100"], nlv),
        "batch_ratio": thin_to(["1", "4", "16"], nlv),
    }
    if cfg.driver == "server":
        # decode threads (-t) and prefill threads (-tb) can want different
        # counts; only llama-server exposes the split
        factors["threads_batch"] = [str(x) for x in thread_levels(phys, logical, nlv)]
    # only sweep NUMA policy on a machine that actually has multiple nodes
    # (on a single node it's an inert column)
    if cfg.hw.get("numa_nodes", 1) > 1:
        factors["numa"] = ["distribute", "isolate"]
    # For MoE models, expert CPU-offload (-ncmoe) is the biggest RAM/VRAM lever,
    # so promote it to a swept factor. For dense models the equivalent lever is
    # tensor placement (-ot): keeping FFN tensors on CPU at full -ngl often
    # beats dropping whole layers (attn_cpu is left out — attention on CPU
    # kills decode; exps_cpu is inert without experts).
    if cfg.hw.get("n_experts", 0) > 0:
        factors["ncmoe"] = [str(x) for x in cpu_offload_levels(n_layers, nlv)]
    else:
        # Dense models: one placement column spanning both mechanisms. -ncffn
        # (llama.cpp b10645) grades how many layers offload; -ot ffn_up_cpu
        # picks a lighter tensor subset across all of them. Neither subsumes
        # the other, and they cannot be two columns — see FFN_PLACE_NONE.
        drv_bin = cfg.llama_bench if cfg.driver == "bench" else cfg.llama_server
        factors["ffn_place"] = thin_to(
            ffn_place_levels(n_layers, supports_flag(drv_bin, "-ncffn")), nlv)
    # A model with an MTP/NextN head on the server driver: sweep the whole
    # speculative-decoding surface — on/off, draft lengths, and acceptance
    # thresholds — so the report MEASURES what MTP buys instead of assuming it.
    # Bench can't speculate (server-only knobs), and --no-mtp opts out.
    # More than one speculative head available (an embedded MTP head AND a
    # supplied draft head): sweep WHICH one rather than MTP on/off, since that is
    # the question and `mtp` cannot express it (issue #19). The spec_* tuning
    # knobs stay at their defaults here — one comparison at a time.
    _spec_levels = spec_type_levels(cfg)
    if cfg.driver == "server" and len(_spec_levels) > 2:
        factors["spec_type"] = _spec_levels
    elif cfg.driver == "server" and cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        factors["mtp"] = ["1", "0"]
        factors["spec_n_max"] = ["1", "2", "3", "4", "6"]
        # relative: n_min = floor(frac * n_max), so n_min <= n_max always. 0.0 is
        # llama.cpp's own default (common/common.h:326) and 1.0 pins them equal.
        factors["spec_n_min_frac"] = ["0.0", "0.5", "1.0"]
        factors["spec_p_min"] = ["0.0", "0.25", "0.5", "0.75", "0.9"]
        factors["spec_p_split"] = ["0.1", "0.3", "0.5"]
    # llama.cpp toggles we were merely inheriting the defaults of. Each is a
    # documented behaviour switch with no universal answer -- whether repacking
    # weights, offloading host ops, or bypassing the host buffer helps depends on
    # the backend, the CPU and the model, which is the case for MEASURING them
    # rather than picking one. Verified free: an L125 holds 31 factors, so adding
    # these keeps the same 125 runs, and none of them changes how the model is
    # loaded (unlike load_mode, which would make every launch slower).
    #
    # Gated on the binary advertising the flag, since these are recent, and safe
    # to sweep blind because a level that turns out not to run is reported and
    # dropped by `dead_levels` rather than silently poisoning the design.
    if cfg.driver == "server":
        for _name, _flag in (("repack", "--repack"),
                             ("no_op_offload", "--op-offload"),
                             ("no_host", "--no-host")):
            if supports_flag(cfg.llama_server, _flag):
                factors[_name] = ["0", "1"]
    # Sliding-window attention: --swa-full decides whether the SWA layers keep a
    # full-size KV cache. Measured on gemma-3-270m at a 15.7k-token prompt with a
    # 90% shared prefix (issue #15): WITHOUT it the second rep re-prefills the
    # entire prompt (0% cache hit) because the window has scrolled past the
    # prefix; WITH it every rep reuses 90%. That is a 3-6x difference on this
    # workload, paid for in KV memory and slower attention -- exactly the shape
    # of a thing to measure rather than assume. Server-only (`--swa-full` has no
    # llama-bench spelling), and skipped on models that attend globally, where
    # llama.cpp disables the flag itself and every level would be the same run.
    if cfg.driver == "server" and cfg.hw.get("n_swa", 0) > 0:
        factors["swa_full"] = ["0", "1"]
    # Projector placement, present only when there IS a projector — without
    # --mmproj llama.cpp has nothing to offload and every level is the same run.
    if cfg.driver == "server" and cfg.mmproj:
        factors["mmproj_offload"] = ["1", "0"]
    # Draft-model placement (F2 step 2). Present ONLY when a draft model was
    # given: without -md llama.cpp never reads these, so every level would be
    # the same run (D2/DM1). Levels come from the DRAFT model's layer count,
    # not the target's — a 0.5B drafter beside a 27B target has its own
    # geometry, and reusing the target's would emit -ngld values the drafter
    # does not have (D5).
    if cfg.driver == "server" and cfg.draft_model:
        d_layers = draft_layer_count(cfg.draft_model)
        factors["spec_draft_ngl"] = [str(x) for x in
                                     ngl_levels(d_layers, nlv)]
        # The drafter's KV is a separate budget from the target's, and on a card
        # they share it is the cheaper one to quantise (F2's vLLM evidence).
        #
        # Deliberately NOT held to --min-kv. That floor exists to protect output
        # quality, and the draft model does not produce output: a token drafted
        # from a degraded draft KV is either accepted after the target verifies
        # it, or discarded. Quantising the drafter costs ACCEPTANCE RATE, which
        # is speed, and speed is what the sweep is measuring. Applying a quality
        # floor here would rule out the cheap end of the one cache where cheap
        # is free.
        factors["spec_draft_kv"] = thin_to(list(DEFAULT_KV_LEVELS), nlv)
    # ngram self-speculative decoding (server only). The --spec-type variant is a
    # gate; its tuning knobs are CONDITIONAL (active_when — docs/CONDITIONAL-FACTORS.md)
    # and must not share a flat array with the gate (that inflates the design and
    # dilutes their effects). So build_factors emits only the GATE here — the
    # Stage-0 "screen" that measures each variant at its default knobs. A variant's
    # knobs enter the design only once the gate is pinned to it (a tuning stage,
    # cfg.ngram_type — driven by run_ngram_stages or --ngram-type). ngram needs no
    # draft model and complements MTP when both are active (--spec-type values
    # combine). spec_n_max is NOT added: --spec-draft-n-max is a draft-model/MTP
    # knob with no effect on ngram.
    if cfg.driver == "server" and cfg.ngram:
        if cfg.ngram_type:                         # tuning stage: gate pinned
            factors["ngram"] = [cfg.ngram_type]
            factors.update(ngram_child_levels(cfg.ngram_type))
        else:                                      # screen stage: variants only
            factors["ngram"] = ["none", "ngram-simple", "ngram-mod",
                                "ngram-map-k", "ngram-map-k4v"]
    lvl_errors = validate_factor_levels(factors)
    assert not lvl_errors, "factor levels invalid: " + "; ".join(lvl_errors)
    return factors


# Orthogonal-array capacity: name -> number of columns (factors) it holds.
_ARRAY_TABLES = {
    2: [("L4", 3), ("L8", 7), ("L16", 15), ("L32", 31), ("L64", 63), ("L128", 127)],
    3: [("L9", 4), ("L27", 13), ("L81", 40), ("L243", 121)],
    5: [("L25", 6), ("L125", 31), ("L625", 156)],
}


def choose_array(factors: dict) -> str | None:
    """Pick the smallest orthogonal array that fits the factor set, based on the
    factors' level counts. Returns an array name, or None to let the binding
    auto-select. Fixes the binding auto-selecting a 5-level array for 3-level
    factors.

    Only factors that actually vary (>1 level) count: a factor refinement has
    already pinned to a single level is a constant and carries no information
    for an orthogonal array. Sizing on all factors instead would draw a full
    25-run L25 to sweep a lone 5-level factor left among four pinned ones. With
    <=1 varying factor an array is meaningless (return None ⇒ direct sweep)."""
    counts = [len(v) for v in factors.values() if len(v) > 1]
    if len(counts) <= 1:
        return None
    nf, mx = len(counts), max(counts)
    # Mixed level counts ride on the array of the largest base (a 2-level factor
    # maps onto a 5-level column with a modulo imbalance the level means absorb).
    # We deliberately pick from the pure 2/3/5-level arrays only. The library
    # does ship one mixed array, L18 (one 2-level column and seven 3-level), but
    # it fits only a factor set whose largest count is 3 — and ours almost always
    # carries a 5-level factor (ngl, n_depth, threads, ubatch all default to 5),
    # so L18 would apply rarely and is not worth a second selection path until a
    # real factor set wants it.
    base = 2 if mx <= 2 else 3 if mx == 3 else 5 if mx <= 5 else None
    if base is None:
        return None
    for name, cap in _ARRAY_TABLES[base]:
        if cap >= nf:
            return name
    return None


# ---------------------------------------------------------------------------
# Taguchi run generation (via the python binding)
# ---------------------------------------------------------------------------
def generate_runs(factors: dict, array: str | None):
    # Split settled (single-level) factors out of the design. They are constants:
    # feeding them to the orthogonal array adds no information but inflates the
    # run count (a lone 5-level factor among four pinned ones would draw a 25-run
    # L25 — 5 real configs each replicated 5×). Build over the ACTIVE factors and
    # re-attach the constants to every generated run so downstream command
    # builders still see a complete factor set.
    active = {k: v for k, v in factors.items() if len(v) > 1}
    const = {k: v[0] for k, v in factors.items() if len(v) <= 1}

    if len(active) <= 1:
        # 0 or 1 varying factor: an orthogonal array is degenerate — enumerate
        # the level(s) directly (a one-way sweep), no wasted replicate rows and
        # no dependency on the array binding.
        if not active:
            return None, [{"run_id": 1, "factors": dict(const)}]
        (name, levels), = active.items()
        return None, [{"run_id": i + 1, "factors": {**const, name: lvl}}
                      for i, lvl in enumerate(levels)]

    sys.path.insert(0, str(find_taguchi_binding()))
    prepare_taguchi_cli()
    from taguchi import Experiment  # noqa: E402

    if array and array.lower() == "auto":
        array = None
    # The binding takes the array in the constructor; None => auto-select.
    exp = Experiment(array_type=array)
    for name, levels in active.items():
        exp.add_factor(name, levels)
    runs = exp.generate()
    for run in runs:                       # constants ride along on every row
        run["factors"].update(const)
    return exp, runs


# ---------------------------------------------------------------------------
# Staged designs for conditional factors — docs/CONDITIONAL-FACTORS.md.
# A flat orthogonal array cannot hold conditional (active_when) factors without
# inert columns (F1–F4). plan_stages decomposes a full factor set into a sequence
# of flat designs, each with every factor active in every row (invariant I1):
#   - one "screen" stage over the unconditional + gate factors (children excluded,
#     each variant measured at its default knobs), then
#   - one "tune:<gate>=<value>" stage per candidate gate value that has children,
#     pinning the gate to that value and sweeping exactly its children.
# The executor runs the screen, keeps the surviving gate values (top-K), and runs
# their tuning stages; the winner is the best MEASURED config across all stages.
# ---------------------------------------------------------------------------
def _active_when(name: str):
    return FACTORS.get(name, {}).get("active_when")


def prune_gated_factors(factors: dict) -> tuple[dict, list[str]]:
    """Drop factors whose gate is pinned to a value that makes them inert.

    `gated_by: (gate, live)` says this knob only does something when `gate`
    resolves into `live`. It is weaker than `active_when`: an inert level here is
    *legal*, it simply has no effect, so emission is left alone and only the
    DESIGN is pruned. (`active_when` governs flags that would be wrong to emit at
    all, and needs the staged decomposition `plan_stages` builds.)

    Reported on issue #11: `--factor mtp=0` turns speculative decoding off, but
    `spec_n_max`, `spec_n_min_frac`, `spec_p_min` and `spec_p_split` kept their
    full level sets, so the sweep generated 25 runs of a configuration that could
    not differ — the array was sized on four columns that no longer moved
    anything. That costs the time of 24 runs, and it is worse than the time: the
    main-effects table then reports an effect for each of those knobs, computed
    entirely from noise.

    Only prunes when the gate is present AND no level of it is live, so a gate
    that is still being swept keeps its children. Returns (factors, dropped)."""
    dropped = []
    for name, spec in FACTORS.items():
        gate_live = spec.get("gated_by")
        if not gate_live or name not in factors:
            continue
        gate, live = gate_live
        if gate in factors and not any(v in live for v in factors[gate]):
            dropped.append(name)
    return ({n: v for n, v in factors.items() if n not in dropped}, dropped)


def plan_stages(factors: dict) -> list[dict]:
    """Ordered list of staged flat designs for `factors` (name → levels). Each
    stage is {name, factors, pin, gate, value}: `factors` is that stage's OA
    design, `pin` the gate constants that make its conditional children live. The
    screen stage has pin={} (only unconditional + gate factors). Tuning stages are
    ordered parent-gate-first so a nested gate is pinned only after the stage that
    selects it. Every factor in a stage is active in every row (I1)."""
    # children grouped by the (gate, value) that makes them live
    kids: dict = {}
    for n in factors:
        aw = _active_when(n)
        if aw:
            gate, live = aw
            for v in live:
                if gate in factors and v in factors[gate]:
                    kids.setdefault((gate, v), []).append(n)

    screen = {n: factors[n] for n in factors if not _active_when(n)}
    stages = [{"name": "screen", "factors": screen, "pin": {},
               "gate": None, "value": None}]

    def gate_depth(g: str) -> int:                 # nesting depth for ordering
        d, seen, aw = 0, set(), _active_when(g)
        while aw and aw[0] not in seen:
            seen.add(aw[0]); d += 1; aw = _active_when(aw[0])
        return d

    for gate, v in sorted(kids, key=lambda gv: (gate_depth(gv[0]), gv[0],
                                                factors[gv[0]].index(gv[1]))):
        fset = {gate: [v]}                          # gate pinned to one value
        for c in kids[(gate, v)]:
            fset[c] = factors[c]
        stages.append({"name": f"tune:{gate}={v}", "factors": fset,
                       "pin": {gate: v}, "gate": gate, "value": v})
    return stages


# ---------------------------------------------------------------------------
# Command building + execution
# ---------------------------------------------------------------------------
# Named -override-tensor patterns → real llama.cpp tensor regex. "none" emits
# nothing. These place whole tensor classes on CPU to free VRAM for more layers.
OT_PATTERNS = {
    "none": "",
    "ffn_cpu": r"\.ffn_(gate|up|down)\.weight=CPU",   # all FFN on CPU (dense)
    "ffn_up_cpu": r"\.ffn_up\.weight=CPU",
    "exps_cpu": r"\.ffn_.*_exps\.=CPU",               # MoE experts on CPU
    "attn_cpu": r"\.attn_.*=CPU",
}

# ---------------------------------------------------------------------------
# Dense FFN placement (`ffn_place`) — one categorical over two mechanisms.
#
# -ot and -ncffn are not rival spellings of one knob, they are different axes:
# -ot ffn_up_cpu moves the up-projection ONLY, across EVERY layer; -ncffn N
# moves the whole FFN (gate/up/down) for the FIRST N layers. Which tensor
# versus how many layers. At equal VRAM freed they shape PCIe traffic
# differently — a thin slice touched in every layer against a contiguous block
# — so which wins is a measurement, not a deduction, and both belong in the
# default design.
#
# They cannot be two orthogonal-array columns. Verified in llama.cpp (6c84c7d5d,
# common/arg.cpp:2787-2798): -ncffn appends per-layer overrides to the same
# params.tensor_buft_overrides vector -ot writes to, so they COMPOSE — and
# ot=ffn_cpu covers all layers, swallowing every -ncffn level. Those rows would
# record an ncffn level that changed nothing, which is the inert-column failure
# CONSTRAINED-FACTORS.md exists to prevent. Clamping at emission is not the fix
# either: it desyncs the recorded level from the config that actually ran.
#
# So: one column whose levels are mutually exclusive by construction, the shape
# the `concurrency` factor already uses for the same reason. Levels are
# PRE-RENDERED from the layer count (the pattern multi-gpu-design.md plans for
# `ts_levels`), so a level is self-describing in the CSV: `first_16` says what
# ran without a lookup.
FFN_PLACE_NONE = "none"


def ffn_place_levels(n_layers: int | None, has_ncffn: bool) -> list[str]:
    """Levels for the dense FFN placement factor, widest span the build allows.

    Always: none / ffn_up_cpu / ffn_cpu — the three -ot regimes, which every build
    can express. With -ncffn (llama.cpp b10645+) two graded levels are inserted
    between "nothing" and "all of it", giving five. The factor NAME does not
    change with build capability, only the level count, so the CSV column means
    the same thing everywhere and results stay comparable across upgrades."""
    levels = [FFN_PLACE_NONE, "ffn_up_cpu"]
    if has_ncffn:
        top = n_layers or 64
        # quarter/half of the layers: the graded middle -ot cannot reach. Guard
        # against collisions on tiny models, where n//4 and n//2 can coincide.
        for frac in (4, 2):
            n = max(1, top // frac)
            lvl = f"first_{n}"
            if lvl not in levels:
                levels.append(lvl)
    levels.append("ffn_cpu")            # all layers, all three FFN tensors
    return levels


def ffn_place_args(level: str) -> list[str]:
    """One level of `ffn_place` → the argument group it emits ([] for none).

    Both drivers take the same spelling, so this is driver-independent."""
    lvl = str(level)
    if lvl.startswith("first_"):
        return ["-ncffn", lvl.split("_", 1)[1]]
    pat = OT_PATTERNS.get(lvl, "")
    return ["-ot", pat] if pat else []

# ngram --spec-type variants that share one {size-n, size-m, min-hits} knob
# structure (differing only in the flag's variant token). ngram-mod has its own
# {n-match, n-min, n-max}. See docs/ngram-design.md.
NGRAM_MAP_VARIANTS = frozenset({"ngram-simple", "ngram-map-k", "ngram-map-k4v"})

# Conditional-child level sets (levels bracket llama.cpp's defaults within the
# documented bounds). These enter the design only in a variant's tuning stage —
# see plan_stages / run_ngram_stages, docs/CONDITIONAL-FACTORS.md.
NGRAM_MAP_LEVELS = {                      # ngram-simple / map-k / map-k4v
    "ngram_size_n":   ["4", "8", "12", "16", "24"],   # default 12
    "ngram_size_m":   ["8", "16", "32", "48", "64"],  # default 48
    "ngram_min_hits": ["1", "2", "3", "5"],           # default 1
}
NGRAM_MOD_LEVELS = {                      # ngram-mod
    "ngram_mod_n_match":   ["8", "16", "24", "32", "48"],   # default 24
    "ngram_mod_n_min":     ["16", "32", "48", "64", "96"],  # default 48
    # relative: n_max = n_min + off, so n_max >= n_min always. llama.cpp's
    # default pair (n_min 48, n_max 64) is reachable as n_min=48, off=16.
    "ngram_mod_n_max_off": ["0", "16", "32", "48", "64"],
}


def ngram_child_levels(variant: str) -> dict:
    """The tuning-knob level sets for one ngram variant (empty for none/unknown)."""
    if variant in NGRAM_MAP_VARIANTS:
        return dict(NGRAM_MAP_LEVELS)
    if variant == "ngram-mod":
        return dict(NGRAM_MOD_LEVELS)
    return {}

# ---------------------------------------------------------------------------
# Unified knob registry — the one place to add a tunable. Each factor declares
# how it maps onto each driver.
#   bench/server : flag tuple for that driver, or None if unsupported there
#   kind         : "num"   integer, refined onto a finer grid between passes
#                  "float" real value, refined by keeping top levels
#                  "cat"   categorical, refined by keeping top levels
#                  "bool"  0/1; bench takes the value, server emits a bare flag
#   off_flag     : bool only — the server spelling for "disabled", when llama.cpp
#                  defaults the flag ON so omitting it would NOT disable it
#   server_only  : only meaningful with the server driver
#   request      : request-time (n_depth) — not a server launch arg
#   translate    : map named level -> real value ("" ⇒ omit the flag)
#   active_when  : (gate_factor, {live values}) — a CONDITIONAL factor that only
#                  participates when factor `gate_factor` resolves to one of the
#                  live values (e.g. an ngram-mod knob is live only when the
#                  `ngram` gate == "ngram-mod"). Outside its live set the factor is
#                  inert: no flag is emitted (I2) and it is not scored (I3). See
#                  docs/CONDITIONAL-FACTORS.md.
#   flag_for     : gate_value -> flag tuple. For a conditional factor whose flag
#                  spelling depends on the live gate value (collapses several
#                  same-shaped knobs into one, e.g. ngram size-n across variants).
#                  Resolved at emission via the row's gate value; takes the place
#                  of a static `server`/`bench` tuple.
# ---------------------------------------------------------------------------
FACTORS = {
    # --- offload / placement ---
    "ngl":          {"bench": ("-ngl",), "server": ("-ngl",), "kind": "num"},
    "ncmoe":        {"bench": ("-ncmoe",), "server": ("-ncmoe",), "kind": "num"},
    "ncffn":        {"bench": ("-ncffn",), "server": ("-ncffn",), "kind": "num"},
    "ot":           {"bench": ("-ot",), "server": ("-ot",), "kind": "cat",
                     "translate": OT_PATTERNS},
    # Dense FFN placement: one categorical spanning -ot and -ncffn, whose level
    # carries its own flag (see ffn_place_args). `emit` exists because neither
    # `translate` (one flag, many values) nor `flag_for` (server-only, keyed on
    # ANOTHER factor's gate) can express "this level picks the flag".
    "ffn_place":    {"bench": ("-ot",), "server": ("-ot",), "kind": "cat",
                     "emit": ffn_place_args},
    "nkvo":         {"bench": ("-nkvo",), "server": ("-nkvo",), "kind": "bool"},
    # --- batching ---
    # -b is DERIVED from -ub (docs/CONSTRAINED-FACTORS.md): a batch below the
    # micro-batch is a contradiction llama.cpp silently clamps, so an absolute -b
    # column meant either inverted rows or (as it was) a 2048 floor that hid the
    # whole low-batch regime. As a MULTIPLE of -ub every row is valid by
    # construction and the low end is reachable again.
    "batch_ratio":  {"bench": ("-b",), "server": ("-b",), "kind": "num",
                     "derived_from": ("ubatch", "scale"), "relation": "at_least",
                     "abs_name": "batch"},
    "ubatch":       {"bench": ("-ub",), "server": ("-ub",), "kind": "num"},
    # --- KV cache ---
    "kv_type":      {"bench": ("-ctk", "-ctv"), "server": ("-ctk", "-ctv"), "kind": "cat"},
    # --- CPU / threads ---
    "threads":      {"bench": ("-t",), "server": ("-t",), "kind": "num"},
    "threads_batch": {"bench": None, "server": ("-tb",), "kind": "num", "server_only": True},
    "poll":         {"bench": ("--poll",), "server": ("--poll",), "kind": "num"},
    "numa":         {"bench": ("--numa",), "server": ("--numa",), "kind": "cat"},
    "cpu_mask":     {"bench": ("-C",), "server": ("-C",), "kind": "cat"},        # hex affinity mask
    "cpu_strict":   {"bench": ("--cpu-strict",), "server": ("--cpu-strict",), "kind": "cat"},  # 0/1
    "cpu_range":    {"bench": None, "server": ("-Cr",), "kind": "cat", "server_only": True},  # lo-hi
    # --- attention ---
    "fa":           {"bench": ("-fa",), "server": ("-fa",), "kind": "cat"},
    # --- context (request-time) ---
    "n_depth":      {"bench": ("-d",), "server": None, "kind": "num", "request": True},
    # --- speculative decoding / MTP (server only) ---
    "mtp":          {"bench": None, "server": ("--spec-type",), "kind": "cat", "server_only": True,
                     "translate": {"1": "draft-mtp", "0": ""}},   # on/off: "" omits the flag
    "spec_n_max":   {"bench": None, "server": ("--spec-draft-n-max",), "kind": "num", "server_only": True,
                     "gated_by": ("mtp", {"1"})},
    # Draft-model placement. Both are inert without -md (llama.cpp consumes them
    # only inside `if (has_draft)`), so build_factors omits them entirely rather
    # than gating them — an inert column reads as "placement doesn't matter"
    # when the truth is it was never tested (D2/DM1).
    # Projector placement. Like op_offload, llama.cpp defaults this ON, so the
    # disabled level has to be spelled rather than omitted (R3).
    "mmproj_offload": {"bench": None, "server": ("--mmproj-offload",),
                       "kind": "bool", "off_flag": "--no-mmproj-offload",
                       "server_only": True},
    "spec_draft_ngl": {"bench": None, "server": ("-ngld",), "kind": "num",
                       "server_only": True},
    "spec_draft_kv":  {"bench": None, "server": ("-ctkd", "-ctvd"), "kind": "cat",
                       "server_only": True},
    # n_min is DERIVED from n_max as a FRACTION of it. Swept as an absolute it
    # produced inverted rows (n_max=1, n_min=2 — issue #8), and llama.cpp does not
    # reject those: it drafts at most n_max tokens and then discards any draft
    # shorter than n_min (common/speculative.cpp:378), so an inverted row runs
    # with speculation silently OFF while still recording mtp=1 — poisoning the
    # mtp main effect, not merely its own score.
    "spec_n_min_frac": {"bench": None, "server": ("--spec-draft-n-min",), "kind": "float",
                        "server_only": True,
                        "derived_from": ("spec_n_max", "scale"), "relation": "at_most",
                        "abs_name": "spec_n_min", "gated_by": ("mtp", {"1"})},
    "spec_p_min":   {"bench": None, "server": ("--spec-draft-p-min",), "kind": "float", "server_only": True,
                     "gated_by": ("mtp", {"1"})},
    "spec_p_split": {"bench": None, "server": ("--spec-draft-p-split",), "kind": "float", "server_only": True,
                     "gated_by": ("mtp", {"1"})},
    "ngram":             {"bench": None, "server": ("--spec-type",), "kind": "cat", "server_only": True,
                          "translate": {"none": "", "ngram-simple": "ngram-simple", "ngram-mod": "ngram-mod",
                                        "ngram-map-k": "ngram-map-k", "ngram-map-k4v": "ngram-map-k4v"}},
    # ngram tuning knobs are CONDITIONAL on the --spec-type variant (active_when):
    # emitted and scored only in rows whose `ngram` gate selects the owning
    # variant. The map/simple variants share one {size-n, size-m, min-hits}
    # structure differing only in the flag's variant token, so they collapse to
    # one factor each via flag_for. ngram-mod carries its own {n-match, n-min,
    # n-max}. Defaults/bounds: llama.cpp common/common.h (see docs/ngram-design.md).
    # NOTE: --spec-draft-n-max (`spec_n_max`) is a draft-model/MTP knob and has NO
    # effect on ngram — it is deliberately not an ngram factor.
    "ngram_size_n":   {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-size-n",)},
    "ngram_size_m":   {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-size-m",)},
    "ngram_min_hits": {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-min-hits",)},
    "ngram_mod_n_match": {"bench": None, "server": ("--spec-ngram-mod-n-match",), "kind": "num",
                          "server_only": True, "active_when": ("ngram", {"ngram-mod"})},
    "ngram_mod_n_min":   {"bench": None, "server": ("--spec-ngram-mod-n-min",), "kind": "num",
                          "server_only": True, "active_when": ("ngram", {"ngram-mod"})},
    # n_max is DERIVED from n_min as an OFFSET above it — same defect as
    # spec_n_min_frac: speculative.cpp:1927 discards a draft shorter than n_min,
    # so an inverted ngram-mod row measures the baseline, not ngram-mod.
    "ngram_mod_n_max_off": {"bench": None, "server": ("--spec-ngram-mod-n-max",), "kind": "num",
                            "server_only": True, "active_when": ("ngram", {"ngram-mod"}),
                            "derived_from": ("ngram_mod_n_min", "offset"),
                            "relation": "at_least", "abs_name": "ngram_mod_n_max"},
    # --- model loading / buffer placement (docs/remaining-factors-design.md) ---
    # Registered but NOT auto-swept (R1): reachable via --factor, kept out of the
    # default design because nothing detectable says whether they matter here.
    # long spelling on both drivers so the swept flag and the fixed one
    # (load_mode_args) are literally the same string — R4 is then checkable
    "load_mode":    {"bench": ("--load-mode",), "server": ("--load-mode",),
                     "kind": "cat"},
    # Named for the NEGATIVE spelling because llama.cpp is (R2): bench has
    # `-nopo <0|1>` and server `--no-op-offload`, so one level means one thing on
    # both drivers and nothing has to be inverted.
    "no_op_offload": {"bench": ("-nopo",), "server": ("--no-op-offload",),
                      "kind": "bool", "off_flag": "--op-offload"},
    "no_host":      {"bench": ("--no-host",), "server": ("--no-host",), "kind": "bool"},
    # default ON upstream, so 0 must emit --no-repack rather than nothing (R3)
    "repack":       {"bench": None, "server": ("--repack",), "kind": "bool",
                     "off_flag": "--no-repack", "server_only": True},
    # --- attention / cache shape (server only) ---
    "swa_full":     {"bench": None, "server": ("--swa-full",), "kind": "bool",
                     "server_only": True},
    "ctx_checkpoints": {"bench": None, "server": ("-ctxcp",), "kind": "num",
                        "server_only": True},
    "checkpoint_min_step": {"bench": None, "server": ("-cms",), "kind": "num",
                            "server_only": True},
    # --- sampling placement (server only) ---
    "backend_sampling": {"bench": None, "server": ("--backend-sampling",),
                         "kind": "bool", "server_only": True},
    # --- process priority ---
    "prio":         {"bench": ("--prio",), "server": ("--prio",), "kind": "num"},
    "prio_batch":   {"bench": None, "server": ("--prio-batch",), "kind": "num",
                     "server_only": True},
    # --- batch-phase CPU affinity: the twins of the knobs above. We already
    # sweep threads_batch, so we already believe the batch phase is worth its own
    # tuning; these complete that surface.
    "cpu_mask_batch":   {"bench": None, "server": ("-Cb",), "kind": "cat",
                         "server_only": True},
    "cpu_range_batch":  {"bench": None, "server": ("-Crb",), "kind": "cat",
                         "server_only": True},
    "cpu_strict_batch": {"bench": None, "server": ("--cpu-strict-batch",),
                         "kind": "cat", "server_only": True},
    "poll_batch":       {"bench": None, "server": ("--poll-batch",), "kind": "cat",
                         "server_only": True},
    # --- concurrency (server only) ---
    "parallel":     {"bench": None, "server": ("--parallel",), "kind": "num", "server_only": True},
    # Emitted by build_server_args rather than factor_flags: a level maps to
    # zero, two or three flags, which a flag tuple cannot express.
    "concurrency":  {"bench": None, "server": None, "kind": "cat", "server_only": True,
                     "emitted_by_caller": True},
    # Which speculative head to use. Emitted by the caller because the levels do
    # not share a flag: `draft-mtp` names a type, a supplied draft head is `-md`
    # and no type at all (llama.cpp reads it off the file), and `none` is silence.
    "spec_type":    {"bench": None, "server": None, "kind": "cat", "server_only": True,
                     "emitted_by_caller": True},
    # --- context extension / capability (server only) ---
    "rope_scaling": {"bench": None, "server": ("--rope-scaling",), "kind": "cat", "server_only": True},
    "yarn_factor":  {"bench": None, "server": ("--yarn-ext-factor",), "kind": "float", "server_only": True},
}


# ---------------------------------------------------------------------------
# Concurrency and the unified KV cache — docs/concurrency-kv-design.md.
#
# llama.cpp couples these two, so they cannot ride an orthogonal array as free
# columns. llama-server defaults slots to auto (common/arg.cpp), and auto means
# 4 slots AND kv_unified=true (tools/server/server.cpp); ANY explicit --parallel
# disables unified KV, including --parallel 1, because the branch tests < 0.
#
# So one categorical over the combinations that actually exist (K1), rather than
# a `parallel` number plus a `kv_unified` boolean that could express states
# llama.cpp cannot be put into.
# ---------------------------------------------------------------------------
def concurrency_spec(level: str) -> tuple:
    """(slots, unified) for a `concurrency` level.

    Levels: "auto" (emit nothing — llama.cpp picks 4 slots and unified KV),
    "N" (--parallel N, split KV), "Nu" (--parallel N --kv-unified)."""
    lv = str(level).strip().lower()
    if lv in ("auto", ""):
        return 4, True                  # what llama.cpp itself chooses
    unified = lv.endswith("u")
    try:
        slots = int(lv[:-1] if unified else lv)
    except ValueError:
        return 1, False
    return max(1, slots), unified


def slots_for(cfg, f: dict) -> int:
    """Slots this row actually runs with, whichever knob set them.

    `concurrency` wins where present because it is the one that can also say
    `auto`; otherwise the older `parallel` factor, then the config default."""
    if "concurrency" in f:
        return concurrency_spec(f["concurrency"])[0]
    return int(f.get("parallel", getattr(cfg, "parallel", 1)))


def kv_unified_for(cfg, f: dict) -> bool:
    """Whether this row runs with a unified KV cache — derivable from the flags,
    so it can be RECORDED rather than guessed at read time (KV2).

    True only when llama.cpp chooses it (slots left auto) or we ask for it
    explicitly; any bare --parallel leaves it off, including --parallel 1."""
    if "concurrency" in f:
        return concurrency_spec(f["concurrency"])[1]
    if "parallel" in f:
        return False                    # explicit --parallel always disables it
    return int(getattr(cfg, "parallel", 1)) <= 1   # nothing emitted -> auto


def concurrency_flags(level: str) -> list:
    """Server flags for a `concurrency` level. `auto` emits nothing — that is
    the whole point of it, and the state we could not previously reach (KV3)."""
    lv = str(level).strip().lower()
    if lv in ("auto", ""):
        return []
    slots, unified = concurrency_spec(level)
    return ["--parallel", str(slots)] + (["--kv-unified"] if unified else [])


def ctx_slots_multiplier(level: str) -> int:
    """How many times the per-slot context to request for this level (K3).

    llama.cpp gives a slot the FULL n_ctx under unified KV and `n_ctx / slots`
    when the cache is split (src/llama-context.cpp): so a split regime must ask
    for slots x the per-slot context, and a unified one must not. Getting this
    backwards would silently give every row a different real context than its
    ctx_floor claims — the batch-floor defect in a new place (KV4)."""
    slots, unified = concurrency_spec(level)
    return 1 if unified else slots


# ---------------------------------------------------------------------------
# Conditional (nested) factors — see docs/CONDITIONAL-FACTORS.md.
# A factor with `active_when: (gate, {values})` participates only when factor
# `gate` resolves to one of `values` in the assignment (a run row or a pinned
# config). `is_active` is the single source of truth for emission (I2) and
# effect-estimation (I3); the stage planner uses it to keep every OA column live
# (I1). A factor without `active_when` is unconditional (always active).
# ---------------------------------------------------------------------------
def is_active(name: str, assignment: dict) -> bool:
    """Whether factor `name` participates under `assignment`. Unconditional
    factors (and unknown names) are always active; a conditional factor is active
    iff its gate is present in the assignment with a value in its live set."""
    spec = FACTORS.get(name)
    if not spec:
        return True
    cond = spec.get("active_when")
    if cond is None:
        return True
    gate, live = cond
    return assignment.get(gate) in live


def is_inert(name: str, assignment: dict) -> bool:
    """Whether `name` provably could not have acted in this row.

    The `gated_by` counterpart to `is_active`, and it differs from it in the case
    that matters: an ABSENT gate is not evidence. `is_active` treats a missing
    gate as inactive, deliberately (I1) — a conditional flag must not leak into a
    row that never established its gate. Here the question is the opposite one,
    "can this row tell us anything about this knob", and a row with no `mtp`
    column has not said that speculation is off. `--draft-model` speculates with
    no `mtp` column at all, so treating its absence as inertness would delete
    every draft-model row from `--spec-draft-n-max`'s effect and stop emitting
    the flag (issue #16)."""
    gate_live = FACTORS.get(name, {}).get("gated_by")
    if not gate_live:
        return False
    gate, live = gate_live
    return gate in assignment and assignment.get(gate) not in live


def active_factors(names, assignment: dict) -> list:
    """The subset of `names` active under `assignment` (a gate value map)."""
    return [n for n in names if is_active(n, assignment)]


def conditional_flags(name: str, gate_value) -> tuple | None:
    """Server flag tuple for a conditional factor, resolved from the live gate
    value via `flag_for` when the flag spelling is variant-dependent. Falls back
    to the static `server` tuple. None if the factor emits nothing here."""
    spec = FACTORS.get(name, {})
    ff = spec.get("flag_for")
    if ff is not None:
        return ff(gate_value)
    return spec.get("server")


# ---------------------------------------------------------------------------
# Constrained (derived) factors — see docs/CONSTRAINED-FACTORS.md.
# Some knobs come in ordered pairs: -b >= -ub, n_min <= n_max. An orthogonal
# array varies its columns independently, so sweeping BOTH members as absolutes
# emits inverted rows — and llama.cpp accepts them, then behaves as if the
# feature were switched off (issue #8). Clamping at emission would desync the
# recorded level from the config that actually ran; dropping the rows would
# unbalance the array. Instead the DEPENDENT member of a pair declares
#     "derived_from": (base_factor, "scale" | "offset")
#     "relation":     "at_most" | "at_least"
# and its LEVELS become relative to the base — a "scale" factor multiplies the
# base, an "offset" factor adds to it. Every row then satisfies the relation by
# construction, the relative level remains a fully orthogonal design axis, and
# the absolute value is a pure function of the two levels in the same row.
# ---------------------------------------------------------------------------
# What to derive from when the base factor is not itself being swept: the fixed
# value that run will use for it.
DERIVED_BASE_FALLBACK = {
    "ubatch":          lambda cfg: 512,                            # llama.cpp -ub default
    "spec_n_max":      lambda cfg: getattr(cfg, "spec_draft_n_max", 2),
    "ngram_mod_n_min": lambda cfg: 48,                             # common/common.h:355
}


# Factors that USED to be swept as absolutes and are now derived (issue #8).
# The rename is deliberate: silently rereading an old `--factor batch=2048` as
# "2048 x ubatch" would be exactly the class of quiet misinterpretation this
# mechanism exists to remove, so the old spelling has to fail loudly.
RENAMED_FACTORS = {
    "batch": ("batch_ratio",
              "-b is now swept as a MULTIPLE of -ub (e.g. batch_ratio=1,4,16); "
              "the emitted -b is batch_ratio x ubatch"),
    "spec_n_min": ("spec_n_min_frac",
                   "--spec-draft-n-min is now swept as a FRACTION of spec_n_max "
                   "(e.g. spec_n_min_frac=0.0,0.5,1.0), so n_min <= n_max always"),
    "ngram_mod_n_max": ("ngram_mod_n_max_off",
                        "--spec-ngram-mod-n-max is now swept as an OFFSET above "
                        "ngram_mod_n_min (e.g. ngram_mod_n_max_off=0,16,32), so "
                        "n_max >= n_min always"),
}


def derived_base(name: str, f: dict, cfg=None) -> int:
    """Absolute value of `name`'s base sibling in row `f` (its fixed default when
    the base is not part of this design)."""
    base_name = FACTORS[name]["derived_from"][0]
    if base_name in f:
        return int(f[base_name])
    return DERIVED_BASE_FALLBACK[base_name](cfg)


def derived_value(name: str, f: dict, cfg=None) -> int:
    """Materialize a derived factor's ABSOLUTE value from its relative level and
    its base sibling in the same row.

    Rounding is picked so the declared relation holds unconditionally: floor for
    `at_most`, ceil for `at_least`. round() would not do — banker's rounding
    sends 0.5*1 to 0 but 0.5*3 to 2, and only floor makes level 1.0 land exactly
    on the base. The final clamp is belt-and-braces for hand-written levels."""
    spec = FACTORS[name]
    _, op = spec["derived_from"]
    at_most = spec["relation"] == "at_most"
    base = derived_base(name, f, cfg)
    lvl = float(f[name])
    if op == "scale":
        raw = math.floor(lvl * base) if at_most else math.ceil(lvl * base)
    else:                                              # "offset"
        raw = base + int(lvl)
    return max(0, min(raw, base)) if at_most else max(raw, base)


def derived_names(names=None) -> list:
    """The derived factors among `names` (default: the whole registry)."""
    return [n for n in (FACTORS if names is None else names)
            if FACTORS.get(n, {}).get("derived_from")]


def derived_abs_name(name: str) -> str:
    """Column name for a derived factor's materialized absolute value."""
    return FACTORS.get(name, {}).get("abs_name", name + "_abs")


def derived_abs_cols(cfg, f: dict) -> dict:
    """The absolute values of the derived factors in row `f`, keyed by their
    absolute column name. A derived factor's own column holds its RELATIVE level
    (that is the design axis the main effects are computed over); this records
    what actually reached llama.cpp alongside it, so a CSV row is reproducible on
    its own (C3). Blank for a conditional factor inactive in this row — nothing
    was emitted for it."""
    return {derived_abs_name(n): (derived_value(n, f, cfg) if is_active(n, f) else "")
            for n in derived_names(f)}


def derived_base_pins(names) -> set:
    """Bases that `names` derive from but that are NOT themselves swept here.
    Such a base has to be pinned explicitly on the command line: the relation was
    guaranteed against the fallback value, so letting llama.cpp apply its own
    default instead would break C1."""
    return {FACTORS[n]["derived_from"][0] for n in derived_names(names)
            if FACTORS[n]["derived_from"][0] not in names}


def validate_factor_registry(factors: dict = FACTORS) -> list:
    """Return a list of registry errors (empty ⇒ valid). Checks that every
    conditional factor's gate exists, its live values are real gate levels (when
    the gate enumerates them via `translate`), `flag_for` is total over the live
    set, the factor can emit a flag, and the gate graph is acyclic; and that every
    derived factor's base exists, is numeric, is not itself derived, and that the
    derivation graph is acyclic. Run in selftest and asserted at build_factors
    time so a bad registry fails fast."""
    errors: list[str] = []
    edges: dict[str, str] = {}          # factor -> its gate (for cycle check)
    for name, spec in factors.items():
        cond = spec.get("active_when")
        if cond is None:
            continue
        try:
            gate, live = cond
            live = set(live)
        except (TypeError, ValueError):
            errors.append(f"{name}: active_when must be (gate, {{values}})")
            continue
        edges[name] = gate
        if gate not in factors:
            errors.append(f"{name}: active_when gate '{gate}' is not a factor")
        else:
            tr = factors[gate].get("translate")
            if tr is not None:
                unknown = live - set(tr)
                if unknown:
                    errors.append(f"{name}: active_when values "
                                  f"{sorted(unknown)} are not levels of gate '{gate}'")
        ff = spec.get("flag_for")
        if ff is not None:
            for v in sorted(live):
                try:
                    out = ff(v)
                except Exception as e:                       # noqa: BLE001
                    errors.append(f"{name}: flag_for({v!r}) raised {e!r}")
                    continue
                if not out or not all(isinstance(x, str) and x for x in out):
                    errors.append(f"{name}: flag_for({v!r}) must return a "
                                  f"non-empty flag tuple, got {out!r}")
        elif spec.get("server") is None and spec.get("bench") is None:
            errors.append(f"{name}: conditional factor has no flag mapping "
                          f"(need server/bench tuple or flag_for)")
    # Derived (constrained) factors: base must exist, be numeric, and be a leaf
    # of the derivation graph — deriving from a derived factor would make a row's
    # absolute value depend on a chain the OA never balanced.
    dedges: dict[str, str] = {}         # factor -> its base (for cycle check)
    for name, spec in factors.items():
        dfrom = spec.get("derived_from")
        if dfrom is None:
            continue
        try:
            base, op = dfrom
        except (TypeError, ValueError):
            errors.append(f"{name}: derived_from must be (base, 'scale'|'offset')")
            continue
        dedges[name] = base
        if op not in ("scale", "offset"):
            errors.append(f"{name}: derived_from op must be 'scale' or 'offset', "
                          f"got {op!r}")
        if spec.get("relation") not in ("at_most", "at_least"):
            errors.append(f"{name}: derived factor needs relation "
                          f"'at_most' or 'at_least', got {spec.get('relation')!r}")
        if base not in factors:
            errors.append(f"{name}: derived_from base '{base}' is not a factor")
        else:
            if factors[base].get("kind") not in ("num", "float"):
                errors.append(f"{name}: derived_from base '{base}' must be numeric "
                              f"(kind num/float), got {factors[base].get('kind')!r}")
            if factors[base].get("derived_from") is not None:
                errors.append(f"{name}: derived_from base '{base}' is itself "
                              "derived (chains are not supported)")
        if base not in DERIVED_BASE_FALLBACK:
            errors.append(f"{name}: base '{base}' has no DERIVED_BASE_FALLBACK "
                          "entry (needed when the base is not swept)")

    def has_cycle(graph: dict) -> str | None:
        """The first node on a cycle in a node -> node graph, else None."""
        WHITE, GREY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph}

        def visit(n: str) -> bool:                           # True ⇒ cycle found
            color[n] = GREY
            g = graph.get(n)
            if g in graph:
                if color[g] == GREY or (color[g] == WHITE and visit(g)):
                    return True
            color[n] = BLACK
            return False

        for n in graph:
            if color[n] == WHITE and visit(n):
                return n
        return None

    n = has_cycle(edges)                    # factor -> gate must be a DAG
    if n is not None:
        errors.append(f"active_when gate graph has a cycle through '{n}'")
    n = has_cycle(dedges)                   # factor -> base must be a DAG
    if n is not None:
        errors.append(f"derived_from graph has a cycle through '{n}'")
    return errors


def validate_factor_levels(factors: dict) -> list:
    """Return errors for a LEVEL SET (name -> levels), as opposed to the registry.
    A derived factor's levels are relative to its base, so they must honour the
    declared relation: `at_most` needs scale levels <= 1 / offsets <= 0,
    `at_least` needs scale levels >= 1 / offsets >= 0. Checked at build/argparse
    time so `--factor spec_n_min_frac=2.0` fails before the sweep starts, not
    silently as an inverted row (C1).

    For `emit` factors the levels ARE the flags, so two levels that emit the
    same arguments are two names for one run: the array balances a column whose
    levels are not distinct, and the main effect reads as "placement doesn't
    matter" when the truth is that it was never varied. Caught here rather than
    in a test because the failure is silent at every later stage — a level
    whose spelling does not match its pattern table emits nothing at all and
    collides with `none`."""
    errors: list[str] = []
    for name, levels in factors.items():
        spec = FACTORS.get(name, {})
        if spec.get("emit") is not None:
            seen: dict[tuple, str] = {}
            for lvl in levels:
                try:
                    key = tuple(spec["emit"](lvl))
                except Exception as e:                          # noqa: BLE001
                    errors.append(f"{name}: emit({lvl!r}) raised {e!r}")
                    continue
                if key in seen:
                    errors.append(
                        f"{name}: levels {seen[key]!r} and {lvl!r} both emit "
                        f"{list(key)!r} — an orthogonal-array column whose "
                        f"levels are not distinct measures nothing")
                seen[key] = lvl
        dfrom = spec.get("derived_from")
        if dfrom is None:
            continue
        _, op = dfrom
        at_most = spec.get("relation") == "at_most"
        bound = 1.0 if op == "scale" else 0.0
        what = f"{'a fraction of' if at_most else 'a multiple of'} {dfrom[0]}" \
            if op == "scale" else f"an offset from {dfrom[0]}"
        for lvl in levels:
            try:
                v = float(lvl)
            except (TypeError, ValueError):
                errors.append(f"{name}: level {lvl!r} is not numeric "
                              f"({name} is {what})")
                continue
            if at_most and v > bound:
                errors.append(f"{name}: level {lvl} > {bound:g} would make "
                              f"{name} exceed {dfrom[0]} ({name} is {what})")
            elif not at_most and v < bound:
                errors.append(f"{name}: level {lvl} < {bound:g} would make "
                              f"{name} fall below {dfrom[0]} ({name} is {what})")
    return errors


def factor_flags(cfg: Config, f: dict, driver: str) -> list[list[str]]:
    """Argument groups for the sweepable factors in `f` on the given driver, e.g.
    [["-ngl","64"], ["-nkvo"], ["-ctk","f16"]]. Skips env factors, request-time
    factors (n_depth), factors unsupported on the driver, and conditional factors
    whose gate isn't selected in this row (I2). Handles kv (two flags), booleans,
    derived factors (relative level -> absolute value, C1/C4), named -ot patterns,
    and variant-dependent flag_for spellings."""
    groups = []
    for name, val in f.items():
        spec = FACTORS.get(name)
        if spec is None or name in cfg.env_factor_names or spec.get("request"):
            continue
        if not is_active(name, f):                 # conditional factor inactive here
            continue
        if is_inert(name, f):     # gate switched the feature off: the flag is
            continue              # legal but does nothing, so do not paste it
        if spec.get("emit") is not None:
            # The LEVEL picks the flag, not just the value. Emitted only where
            # the driver has some mapping at all, so a driver that cannot
            # express the factor still skips it.
            if spec.get(driver) is None:
                continue
            group = spec["emit"](val)
            if group:
                groups.append(list(group))
            continue
        if spec.get("flag_for") is not None:       # variant-dependent flag spelling
            if driver != "server":                 # flag_for factors are server-only
                continue
            flags = conditional_flags(name, f.get(spec["active_when"][0]))
        else:
            flags = spec.get(driver)
        if flags is None:
            continue
        if spec.get("translate") is not None:
            val = spec["translate"].get(str(val), str(val))
            if val == "":
                continue
        if spec.get("derived_from") is not None:    # relative level -> absolute
            val = str(derived_value(name, f, cfg))
        if spec["kind"] == "bool":
            if driver == "server":
                if str(val) in ("1", "on", "true", "True"):
                    groups.append([flags[0]])      # server: bare flag when enabled
                elif spec.get("off_flag"):
                    # llama.cpp defaults this ON, so "disabled" has to be SAID.
                    # Emitting nothing would leave it enabled while the column
                    # claims otherwise (docs/remaining-factors-design.md, R3).
                    groups.append([spec["off_flag"]])
            else:
                groups.append([flags[0], str(val)])  # bench: -flag 0|1
        else:
            for fl in flags:
                groups.append([fl, str(val)])
    # Pin any base a derived factor leaned on but that is not swept here, at the
    # exact value the derivation assumed (C1). Without this, `--factor
    # spec_n_min_frac=1.0` on a design that does not sweep spec_n_max would
    # compute n_min against our fallback while llama.cpp used its own n_max.
    for base_name in derived_base_pins(f):
        if base_name in cfg.env_factor_names:
            continue
        flags = FACTORS.get(base_name, {}).get(driver)
        if flags is None:
            continue
        pin = str(DERIVED_BASE_FALLBACK[base_name](cfg))   # base absent from f
        for fl in flags:
            groups.append([fl, pin])
    return groups


def _flat(groups: list[list[str]]) -> list[str]:
    return [tok for g in groups for tok in g]


def is_server_only(name: str) -> bool:
    return bool(FACTORS.get(name, {}).get("server_only"))


def bench_command(cfg: Config, f: dict) -> list[str]:
    cmd = [
        str(cfg.llama_bench),
        "-m", str(cfg.model),
        # fixed unless swept: emitting both would give llama.cpp the flag twice
        # and it takes the last one (docs/remaining-factors-design.md, R4)
        *(load_mode_args(cfg.llama_bench, "bench") if "load_mode" not in f else []),
        "-p", str(cfg.n_prompt),
        "-n", str(cfg.n_gen),
        "-r", str(cfg.reps),
        "-o", "json",
    ]
    # No auto-fit suppression here: llama-bench has no --fit. It exposes only
    # --fit-target/--fit-ctx, both off unless asked, so there is nothing to
    # disable — and emitting a flag it does not know makes it exit non-zero,
    # which would fail every bench run rather than fail safe.
    if "fa" not in f:                              # flash-attn fixed unless swept
        cmd += ["-fa", str(FIXED_FA)]
    ub = int(f.get("ubatch", 512))
    if "batch_ratio" not in f:                     # batch fixed; needs -b >= -ub
        cmd += ["-b", str(max(FIXED_BATCH, ub))]
    cmd += _flat(factor_flags(cfg, f, "bench"))
    return cmd


def run_env(cfg: Config, f: dict) -> dict:
    """Process environment for a run: base env plus any env-factor values."""
    env = dict(os.environ)
    for name in cfg.env_factor_names:
        if name in f:
            env[name] = f[name]
    return env




def _merge_spec_type_parts(parts: list[str]) -> list[str]:
    """Combine multiple --spec-type <value> entries in a string-parts list
    into a single comma-separated --spec-type <v1,v2> entry. llama.cpp accepts
    comma-separated spec types and the orthogonal array may emit one --spec-type
    from the mtp factor and another from the ngram factor simultaneously."""
    kept: list[str] = []
    vals: list[str] = []
    for p in parts:
        if p.startswith("--spec-type "):
            v = p.split(None, 1)[1].strip()
            if v:
                vals.append(v)
        else:
            kept.append(p)
    if vals:
        kept.append("--spec-type " + ",".join(vals))
    return kept


def _merge_spec_type_args(args: list[str]) -> list[str]:
    """Combine multiple --spec-type / value flag-pairs in a flat arg list
    into a single --spec-type / v1,v2 pair."""
    result: list[str] = []
    vals: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--spec-type" and i + 1 < len(args):
            v = args[i + 1]
            if v:
                vals.append(v)
            i += 2
        else:
            result.append(args[i])
            i += 1
    if vals:
        result.append("--spec-type")
        result.append(",".join(vals))
    return result


def server_command(cfg: Config, f: dict, ctx: int) -> str:
    ub = int(f.get("ubatch", 512))
    parts = [f"-m {cfg.model.name}", f"-c {ctx}"]
    # one part per flag+value: `parts` is rendered one-per-line in the emitted
    # command, and a bare "--load-mode" over "mmap" reads like two flags
    _lm = load_mode_args(cfg.llama_server, "server") if "load_mode" not in f else []
    if _lm:
        parts.append(" ".join(_lm))
    if "fa" not in f:
        parts.append(f"-fa {FIXED_FA}")
    if "batch_ratio" not in f:
        parts.append(f"-b {max(FIXED_BATCH, ub)}")
    parts += [" ".join(shlex.quote(t) for t in g)
              for g in factor_flags(cfg, f, "server")]
    # The pasted command is the deliverable, so it has to carry the draft model
    # the row was measured with. It did not: a --draft-model sweep printed a
    # command with no `-md` at all, which would not reproduce the number above it.
    _dm = (spec_type_args(cfg, f["spec_type"]) if "spec_type" in f
           else draft_model_args(cfg))
    if _dm:
        parts.append(" ".join(shlex.quote(t) for t in _dm))
    if "concurrency" in f:
        parts += [" ".join(concurrency_flags(f["concurrency"]))] if \
            concurrency_flags(f["concurrency"]) else []
    elif "parallel" not in f and cfg.parallel > 1:
        parts.append(f"--parallel {cfg.parallel}")
    # Multi-token prediction: if the model ships a NextN/MTP head, enable
    # draft-mtp speculative decoding for extra generation throughput. With the
    # server driver this speedup IS measured; with llama-bench it is NOT (bench
    # can't do speculative decoding) and stacks on top of the reported t/s.
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        # MTP fixed on unless swept -- or unless a draft model was supplied, which
        # carries its own type and must not be overridden (issue #19)
        if "mtp" not in f and not draft_decides_spec_type(cfg):
            parts.append("--spec-type draft-mtp")
        # (skipped when factor_flags already pinned it for a derived sibling, and
        # when mtp is explicitly OFF -- there is no drafter for it to bound, and
        # a pasted command should not carry a flag that does nothing)
        if (f.get("mtp") != "0" and "spec_n_max" not in f
                and "spec_n_max" not in derived_base_pins(f)):
            parts.append(f"--spec-draft-n-max {cfg.spec_draft_n_max}")
    # ngram self-speculation: pattern-matching decoding that needs no draft model.
    # When enabled but not swept, activate a sensible default variant (ngram-mod);
    # its own draft length is ngram-mod's default, not --spec-draft-n-max (a
    # draft-model/MTP knob that has no effect on ngram).
    if cfg.ngram and "ngram" not in f:
        parts.append("--spec-type ngram-mod")
    # MTP + ngram can both emit --spec-type; merge into one comma-separated value.
    parts = _merge_spec_type_parts(parts)
    cmd = " \\\n    ".join(["./llama-server"] + parts)
    # prepend any winning env-var factor values as an env prefix
    env_prefix = " ".join(f"{n}={f[n]}" for n in sorted(cfg.env_factor_names) if n in f)
    return (env_prefix + " \\\n  " + cmd) if env_prefix else cmd


def died_on_signal(returncode) -> int | None:
    """The signal a child died on, or None if it exited normally.

    Worth distinguishing from a non-zero exit, because the two mean different
    things and only one of them is about the config. A process that *ran* and
    returned an error was rejected by llama.cpp; a process killed by a signal
    crashed inside it.

    Observed on this project's own hardware: Qwen3.8 (Gated Delta Net) on gfx906
    segfaults during context init above roughly 64k, with **no allocation failure
    logged at all** — the OOM patterns do not match, so it would otherwise land
    as an indistinguishable generic ERROR. It is not an OOM: the same model
    cleanly reports "cudaMalloc failed: out of memory" at 200k, well above where
    the segfaults start.

    subprocess reports this as a negative returncode; shells report 128+N."""
    try:
        rc = int(returncode)
    except (TypeError, ValueError):
        return None
    if rc < 0:
        return -rc
    return rc - 128 if 128 < rc < 160 else None


_OOM_PAT = re.compile(
    r"out of memory|failed to allocate|ROCm error|hipErrorOutOfMemory|"
    r"cudaErrorMemoryAllocation|ggml_backend_.*failed",
    re.IGNORECASE,
)


def parse_bench_json(stdout: str):
    """Return (pp_tps, tg_tps) from llama-bench JSON output, or (None, None)."""
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return None, None
    pp = tg = None
    for row in rows:
        n_gen = row.get("n_gen", 0)
        n_prompt = row.get("n_prompt", 0)
        ts = row.get("avg_ts")
        if ts is None:
            continue
        if n_gen and not n_prompt:
            tg = ts
        elif n_prompt and not n_gen:
            pp = ts
    return pp, tg


def parse_bench_backend(stdout: str) -> str:
    """llama-bench's own name for what actually ran ("ROCm", "CPU", ...), or "".

    The companion to gpu_visibility: that check runs BEFORE a sweep and infers a
    CPU-only build from a device list, this records what the binary says AFTER
    each run. A sweep whose rows all say "CPU" while `ngl` varies is the same
    3.9x fault, visible in the results file long after the console output is
    gone — and visible to whoever is handed the CSV, who did not see the warning
    at all."""
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return ""
    for row in rows:
        if isinstance(row, dict) and row.get("backends"):
            return str(row["backends"])
    return ""


# ---------------------------------------------------------------------------
# Measurement validity — docs/measurement-validity.md.
# "OK" only ever meant the process exited cleanly and a number parsed; it never
# meant the number could be true. llama-bench derives throughput as
# 1e9 * n_tokens / t_ns using the NOMINAL token count (llama-bench.cpp:1529), so
# a decode loop that returns without decoding keeps the numerator and collapses
# the denominator — issue #3 saw 1,000,000 t/s (exactly 1.0 us/token for any
# n_gen) crowned the winner on a box that really does ~25 t/s. Repetition does
# not catch this: the fault is deterministic, so verify_picks confirmed it at
# "spread 0%". Reproducibility is not validity.
# ---------------------------------------------------------------------------
# Backstop for when prefill is unavailable (I2). Generous by design: rejecting a
# real measurement silently deletes a config from the design, which is the worse
# error (P3).
MAX_PLAUSIBLE_TPS = 100_000.0
# I1 margin. Decode re-reads the weights per token and cannot outrun batched
# prefill; the honest ratio runs 10-100x in prefill's favour, and speculative
# decoding moves tg by small integer factors. 10x rejects only the absurd —
# issue #3's row sits at 2251x.
TG_OVER_PP_LIMIT = 10.0


def implausible_reason(pp: float, tg: float, parallel: int = 1) -> str | None:
    """Why this measurement cannot be true, or None if it might be.

    Deliberately one-sided: it rejects impossibly FAST decode. A too-slow number
    is a real (if disappointing) measurement and must survive."""
    if tg <= 0:
        return None                        # not a measurement; other paths own it
    if pp > 0 and tg > TG_OVER_PP_LIMIT * pp:
        return f"tg={tg:.1f} exceeds pp={pp:.1f} by {tg / pp:.0f}x (decode cannot outrun prefill)"
    ceiling = MAX_PLAUSIBLE_TPS * max(1, parallel)
    if tg >= ceiling:
        return f"tg={tg:.1f} t/s exceeds the plausible ceiling ({ceiling:.0f} t/s)"
    return None


def validate_measurement(res: dict, parallel: int = 1) -> dict:
    """Downgrade a physically impossible measurement to IMPLAUSIBLE (I3).

    Applied where measurements are born, not where they are reported (I4), so
    the sweep, the max-context probe and pick verification all inherit it —
    otherwise the probe would keep steering by numbers the report rejects."""
    if res.get("status") != "OK":
        return res
    why = implausible_reason(float(res.get("pp_tps") or 0.0),
                             float(res.get("tg_tps") or 0.0), parallel)
    if why:
        res["status"] = "IMPLAUSIBLE"
        res["implausible"] = why
        # Zero the scores so anything reading them before checking status (the
        # analyzer scores failures as 0) treats this as the non-result it is.
        # eff_tps too when present: a row loaded from CSV arrives with one
        # already computed, and leaving it would let the objective keep the
        # rejected number.
        res["pp_tps"] = res["tg_tps"] = 0.0
        if "eff_tps" in res:
            res["eff_tps"] = 0.0
    return res


def estimate_secs_per_run(cfg) -> float:
    """Rough seconds per sweep run, scaled by the things that actually drive it.

    Was a flat 90s, which reported the same ~187 minutes for a 270M model and a
    27B one (docs/constants-audit.md C-F). Users decide whether to start a
    multi-hour sweep from this number, so being wrong by an order of magnitude at
    both ends is worse than being roughly right.

    Model size dominates: every run reloads the weights (the server driver
    amortises this across configs that share load params, the bench driver does
    not). Reps and generated tokens scale the measured part on top. Deliberately
    crude and labelled as a guess — the honest estimate is the one the sweep
    prints from its own elapsed time once it is running."""
    gib = 0.0
    try:
        gib = cfg.model.stat().st_size / (1024 ** 3)
    except OSError:
        pass
    load = 6.0 + 2.5 * gib                      # weights off disk / into VRAM
    if cfg.driver == "server":
        load *= 0.5                             # sessions are reused across runs
    # The decode-rate prior has to scale too, or the estimate just moves the
    # hardcoded assumption somewhere less visible: a fixed "20 tok/s" is as
    # wrong for a 270M model (hundreds) as for a 27B one (tens). Throughput
    # roughly tracks 1/params at fixed bandwidth, which is the shape used here.
    tps_prior = min(300.0, max(5.0, 300.0 / max(0.3, gib)))
    measure = max(1, cfg.reps) * (cfg.n_gen / tps_prior) + 4.0
    return max(8.0, load + measure)


def fmt_dur(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# Time budget for one config (T1).
#
# `--timeout` meant different things per driver. On bench it bounds the whole
# llama-bench process, which is what it reads like. On the server driver it was
# handed to each HTTP request, so one config could legitimately spend
# (1 warm + reps) x timeout — at the default that is 80 minutes for a "20 minute"
# timeout. It was never a bound on anything a user could name.
#
# It is now a DEADLINE for the whole config on both drivers: launch, warm-up and
# every rep together. The per-request value is whatever is left of it.
#
# A throughput floor tightens that deadline rather than adding a second
# mechanism. If a config must reach `--min-tgs` to be of interest, then a config
# that IS of interest finishes `reps x n_gen` tokens within `tokens / min_tgs`
# seconds; anything slower has already answered the question and does not need
# to finish. This is why the floor buys time on both drivers, including
# llama-bench, where nothing can be observed mid-run: the bound is arithmetic,
# not instrumentation.
# ---------------------------------------------------------------------------
def slow_budget_secs(cfg, timeout: int) -> int:
    """Seconds to allow one config, given the throughput floor (0 ⇒ --timeout).

    `slow_grace` is a floor on the floor: with a small `n_gen` the derived bound
    can be a couple of seconds, and judging a config on that measures model load
    and scheduler noise rather than its throughput."""
    min_tgs = float(getattr(cfg, "min_tgs", 0.0) or 0.0)
    if min_tgs <= 0:
        return int(timeout)
    reps = max(1, getattr(cfg, "reps", 1))
    # +1 covers the warm/prefill request the server driver sends before its reps;
    # on bench it is slack, and slack is the safe direction here (P3).
    tokens = getattr(cfg, "n_gen", 0) * (reps + 1)
    derived = tokens / min_tgs
    return int(max(getattr(cfg, "slow_grace", 60), min(float(timeout), derived)))


def too_slow_reason(cfg, pp: float, tg: float) -> str | None:
    """Why this measurement is below the floors the user set, or None.

    Separate from `implausible_reason`: that one rejects numbers that cannot be
    true, this one rejects numbers that are true and unwanted. Keeping them
    apart matters because the CSV should not call a slow machine a broken one."""
    min_pps = float(getattr(cfg, "min_pps", 0.0) or 0.0)
    min_tgs = float(getattr(cfg, "min_tgs", 0.0) or 0.0)
    if min_pps > 0 and 0 < pp < min_pps:
        return f"pp={pp:.1f} t/s is below --min-pps {min_pps:.1f}"
    if min_tgs > 0 and 0 < tg < min_tgs:
        return f"tg={tg:.1f} t/s is below --min-tgs {min_tgs:.1f}"
    return None


def _left(timeout: int, deadline: float | None) -> int:
    """Per-request timeout: whatever is left of the config's deadline, capped by
    --timeout. At least 1s — a zero would mean "no timeout" to urlopen, which is
    the opposite of what an exhausted budget means."""
    if deadline is None:
        return int(timeout)
    return max(1, int(min(float(timeout), deadline - time.time())))


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


# ---------------------------------------------------------------------------
# Setup questionnaire (Q1).
#
# The cost dials exist (--levels, --ctx-size, --min-tgs, --quick) and are the
# part nobody finds. Worse, the most natural way to use them does nothing:
# narrowing one knob leaves the array sized by the widest one, so a user who
# "limited context sizes" and saw 125 runs anyway reasonably concludes the tool
# ignores them. Asking a handful of questions and DERIVING the flags is the only
# interface where the answers compose into a design by construction.
#
# Answers -> argv is a pure function so it is testable with no GPU and no
# terminal; the interactive part only fills the dataclass. The derived command
# is printed rather than silently applied — the argv IS the record of what ran,
# and a user has to be able to save, edit and re-run it.
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    """What the user says about their workload, before it becomes flags."""
    ctx: int | None = None          # the ONE context they serve at, if fixed
    levels: int = 5                 # cost dial (see --levels)
    min_tgs: float = 0.0            # slowest result worth deploying
    reps: str = "standard"          # quick | standard | full


def intent_args(intent: Intent) -> list[str]:
    """The CLI arguments an Intent implies. Pure — no I/O, no hardware."""
    args: list[str] = []
    if intent.ctx:
        args += ["--ctx-size", str(int(intent.ctx))]
    if intent.levels != 5:
        args += ["--levels", str(int(intent.levels))]
    if intent.min_tgs > 0:
        args += ["--min-tgs", f"{intent.min_tgs:g}"]
    if intent.reps == "quick":
        args.append("--quick")
    elif intent.reps == "full":
        args.append("--full")
    return args


def _ask(prompt: str, default: str = "") -> str:
    """One line of input, or the default. Empty answer keeps the default."""
    suffix = f" [{default}]" if default else ""
    try:
        got = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise _Abort() from None
    return got or default


class _Abort(Exception):
    """User pressed Ctrl-C / Ctrl-D at a prompt."""


def _ask_int(prompt: str, default: str = "") -> int | None:
    while True:
        got = _ask(prompt, default)
        if got.lower() in ("", "no", "none", "any", "-"):
            return None
        try:
            return int(got.replace("_", "").replace(",", ""))
        except ValueError:
            print("    (a whole number, or blank to skip)")


def _ask_float(prompt: str, default: str = "") -> float:
    while True:
        got = _ask(prompt, default)
        if got.lower() in ("", "no", "none", "-"):
            return 0.0
        try:
            return max(0.0, float(got))
        except ValueError:
            print("    (a number, or blank to skip)")


def ask_intent(n_ctx_train: int | None) -> Intent:
    """Interview the user. Raises _Abort if they bail out."""
    it = Intent()
    print()
    print("Setup — four questions, then it prints the command it would run.")
    print("(Enter accepts the default; Ctrl-C to skip and use defaults.)")
    print()
    native = f" (this model trains to {n_ctx_train})" if n_ctx_train else ""
    print(f"1. Context{native}. If you always serve at one size, give it —")
    print("   pinning context is the single biggest saving, and tuning at a")
    print("   size you never use is the most common way to waste a sweep.")
    it.ctx = _ask_int("context you serve at, blank to sweep a range", "")
    print()
    print("2. Speed floor. Configs below this are abandoned rather than waited")
    print("   out — the main reason sweeps take all night.")
    it.min_tgs = _ask_float("slowest generation t/s worth deploying, blank for none", "")
    print()
    print("3. How much of the space to search.")
    print("   coarse = 3 levels/knob (~27 runs), full = 5 (~125 runs).")
    coarse = _ask("coarse or full", "coarse").lower().startswith("c")
    it.levels = 3 if coarse else 5
    print()
    print("4. Repeats per config. More reps = less noise, proportionally longer.")
    it.reps = _ask("quick / standard / full", "standard").lower()
    if it.reps not in ("quick", "standard", "full"):
        it.reps = "standard"
    return it


def offer_to_run(argv: list[str]) -> bool:
    """Print the derived command and ask whether to execute it."""
    print()
    print("Derived command:")
    print()
    print("  python3 " + " ".join([Path(__file__).name] + argv))
    print()
    try:
        return _ask("run it now? (y/N)", "N").lower().startswith("y")
    except _Abort:
        return False


# Flags that mean the user already knows what they want. If any appears, the
# interview would be overriding a decision rather than helping make one.
_INTENT_FLAGS = ("--run", "--report-only", "--diff", "--selftest", "--factor",
                 "--levels", "--ctx-size", "-c", "--min-context", "--max-context",
                 "--ctx-floor", "--max-depth", "--min-tgs", "--min-pps", "--quick",
                 "--full", "--screen", "--iterate", "--use-case", "--profile",
                 "--array", "--ctx-scan", "--probe-ctx", "--merge-results")


def interview_wanted(args, argv: list[str]) -> "Intent | None":
    """Run the setup interview and return the Intent, or None to stay as-is.

    Deliberately conservative. A redirected stdout is a script, and a script
    that blocks on input() is a hang, not a prompt — so the non-TTY path keeps
    the old plan-only behaviour exactly."""
    if not args.model or not sys.stdout.isatty():
        return None
    if any(a == f or a.startswith(f + "=") for a in argv for f in _INTENT_FLAGS):
        return None
    try:
        return ask_intent(None)
    except _Abort:
        print("\n(skipped — using defaults)")
        return None


def apply_intent(ap, args, intent: "Intent | None"):
    """Re-parse argv with the interview's answers appended, so the derived flags
    go through the same validation as typed ones — no second code path."""
    if intent is None:
        return args
    return ap.parse_args([str(args.model)] + intent_args(intent))


def with_ticker(prefix: str, timeout: int, fn):
    """Run fn() showing a live elapsed ticker on a TTY; plain start line if the
    output is redirected (keeps logs one-line-in/one-line-out)."""
    if not sys.stdout.isatty():
        print("trying " + prefix + " ...", flush=True)
        return fn()
    stop = threading.Event()
    t0 = time.time()

    def tick():
        while not stop.wait(1.0):
            sys.stdout.write(f"\rtrying {prefix} ... {fmt_dur(time.time() - t0)} "
                             f"(timeout {fmt_dur(timeout)})   ")
            sys.stdout.flush()

    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=0.2)
        sys.stdout.write("\r" + " " * (len(prefix) + 56) + "\r")  # clear line
        sys.stdout.flush()


def run_with_progress(cfg: Config, f: dict, timeout: int, prefix: str):
    return with_ticker(prefix, timeout, lambda: drive_one(cfg, f, timeout))


def run_one(cfg: Config, f: dict, timeout: int):
    cmd = bench_command(cfg, f)
    t0 = time.time()
    status, pp, tg = "OK", 0.0, 0.0
    sampler = VRAMSampler().__enter__() if cfg.measure_vram else None
    backend = ""
    # llama-bench is one opaque process: nothing can be watched mid-run, so a
    # throughput floor has to act as a shorter deadline. A config that would
    # MEET the floor finishes inside it, so hitting it is itself the finding.
    budget = slow_budget_secs(cfg, timeout)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=budget,
                              env=run_env(cfg, f))
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or _OOM_PAT.search(combined):
            sig = died_on_signal(proc.returncode)
            status = ("OOM" if _OOM_PAT.search(combined) else
                      "SIGNAL" if sig else "ERROR")
        else:
            pp_, tg_ = parse_bench_json(proc.stdout)
            backend = parse_bench_backend(proc.stdout)
            if tg_ is None:
                status = "PARSE_FAIL"
            else:
                pp, tg = pp_ or 0.0, tg_
    except subprocess.TimeoutExpired:
        # Distinguish "we stopped waiting because it is too slow to want" from
        # "it hung": only the first is a deliberate answer to a question asked.
        status = "SLOW" if budget < timeout else "TIMEOUT"
    finally:
        if sampler:
            sampler.__exit__()
    res = {"status": status, "pp_tps": pp, "tg_tps": tg, "backend": backend,
           "secs": time.time() - t0, "vram_mib": sampler.peak if sampler else 0}
    if status == "SLOW":
        res["too_slow"] = (f"did not finish within {budget}s, the budget implied "
                           f"by --min-tgs {cfg.min_tgs:.1f}")
    elif status == "OK":
        why = too_slow_reason(cfg, pp, tg)   # finished, but under a floor
        if why:
            res.update(status="SLOW", too_slow=why)
    return validate_measurement(res)


# ---------------------------------------------------------------------------
# Server driver: launch llama-server and drive real generation (measures MTP /
# speculative decoding and real concurrency, which llama-bench cannot).
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_server_args(cfg: Config, f: dict, port: int, n_ctx: int) -> list[str]:
    ub = int(f.get("ubatch", 512))
    args = [
        str(cfg.llama_server), "-m", str(cfg.model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(n_ctx),                          # mmap on is the server default
    ]
    # A draft model is an INPUT, not a factor (D1): a sweep either has one or it
    # does not. Its presence is also what switches llama.cpp's speculation from
    # the target's own head to a separate model — has_dft() is simply "was a -md
    # path given" (common/common.h) — so the same MTP head can be driven either
    # way, and only this makes the second route expressible (issue #12).
    # A supplied draft model is loaded only when this row asks for it: with a
    # `spec_type` column the level decides, and loading a head the row is not
    # measuring would both cost VRAM and let llama.cpp infer a type the row did
    # not choose (issue #19).
    if cfg.draft_model and "spec_type" not in f:
        args += draft_model_args(cfg)
    # A multimodal projector is a third resident artifact and the same kind of
    # input: it occupies VRAM from load, whether or not any image ever arrives.
    if cfg.mmproj:
        args += ["--mmproj", str(cfg.mmproj)]
    # --fit is on by default and adjusts unset arguments at load time to make
    # the model fit — silently changing the very configuration being measured,
    # so a row's factors and what actually ran can disagree. Turn it off, but
    # only where the binary has it: the flag is recent, and older builds reject
    # unknown arguments outright. Spelled `--fit off`; there is no `--no-fit`.
    if supports_flag(cfg.llama_server, "--fit"):
        args += ["--fit", "off"]
    if "load_mode" not in f:                       # fixed unless swept (R4)
        args += load_mode_args(cfg.llama_server, "server")
    if "fa" not in f:                              # flash-attn fixed unless swept
        args += ["-fa", str(FIXED_FA)]
    if "batch_ratio" not in f:
        args += ["-b", str(max(FIXED_BATCH, ub))]
    args += _flat(factor_flags(cfg, f, "server"))
    if "spec_type" in f:                          # swept: level decides the flags
        args += spec_type_args(cfg, f["spec_type"])
    if "concurrency" in f:                        # swept: level decides the flags
        args += concurrency_flags(f["concurrency"])
    elif "parallel" not in f and cfg.parallel > 1:  # concurrency (fixed) if not swept
        args += ["--parallel", str(cfg.parallel)]
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        # ...unless a draft model was supplied: llama.cpp reads its type off the
        # draft, and forcing draft-mtp here ran MTP instead of what was loaded
        if ("mtp" not in f and "spec_type" not in f
                and not draft_decides_spec_type(cfg)):
            args += ["--spec-type", "draft-mtp"]
        # default n_max if not swept (and not already pinned for a derived
        # sibling); nothing to bound when mtp is explicitly off
        if (f.get("mtp") != "0" and "spec_n_max" not in f
                and "spec_n_max" not in derived_base_pins(f)):
            args += ["--spec-draft-n-max", str(cfg.spec_draft_n_max)]
    # ngram self-speculation: enable ngram-mod by default when ngram is on but not
    # swept. ngram's draft length is ngram-mod's own default, not --spec-draft-n-max
    # (a draft-model/MTP knob with no effect on ngram).
    if cfg.ngram and "ngram" not in f:
        args += ["--spec-type", "ngram-mod"]
    # Merge duplicate --spec-type entries (MTP + ngram) into one comma-sep value
    args = _merge_spec_type_args(args)
    return args


def _wait_health(port: int, deadline: float, proc=None) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server process exited (died during load) — stop waiting
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# Varied prose so speculative-decoding acceptance is realistic. A repeated single
# token is trivially predictable and would inflate the measured MTP speedup.
_CORPUS = (
    "The history of computing spans centuries, from mechanical calculators to "
    "modern processors running billions of operations per second. In distributed "
    "systems, consensus algorithms such as Raft and Paxos let unreliable machines "
    "agree on a single value despite failures. Photosynthesis converts sunlight, "
    "water, and carbon dioxide into glucose and oxygen. The novel opens in a quiet "
    "coastal town where the protagonist, returning after many years, confronts an "
    "old rival and a buried secret. Interest-rate decisions ripple through global "
    "markets as traders reprice risk and adjust their portfolios accordingly. A "
    "recipe for bread needs flour, water, salt, and time for the dough to rise. "
)


# ---------------------------------------------------------------------------
# Prompt battery and prefix reuse — docs/workload-shape-design.md.
#
# Category mix follows the reasoning in field-reports F6: real traffic is
# dominated by cheap requests but conditioned at high percentiles by expensive
# ones, so an even mix of shapes is not neutral either. Weights are theirs.
#
# NOTE ON SCOPE: only the prompt TEXT varies by category here, not the output
# length. Their battery gives each category its own token target (32..384),
# which we cannot do without breaking this tool's measurement contract — a
# single tg number per config is only meaningful if every rep generated the same
# amount. Varying output length per category is a real change to what tg means
# and is left as a decision, not smuggled in. See the design doc's checklist.
# ---------------------------------------------------------------------------
PROMPT_BANKS = (
    ("simple_qa", 0.40,
     "What is the capital of Portugal? How many minutes are in a day? Name the "
     "largest ocean. Who wrote the play about the prince of Denmark? What does "
     "a barometer measure? When did the first powered flight take place? "),
    ("reasoning", 0.20,
     "Suppose a train leaves at noon travelling west while a second departs an "
     "hour later travelling east; reason step by step about when the distance "
     "between them exceeds four hundred kilometres, stating each assumption you "
     "make about speed, rest stops, and the curvature you are ignoring. "),
    ("code", 0.20,
     "def merge_intervals(spans): spans.sort(key=lambda s: s[0]); out = [] ; "
     "for start, end in spans: if out and start <= out[-1][1]: out[-1][1] = "
     "max(out[-1][1], end) else: out.append([start, end]) ; return out  # fix "
     "the off-by-one when spans touch but do not overlap, and add a test. "),
    ("rag", 0.15,
     "CONTEXT: The quarterly filing notes that supply-chain costs rose 12% "
     "year over year, driven mainly by shipping rates and a one-off inventory "
     "write-down in the second half. Management guidance was left unchanged. "
     "QUESTION: what drove the increase, and did guidance change? "),
    ("long_ctx", 0.05,
     "The history of computing spans centuries, from mechanical calculators to "
     "modern processors running billions of operations per second. In distributed "
     "systems, consensus algorithms such as Raft and Paxos let unreliable machines "
     "agree on a single value despite failures. Photosynthesis converts sunlight, "
     "water, and carbon dioxide into glucose and oxygen. The novel opens in a quiet "
     "coastal town where the protagonist, returning after many years, confronts an "
     "old rival and a buried secret. "),
)

# Bootstrap only. The real ratio is model- and language-specific — code and CJK
# are nowhere near 4 — and it decides the actual n_depth of every server
# measurement, so it is measured rather than assumed as soon as llama.cpp tells
# us (docs/constants-audit.md C-B).
CHARS_PER_TOKEN = 4


def calibrate_chars_per_token(prompt: str, prompt_n) -> float | None:
    """chars-per-token for THIS model, from a response llama.cpp already sent.

    Every completion reports `prompt_n` — the tokens the prompt actually became.
    Divided by the characters we sent, that is a measured ratio for this
    tokenizer, replacing a constant that is only right for English prose.

    Returns None when the sample cannot support a ratio: an absent or zero
    `prompt_n`, or a value so far from any real tokenizer (outside 1..20
    chars/token) that it is more likely a malformed response than a surprising
    model. Rejecting a wrong ratio matters more than adopting a slightly better
    one — it sizes every subsequent prompt."""
    try:
        n = int(prompt_n or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0 or not prompt:
        return None
    ratio = len(prompt) / n
    return ratio if 1.0 <= ratio <= 20.0 else None


# A fixed corpus cannot fill a long prompt without repeating itself, and n-gram
# speculation feeds on that repetition. The literal banks above total ~1.5k
# characters; the `agents` profile asks for 32k-character prompts, so every one
# of them would contain the whole corpus about twenty times over. Measured
# consequence: acceptance pinned near 1.00 even at 0% prefix reuse.
#
# So sentences are COMBINATORIAL rather than literal. These fragments generate
# ~14k distinct sentences from a few lines of source, which is enough that a
# 32k-character prompt draws each at most once (docs/workload-shape-design.md).
_FRAGMENTS = {
    "subj": ["the compiler", "a distributed cache", "the harbour authority",
             "our field biologist", "the second violinist", "a tidal turbine",
             "the archive clerk", "this alloy", "the referee", "a monsoon front",
             "the pension fund", "her thesis committee"],
    "verb": ["reconsidered", "quietly abandoned", "measured", "argued against",
             "reproduced", "underestimated", "catalogued", "escalated",
             "deferred", "audited"],
    "obj": ["the third revision", "an inherited assumption", "eleven anomalies",
            "the winter schedule", "a contested boundary", "its own baseline",
            "the fallback path", "two conflicting reports", "the settlement terms",
            "a rounding convention", "the spare capacity", "last season's data"],
    "tail": ["before the deadline", "without telling anyone", "under protest",
             "for reasons never minuted", "on the strength of one sample",
             "after the storm passed", "despite the cost", "in a footnote",
             "over three consecutive quarters", "at considerable expense"],
}


def _sentences() -> list:
    """A large pool of distinct sentences, generated rather than transcribed.

    Combinatorial so that a long prompt does not have to repeat itself: the
    product of the fragment lists is ~14k sentences, against the ~360 a 32k-char
    prompt consumes. Deterministic order — callers shuffle."""
    out = []
    for a in _FRAGMENTS["subj"]:
        for b in _FRAGMENTS["verb"]:
            for c in _FRAGMENTS["obj"]:
                for d in _FRAGMENTS["tail"]:
                    out.append(f"{a} {b} {c} {d}. ")
    return out


def repeated_fraction(text: str, n: int = 8) -> float:
    """Fraction of `text`'s n-grams (word-level) that are not the first of their
    kind — i.e. how self-similar it is.

    This is the property that decides whether a prompt can honestly measure
    speculative decoding, and it is measurable, so it is measured rather than
    reasoned about. A prompt built by tiling a corpus scores near 1.0; genuinely
    varied prose scores near 0."""
    words = text.split()
    if len(words) <= n:
        return 0.0
    seen, repeats, total = set(), 0, 0
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i:i + n])
        total += 1
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return round(repeats / total, 4) if total else 0.0


def _fill(n_chars: int, rnd) -> str:
    """Varied prose of ~n_chars, composed from shuffled sentences.

    NOT `text * n`: repeating a short passage to length makes the prompt
    massively self-similar, and n-gram speculation feeds on exactly that. Built
    that way, a battery of *distinct* prompts still drafted at 100% acceptance —
    the requests differed from each other while each one was internally
    repetitive, so the contamination F4 set out to remove survived in a new
    place. Measured: acceptance stayed 1.00 at every reuse level until this was
    fixed (docs/workload-shape-design.md).

    Sentences are drawn in a shuffled order and the order is reshuffled each time
    the pool is exhausted, so repetition only appears at a long period rather
    than every few hundred characters."""
    pool = _sentences()
    out, n = [], 0
    while n < n_chars:
        order = list(range(len(pool)))
        rnd.shuffle(order)
        for i in order:
            out.append(pool[i])
            n += len(pool[i])
            if n >= n_chars:
                break
    return "".join(out)[:max(0, n_chars)]


def prompt_battery(n_tokens: int, count: int, reuse: float = 1.0,
                   seed: int = 42, chars_per_token: float | None = None) -> list:
    """`count` prompts of ~n_tokens tokens, sharing the first `reuse` fraction.

    `reuse` is the workload's SHAPE, not a tunable: an agent stack with a fixed
    system prompt has ~90% shared prefix whether or not that is convenient, and
    the right cache settings for it are the ones that win at its reuse level
    (docs/workload-shape-design.md). 1.0 reproduces the historical behaviour —
    every request byte-identical.

    The shared part is shared *content*, not merely a shared length: llama.cpp
    matches on tokens, so generating a fresh prefix per request would reuse
    nothing while looking like it should (W-D3). Suffixes are drawn from the
    category banks so the varying part is varied prose rather than a counter,
    which matters because speculation feeds on how predictable text is."""
    n_chars = int(max(1, n_tokens) * (chars_per_token or CHARS_PER_TOKEN))
    reuse = min(1.0, max(0.0, reuse))
    n_shared = int(n_chars * reuse)
    rnd = random.Random(seed)
    # the shared part is generated once, from its own stream, so it is identical
    # across requests no matter how many suffixes follow it
    shared = _fill(n_shared, random.Random(seed))
    out = []
    for i in range(max(1, count)):
        if n_shared >= n_chars:
            out.append(shared[:n_chars])       # fully shared: identical requests
            continue
        # a per-request stream: suffixes differ from each other AND are varied
        # prose internally, which is the part that took a measurement to get right
        out.append(shared + _fill(n_chars - n_shared, random.Random(seed + 1 + i)))
    return out


def achieved_reuse(prompts: list) -> float:
    """The prefix fraction the battery ACTUALLY shares, measured not assumed.

    Requested reuse and delivered reuse can differ — short prompts round, and a
    bank shorter than the suffix repeats. The Bayesian autotuner this idea came
    from ships `duplication_report()` for the same reason: prefix-cache
    contamination is worth quantifying rather than declaring
    (docs/field-reports.md F6). Reported alongside the result so a number can be
    read in the light of the traffic that produced it."""
    if not prompts:
        return 0.0
    if len(prompts) == 1:
        return 1.0
    shortest = min(len(p) for p in prompts)
    first = prompts[0]
    i = 0
    while i < shortest and all(p[i] == first[i] for p in prompts):
        i += 1
    mean_len = sum(len(p) for p in prompts) / len(prompts)
    return round(i / mean_len, 4) if mean_len else 0.0


def _realistic_prompt(n_tokens: int) -> str:
    """A varied-text prompt of roughly n_tokens tokens (~4 chars/token).

    Shares `_fill`'s composition rather than repeating a fixed corpus: the old
    version tiled `_CORPUS` to length, which made every long prompt strongly
    self-similar and inflated speculative acceptance regardless of what the
    battery did across requests."""
    return _fill(max(1, n_tokens) * CHARS_PER_TOKEN, random.Random(0))


# Slack on the wall-clock bound (I5). What is left of the wall after prefill is
# credited is still not pure decode: it carries the HTTP round trip, the JSON
# transfer, and server-side tokenization — which for a 160 KB prompt is not free,
# and which `prompt_ms` does not cover, since llama.cpp starts that clock AFTER
# tokenizing (server-context.cpp). Rejecting a real measurement is the worse
# error (P3), so the margin is deliberately loose: the defect it exists to catch
# overshoots by four orders of magnitude, not by tens of percent.
WALL_CLOCK_MARGIN = 2.0

# The most of a request's wall clock that may be credited to prefill. The credit
# comes from the server's own `timings.prompt_ms` — the same clock I5 exists to
# distrust — so it is capped rather than trusted: a server whose counters are
# broken can loosen its own ceiling by at most 10x, and issue #3's 1e6 t/s row
# overshot by ~1500x. Uncapped, a response claiming `prompt_ms == wall` drives
# the denominator to zero and switches the check off against exactly the fault it
# was written for.
WALL_PREFILL_CREDIT_MAX = 0.9


def decode_wall(wall: float, prefill_s: float) -> float:
    """The part of a request's elapsed time that could have been decode.

    Bounding a decode-only rate by a whole-request wall only holds while the
    request IS decode. It stopped being that when profiles gained a shared-prefix
    fraction (`4ffa97a`): at reuse < 1.0 — and whenever the prompt cache misses
    for any other reason — every rep re-prefills inside the request I5 is timing,
    and the prefill can be the larger part of it (issue #11: 17.6s of a 23.5s
    request, which read as "4x faster than the wall clock permits")."""
    if wall <= 0:
        return 0.0
    return wall - min(max(prefill_s, 0.0), WALL_PREFILL_CREDIT_MAX * wall)


@dataclass(frozen=True)
class RepClock:
    """One rep's clocks: what we timed from outside, and how much of it the
    server says it spent on prefill rather than decode."""
    n_dec: int
    wall: float
    prefill_s: float = 0.0

    @property
    def ceiling(self) -> float:
        """The fastest decode rate this rep's elapsed time can support."""
        d = decode_wall(self.wall, self.prefill_s)
        return self.n_dec / d if d > 0 else 0.0


def exceeds_wall_clock(tg: float, reps: list) -> str | None:
    """Why the server's self-reported rate is impossible, or None.

    Compares a server-reported tg against what its own request duration allows.
    Independent of any hardware assumption: it asks only whether the tokens
    could have been produced in the time the request actually took."""
    usable = [r for r in reps if r.n_dec > 0 and r.ceiling > 0]
    if tg <= 0 or not usable:
        return None
    best = max(usable, key=lambda r: r.ceiling)   # kindest rep to the server
    if tg <= best.ceiling * WALL_CLOCK_MARGIN:
        return None
    # The breakdown travels with the verdict. A rejected row is zeroed, so the
    # reason line is the only surviving evidence of what was measured — and
    # issue #11 took five round trips because it did not say which term was
    # large (docs/measurement-validity.md).
    credited = min(max(best.prefill_s, 0.0), WALL_PREFILL_CREDIT_MAX * best.wall)
    spent = (f"{best.wall:.1f}s wall − {credited:.1f}s prefill"
             if credited > 0 else f"{best.wall:.1f}s wall")
    return (f"server reported tg={tg:.1f} t/s, but the request's own "
            f"duration allows at most {best.ceiling:.1f} t/s "
            f"({tg / best.ceiling:.0f}x faster than the wall clock permits; "
            f"{best.n_dec} tokens in {spent})")


@dataclass(frozen=True)
class RepSample:
    """One rep's reported decode rate together with the clock that can bound it.

    `clock` is None when the rep returned nothing the bound can be built from
    (no wall, no decoded tokens); such a rep is unvalidatable, not innocent, so
    it is kept but can never be the evidence that clears the others."""
    tg: float
    clock: RepClock | None = None


def screen_reps(samples: list) -> tuple[list, list]:
    """Split reps into those their OWN request duration can support, and those
    it cannot. Returns (kept, dropped).

    Checking each rep against its own clock rather than an aggregate against the
    kindest one is the difference between rejecting a measurement and rejecting
    a rep. Issue #11 ended on a config where one rep came back at 333,362 t/s
    and the others at ~45: the mean carried the outlier into the verdict, the
    whole row was zeroed, and the surviving reason could not say whether one rep
    had gone bad or all three had. A broken counter on one request is a fault in
    that request, and the other reps remain a measurement."""
    kept, dropped = [], []
    for s in samples:
        if s.clock is not None and exceeds_wall_clock(s.tg, [s.clock]):
            dropped.append(s)
        else:
            kept.append(s)
    return kept, dropped


def _rejected_reason(dropped: list) -> str | None:
    """The verdict for a config where no rep survived its own clock.

    Reported against the median of the rejected rates and the kindest rejected
    rep, so the sentence describes the measurement that would have been recorded
    had the check not fired."""
    clocks = [s.clock for s in dropped if s.clock is not None]
    if not clocks:
        return None
    return exceeds_wall_clock(statistics.median([s.tg for s in dropped]), clocks)


def prompt_tokens_of(timings: dict) -> int:
    """Tokens the prompt actually became: the ones the model ran plus the ones
    the cache spared it. The delivered counterpart to the requested depth."""
    try:
        return int(timings.get("prompt_n") or 0) + int(timings.get("cache_n") or 0)
    except (TypeError, ValueError):
        return 0


def cache_miss_advice(hw: dict) -> list[str]:
    """What to actually do about a prompt cache that is not hitting, given the
    architecture of the model being measured.

    Every line here is measured rather than reasoned (issue #15,
    docs/workload-shape-design.md), because the first two versions of this advice
    were wrong in opposite directions:

    - **SWA** (`gemma3`, window 512): 90% reuse at a 1,960-token prompt, 0% on the
      SECOND rep at 15,700. `--swa-full` restores it to 90%. `--ctx-checkpoints`
      at 0/32/128/512 and `--cache-ram` all change nothing.
    - **Hybrid/recurrent** (`qwen35moe`): 87% at 4,002 tokens, 43% at 8,004, 0% at
      16,352 — and 0% at both `-ctxcp 32` and `-ctxcp 512`. So reuse is not
      unavailable on these models, as this advice once claimed; it is available
      until some depth and then it is not, and no flag moves the boundary.

    The common variable is DEPTH, in both classes. The architecture only decides
    whether a knob exists."""
    out = ["Reps are re-prefilling instead of decoding off a cache hit, so",
           "this measures a colder workload than the one being tuned for."]
    if int(hw.get("n_swa") or 0) > 0:
        out += [
            "This model uses sliding-window attention, and past the window a",
            "shared prefix stops being reusable. `--swa-full` restores it, at the",
            "cost of the sliding-window shortcut in attention -- which is why",
            "`swa_full` is swept rather than switched on. Compare the two levels",
            "in the results rather than assuming either way.",
        ]
    elif int(hw.get("ssm_state") or 0) > 0:
        out += [
            "This model has recurrent (SSM) memory, which cannot be rolled back",
            "to an arbitrary prefix past a certain depth. Measured on qwen35moe:",
            "87% reuse at 4k tokens, 43% at 8k, 0% at 16k -- and unchanged by",
            "--ctx-checkpoints. No flag moves that boundary, so the only lever is",
            "depth: a smaller --n-depth measures the workload you asked for.",
        ]
    else:
        out += [
            "Most likely the context cannot hold prompt + generation, so the slot",
            "is reset between requests: try a smaller --n-depth.",
        ]
    return out


def delivered_cache_hit(timings: list) -> float | None:
    """Fraction of prompt tokens the server actually served from its cache.

    The DELIVERED counterpart to `reuse`, which records only what the battery
    ASKED for. llama.cpp matches on tokens and can decline to reuse a prefix the
    client believes it shares — a context that will not hold prompt plus
    generation, a slot reset, an eviction — and when it does, a rep the tool
    believes is pure decode pays a full prefill instead. Nothing in the row could
    tell those apart, which is most of why issue #11 stayed open.

    `prompt_n` is the tokens llama.cpp ran through the model and `cache_n` the
    ones it skipped, so the two sum to the prompt. None when the server reports
    no `cache_n` at all — an unknown, not a zero."""
    reused = processed = 0
    seen = False
    for t in timings:
        if "cache_n" not in t:
            continue
        seen = True
        reused += int(t.get("cache_n") or 0)
        processed += int(t.get("prompt_n") or 0)
    if not seen:
        return None
    total = reused + processed
    return round(reused / total, 4) if total else 0.0


def raise_for_server_error(body):
    """Pass a completion response through, or raise if it is really a failure.

    llama-server answers some failures with HTTP 200 and an "error" object.
    Treating that as a measurement is how a non-result acquires timings of zero
    and turns into an infinite rate (P1)."""
    if isinstance(body, dict) and body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise OSError(f"llama-server returned an error: {msg}")
    return body


def _completion(port: int, prompt, n_gen: int, timeout: int, cache: bool = False) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_gen,
        "temperature": 0,
        "cache_prompt": cache,
        "timings_per_token": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return raise_for_server_error(json.loads(r.read()))


def _measure_round(port: int, prompts, n_gen: int, par: int, timeout: int, cache=False):
    """One round of `par` concurrent completions.

    Returns (responses, n_failed, wall_s) — responses only for requests that
    SUCCEEDED. A round used to die on the first exception (`ex.map` re-raises
    while iterating), which collapsed "one of eight requests failed" into the
    same ERROR as "the model would not load". Those are different configurations
    and the report should be able to tell them apart, so failures are counted
    rather than fatal (docs/measurement-validity.md).

    `prompts` is a list of one prompt per slot, so a round can issue distinct
    requests; a bare string is broadcast to every slot."""
    if isinstance(prompts, str):
        prompts = [prompts] * par
    ok, failed = [], 0

    def one(i):
        return _completion(port, prompts[i % len(prompts)], n_gen, timeout, cache)

    with ThreadPoolExecutor(max_workers=par) as ex:
        w0 = time.time()
        futures = [ex.submit(one, i) for i in range(par)]
        for fut in futures:
            try:
                ok.append(fut.result())
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                failed += 1
        return ok, failed, time.time() - w0


# ---------------------------------------------------------------------------
# Speculative telemetry (docs/field-reports.md, F1). Acceptance is the number
# that explains a speculative result, and llama.cpp hands it to us for free in
# the same `timings` block we already read for the throughput rate.
# ---------------------------------------------------------------------------
def _draft_totals(timings: list) -> dict:
    """Sum llama.cpp's draft counters across a set of `timings` blocks.

    llama.cpp emits `draft_n`/`draft_n_accepted` only when a draft actually ran
    (tools/server/server-common.cpp), so responses carrying neither are positive
    evidence that speculation did not happen — not merely a gap in the telemetry.
    Returns {} in that case; `draft_cols` turns it into the `spec_off` flag."""
    drafted = accepted = generated = 0
    ran = False
    for t in timings:
        if not isinstance(t, dict) or "draft_n" not in t:
            continue
        ran = True
        drafted += int(t.get("draft_n") or 0)
        accepted += int(t.get("draft_n_accepted") or 0)
        generated += int(t.get("predicted_n") or 0)
    return ({"drafted": drafted, "accepted": accepted, "generated": generated}
            if ran else {})


def speculation_requested(cfg, f: dict) -> bool:
    """Whether this run asks llama.cpp for speculative decoding at all.

    Both gates ride `--spec-type`: `mtp` selects draft-mtp and `ngram` selects a
    self-speculative variant, and each spells "off" as a level that translates to
    an omitted flag (FACTORS). But a gate is not always a *factor* — when it is
    not swept, `build_server_args` emits it fixed-on (MTP whenever the model has
    a NextN head and --no-mtp was not given; ngram-mod whenever --ngram is on).
    Reading only the assignment would miss exactly those runs, which are the
    common case."""
    if "spec_type" in f:
        # the level IS the answer here: `none` asked for nothing, and every
        # other level names a head. Checked first, because with this column
        # present `mtp` is absent and the fixed-on branch below would flag every
        # row -- including the `none` rows -- as speculation that failed to run.
        if str(f["spec_type"]) != "none":
            return True
    elif "mtp" in f:
        if str(f["mtp"]) == "1":
            return True
    elif cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        return True                        # fixed on, not swept
    if "ngram" in f:
        if str(f["ngram"]) not in ("none", "", "0"):
            return True
    elif cfg.ngram:
        return True                        # fixed on (ngram-mod), not swept
    return False


def spec_cols_wanted(cfg) -> bool:
    """Whether the results CSV should carry the speculative telemetry columns.

    Only the server driver can speculate (every speculative knob is server_only),
    and only a run that can actually produce a draft has anything to record.
    Elsewhere these would be blank in every row — the inert-column problem the
    factor design already rejects (docs/multi-gpu-design.md, M1)."""
    if cfg.driver != "server":
        return False
    return bool((cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0) or cfg.ngram
                or "mtp" in cfg.factors or "ngram" in cfg.factors)


def draft_cols(cfg, f: dict, draft: dict) -> dict:
    """Speculative telemetry columns for one measurement.

    `draft_acc` is the fraction of DRAFTED tokens llama.cpp accepted: draft
    *quality*. `draft_cov` is the fraction of GENERATED tokens that came from an
    accepted draft: how much speculation actually contributed.

    Both are needed, because acc alone is misleading in the exact case that
    matters. Measured on ngram-mod: a config drafting 59 tokens and accepting all
    59 scores acc=1.00 and ran at 579 t/s, while one drafting 124 and accepting
    104 scores acc=0.84 and ran at 846. Acceptance ranked them backwards; cov
    (0.46 vs 0.81) tracks throughput. A drafter that is always right about the
    few tokens it dares to guess is not helping much.

    `spec_off` is the guard: the row asked for speculation and no draft ever ran.
    That is the issue #8 shape — a row recording `mtp=1` that silently measured
    the baseline and poisoned the `mtp` main effect. `spec_n_min_frac` and
    `ngram_mod_n_max_off` now make the inverted assignment unconstructible
    (docs/CONSTRAINED-FACTORS.md); this is the independent check that the
    construction held, in the same spirit as the wall-clock ceiling over the
    server's self-reported rate (docs/measurement-validity.md).

    It is a flag, never a status: the measurement is real and correctly scored,
    it simply is not measuring what its factor column claims."""
    if draft:
        drafted = draft.get("drafted", 0)
        generated = draft.get("generated", 0)
        acc = draft.get("accepted", 0) / drafted if drafted else 0.0
        cov = draft.get("accepted", 0) / generated if generated else 0.0
        return {"draft_acc": round(acc, 4), "draft_cov": round(cov, 4)}
    return {"spec_off": 1} if speculation_requested(cfg, f) else {}


class ServerSession:
    """A running llama-server, reusable across runs that share load-time params
    (only the request — prompt length via n_depth — varies). Launch once, issue
    many measurements, close once."""

    # class-level so it survives sessions constructed without __init__, and so
    # the cache-miss warning is said once rather than once per config
    _cache_warned = False

    def __init__(self, cfg: Config, launch_f: dict, n_ctx: int, timeout: int):
        self.cfg = cfg
        self.ok = False
        self.err = ""
        self.port = _free_port()
        args = build_server_args(cfg, launch_f, self.port, n_ctx)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True,
                                     env=run_env(cfg, launch_f))
        # Wait for the server to come up, but give up fast if the process dies
        # (crashed on load) or exceeds the startup budget — don't hang for the
        # whole per-run timeout.
        start_deadline = time.time() + cfg.server_start_timeout
        if _wait_health(self.port, start_deadline, self.proc):
            self.ok = True
        else:
            died = self.proc.poll() is not None
            self.proc.terminate()
            try:
                _, err = self.proc.communicate(timeout=10)
                self.err = err or ""
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.signal = died_on_signal(self.proc.returncode) if died else None
            if not self.err:
                self.err = (f"server died on signal {self.signal} during load"
                            if self.signal else
                            "server exited during load" if died else
                            f"server not healthy within {cfg.server_start_timeout}s")

    # Characters in the calibration probe. Big enough that the ratio is not
    # dominated by the first few tokens, small enough that prefilling it costs
    # nothing next to the measurement it is protecting.
    CALIBRATION_CHARS = 4096

    def calibrate(self, timeout):
        """Measure this model's chars-per-token BEFORE any prompt is sized.

        `CHARS_PER_TOKEN = 4` is a bootstrap for English prose; real tokenizers
        wander a long way from it. Measured on the reporter's model in issue #11
        (`qwen35`, via `llama-tokenize`): 32,000 characters of the battery's own
        prose became 5,290 tokens — **6.05** chars/token. Every server prompt was
        therefore built two thirds as deep as it claimed, and a row labelled
        `n_depth=32768` had measured about 27,000 tokens of context.

        The ratio was already being measured — from the warm request — but the
        prompts were built before that response existed, so on a single-config
        run the calibration never sized anything, and in a sweep only the first
        config in each session ran uncalibrated. That is worse than a constant
        error: it makes one row per session measure a different depth from its
        siblings, in a design that randomizes execution order specifically so
        that per-row differences do not correlate with anything.

        One short request, at session open, costs a prefill of a few hundred
        tokens and makes the depth honest for every config that follows."""
        if getattr(self, "_calibrated", False):
            return getattr(self, "cpt", None)
        self._calibrated = True   # one probe per session, hit or miss
        probe = _fill(self.CALIBRATION_CHARS, random.Random(7))
        try:
            r = _completion(self.port, probe, 1, timeout)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None      # not fatal: the bootstrap constant still applies
        self.cpt = calibrate_chars_per_token(
            probe, r.get("timings", {}).get("prompt_n"))
        return self.cpt

    def measure(self, prompt_len, n_gen, par, reps, timeout, deadline=None):
        """Measure one config. Returns a dict:

            pp, tg      throughput (prefill / decode)
            problem     why the numbers cannot be believed, else None
            draft       llama.cpp's speculative counters, {} if no draft ran
            err_rate    fraction of requests that failed (0.0 when all succeeded)

        A dict rather than a tuple because this has grown three times and each
        caller wants a different subset.

        `err_rate` exists because a failure used to be fatal to the whole
        measurement: one bad request out of eight raised, and the config was
        recorded as ERROR — indistinguishable from a model that would not load.
        A config that serves 7 of 8 requests quickly is a real and different
        thing, and the sweep should be able to see it (F7)."""
        # One prompt per request rather than one prompt reused: at reuse 1.0 the
        # battery reproduces the historical identical-prompt behaviour exactly,
        # and below it the requests genuinely differ (W-D1/W-D3).
        reuse = float(getattr(self.cfg, "prefix_reuse", 1.0))
        n_req = (max(1, reps) + 1) if par == 1 else par
        # cpt is measured once per session, by `calibrate`, BEFORE the first
        # prompt is sized — a model whose tokenizer is nothing like 4 chars/token
        # then gets prompts of the requested SIZE from the very first config
        # rather than from the second (docs/constants-audit.md C-B, issue #11).
        self.calibrate(_left(timeout, deadline))
        prompts = prompt_battery(prompt_len, n_req, reuse,
                                 chars_per_token=getattr(self, "cpt", None))
        self.last_reuse = achieved_reuse(prompts)
        prompt = prompts[0]
        n_err = 0
        if par == 1:
            # Single stream: the warm request prefills (real pp + primes the KV
            # cache), then each rep sends the NEXT prompt in the battery. At
            # reuse 1.0 that is the same prompt, so the reps are pure decode off
            # a full cache hit; below 1.0 they deliberately are not, and the rep
            # re-prefills the differing suffix. `wall` below therefore covers
            # prefill as well as decode at any reuse < 1.0 — see issue #11.
            prompt_tok = 0
            try:
                warm = _completion(self.port, prompt, n_gen,
                                   _left(timeout, deadline), cache=True)
                wt = warm.get("timings", {})
                pp = wt.get("prompt_per_second", 0.0) or 0.0
                # What the prompt ACTUALLY became. `n_depth` records what was
                # asked for, and the two are not the same number whenever the
                # tokenizer is not 4 chars/token — the row should carry the one
                # that was measured (issue #11).
                prompt_tok = prompt_tokens_of(wt)
                # Only as a fallback for a session whose probe failed: once a
                # ratio is in hand it is held for the whole session, so two
                # configs measured against one server are measured at the same
                # depth. Re-deriving it here is what used to make the first
                # config in every session the odd one out (issue #11).
                self.cpt = (getattr(self, "cpt", None)
                            or calibrate_chars_per_token(prompt,
                                                         wt.get("prompt_n")))
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pp, n_err = 0.0, n_err + 1
            samples, timings = [], []
            # Prefill the rep cannot avoid, priced client-side, for servers that
            # report no `prompt_ms`. Deliberately a LOWER bound on the credit:
            # `pp` is measured on the warm request's uncontended full prefill, so
            # this under-credits rather than over-credits and the ceiling stays
            # tighter than the truth. That direction costs some P3 headroom, but
            # it cannot weaken P1 — an estimate that can only under-credit can
            # never talk the check out of a rejection it should make.
            prefill_est = ((1.0 - self.last_reuse) * prompt_len / pp
                             if pp > 0 else 0.0)
            # A prefill floor is answerable the moment the warm request returns,
            # before any decode rep is paid for — the cheapest possible exit.
            _min_pps = float(getattr(self.cfg, "min_pps", 0.0) or 0.0)
            if _min_pps > 0 and 0 < pp < _min_pps:
                return {"pp": pp, "tg": 0.0, "problem": None, "draft": {},
                        "err_rate": 0.0, "reuse": self.last_reuse,
                        "prompt_tok": prompt_tok,
                        "too_slow": (f"pp={pp:.1f} t/s is below --min-pps "
                                     f"{_min_pps:.1f}; decode not measured")}
            for i in range(max(1, reps)):
                if _expired(deadline):     # budget gone; report what we have
                    break
                w0 = time.time()
                try:
                    r = _completion(self.port, prompts[(i + 1) % len(prompts)],
                                    n_gen, _left(timeout, deadline), cache=True)
                except (urllib.error.URLError, OSError, json.JSONDecodeError):
                    n_err += 1
                    continue
                wall = time.time() - w0
                t = r.get("timings", {})
                timings.append(t)   # draft counters live here too (F1)
                rate = t.get("predicted_per_second", 0.0) or 0.0
                # Independent clock. The server derives its rate as
                # 1e3 / t_token_generation * n_decoded from its own counter
                # (server_slot stats, `n_gen_tps()`); when that counter is wrong nothing
                # inside the response can reveal it. Our wall time is the one
                # number the server does not supply, and the request cannot have
                # produced tokens faster than it elapsed — so predicted_n over the
                # DECODE part of that wall is a hard upper bound on any rate it
                # may claim. Only the decode part: the rest of the request is
                # prefill, which the server did not claim that rate for and which
                # at reuse < 1.0 is most of the wall (issue #11).
                n_dec = t.get("predicted_n", 0) or 0
                clock = None
                if wall > 0 and n_dec > 0:
                    try:
                        reported = float(t.get("prompt_ms") or 0.0)
                    except (TypeError, ValueError):
                        reported = 0.0     # unparseable is not a credit
                    prefill = reported / 1000.0 if reported > 0 else prefill_est
                    clock = RepClock(n_dec, wall, prefill)
                samples.append(RepSample(rate, clock))
            # Each rep is judged against its own request's duration, then the
            # survivors decide the number. One rep with a broken counter is a
            # broken rep, not a broken configuration (issue #11).
            kept, dropped = screen_reps(samples)
            # Median, not mean: the aggregate should not move with the worst rep
            # it contains — the same reasoning --verify-picks already applies.
            tg = statistics.median([s.tg for s in kept]) if kept else 0.0
            # the warm request is excluded from the reps (it primes the cache and
            # is not a measured rep) but IS counted in err_rate: a config that
            # cannot serve the first request has failed a request.
            hit = delivered_cache_hit(timings)
            self._warn_cache_miss(hit)
            self._warn_rejected_reps(kept, dropped)
            return {"pp": pp, "tg": tg,
                    # Only when NOTHING survived is the configuration itself
                    # unmeasurable; the reason is then built from the reps that
                    # failed, so the row still says what was rejected.
                    "problem": (None if kept else _rejected_reason(dropped)),
                    "rejected_reps": len(dropped),
                    "prompt_tok": prompt_tok,
                    "draft": _draft_totals(timings),
                    "err_rate": n_err / (max(1, reps) + 1),
                    "reuse": self.last_reuse,
                    "cache_hit": hit}
        # Concurrency: realistic serving — every request prefills; aggregate over
        # the streams. Warmup once, then average per-round throughput over reps.
        # These rates are computed from our own wall clock already, so they
        # cannot exceed it by construction — no cross-check needed.
        warm_res, _, _ = _measure_round(self.port, prompts, n_gen, par,
                                        _left(timeout, deadline))
        if warm_res:                            # warmup discarded, ratio kept
            self.cpt = (getattr(self, "cpt", None) or calibrate_chars_per_token(
                prompts[0], warm_res[0].get("timings", {}).get("prompt_n")))
        pps, tps, timings = [], [], []
        n_sent = 0
        for _ in range(max(1, reps)):
            if _expired(deadline):
                break
            res, failed, wall = _measure_round(self.port, prompts, n_gen, par,
                                               _left(timeout, deadline))
            n_err += failed
            n_sent += par
            timings.extend(r.get("timings", {}) for r in res)
            tg_tok = sum(r.get("timings", {}).get("predicted_n", 0) for r in res)
            pp_tok = sum(r.get("timings", {}).get("prompt_n", 0) for r in res)
            # Throughput stays whole-round: the wall clock covers the failures
            # too, so a config that drops requests does not get to look faster
            # for having done less work.
            tps.append(tg_tok / wall if wall > 0 else 0.0)
            pps.append(pp_tok / wall if wall > 0 else 0.0)
        return {"pp": sum(pps) / len(pps), "tg": sum(tps) / len(tps),
                "problem": None, "draft": _draft_totals(timings),
                "err_rate": n_err / n_sent if n_sent else 0.0,
                "reuse": self.last_reuse,
                "prompt_tok": (prompt_tokens_of(warm_res[0].get("timings", {}))
                               if warm_res else 0),
                "cache_hit": delivered_cache_hit(timings)}

    def _warn_cache_miss(self, hit):
        """Say so once when the server delivers far less prefix reuse than the
        battery asked for.

        Not an IMPLAUSIBLE: the numbers are real, they just answer a different
        question than the requested shape. A run whose reps were meant to be
        cache hits and were not measured a colder workload than the one being
        tuned for, and every derived `secs` and time budget is off with it."""
        want = float(getattr(self.cfg, "prefix_reuse", 1.0) or 0.0)
        if hit is None or want < 0.5 or hit >= want / 2 or self._cache_warned:
            return
        self._cache_warned = True
        print(f"\n!! prompt cache delivered {hit * 100:.0f}% reuse where the "
              f"workload asked for {want * 100:.0f}%.")
        print("!! Reps are re-prefilling instead of decoding off a cache hit, so")
        print("!! this measures a colder workload than the one being tuned for.")
        for line in cache_miss_advice(getattr(self.cfg, "hw", {}) or {}):
            print(f"!! {line}")
        print()

    def _warn_cache_miss_lines(self):
        return cache_miss_advice(getattr(self.cfg, "hw", {}) or {})

    def _warn_rejected_reps(self, kept, dropped):
        """Say so when some — but not all — reps failed their own wall clock.

        A dropped rep is not free to hide: the row keeps its numbers because the
        survivors are a real measurement, which is exactly the situation where a
        silently discarded sample would be invisible."""
        if not dropped or not kept:
            return          # nothing dropped, or the row is IMPLAUSIBLE anyway
        worst = max(dropped, key=lambda s: s.tg)
        print(f"\n!! {len(dropped)} of {len(kept) + len(dropped)} reps reported a "
              f"decode rate their own request duration cannot support "
              f"(worst: {worst.tg:.1f} t/s).")
        print("!! Those reps were dropped; the result is the median of the rest.\n")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def load_key(cfg: Config, f: dict):
    """Server launch identity: every factor except the request-time n_depth."""
    return tuple((k, f.get(k)) for k in cfg.factors if k != "n_depth")


def load_key_str(cfg: Config, f: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in load_key(cfg, f))


# ---------------------------------------------------------------------------
# Crash journal: a config that hard-hangs or reboots the machine writes no
# result. We record "about to try X" and fsync it to disk BEFORE launching, so
# on restart a started-but-never-finished config is a suspected killer and is
# skipped instead of retried (which would reboot again). Two risk phases:
# "load" (server model load into VRAM, per group) and "run" (a measurement).
# ---------------------------------------------------------------------------
def journal_write(jh, *fields):
    jh.write("\t".join(str(x) for x in fields) + "\n")
    jh.flush()
    os.fsync(jh.fileno())  # durable before the risky operation begins


def read_journal(path: Path):
    """Return (tried_load{key:cfg}, ok_load{keys}, tried_run{run_id:cfg})."""
    tried_load, ok_load, tried_run = {}, set(), {}
    if not path.exists():
        return tried_load, ok_load, tried_run
    for line in path.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0] == "TRY" and p[1] == "load":
            try:
                tried_load[p[2]] = json.loads(p[3]) if len(p) > 3 else {}
            except json.JSONDecodeError:
                tried_load[p[2]] = {}
        elif len(p) >= 3 and p[0] == "OK" and p[1] == "load":
            ok_load.add(p[2])
        elif len(p) >= 3 and p[0] == "TRY" and p[1] == "run":
            try:
                tried_run[p[2]] = json.loads(p[3]) if len(p) > 3 else {}
            except json.JSONDecodeError:
                tried_run[p[2]] = {}
    return tried_load, ok_load, tried_run


def measure_in_session(cfg: Config, f: dict, session, timeout: int) -> dict:
    """Measure one config against an (already-launched) server session."""
    t0 = time.time()
    if session is None or not session.ok:
        err = session.err if session else ""
        status = ("OOM" if _OOM_PAT.search(err or "") else
                  "SIGNAL" if getattr(session, "signal", None) else "ERROR")
        return {"status": status, "pp_tps": 0.0, "tg_tps": 0.0, "secs": 0.0,
                "vram_mib": 0}
    prompt_len = cfg.n_prompt + int(f.get("n_depth", 0))
    par = slots_for(cfg, f)
    sampler = VRAMSampler().__enter__() if cfg.measure_vram else None
    status, m = "OK", {}
    try:
        # One deadline for the whole config — launch is already behind us, so
        # this covers the warm request and every rep together (T1).
        budget = slow_budget_secs(cfg, timeout)
        m = session.measure(prompt_len, cfg.n_gen, par, cfg.reps, timeout,
                            deadline=time.time() + budget)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        status = "ERROR"
    finally:
        if sampler:
            sampler.__exit__()
    pp, tg = m.get("pp", 0.0), m.get("tg", 0.0)
    err_rate = float(m.get("err_rate", 0.0))
    # Every request failed: there is no measurement here, whatever the timers
    # say. A partial failure is a different thing and stays OK, penalised.
    if status == "OK" and err_rate >= 1.0:
        status = "ERROR"
    res = {"status": status, "pp_tps": pp, "tg_tps": tg,
           "err_rate": round(err_rate, 4),
           # what the battery ACTUALLY shared, not what was asked for (F6)
           "reuse": m.get("reuse", ""),
           # and what the SERVER actually reused of it, which is a different
           # number and the one that decides whether a rep was decode (#11)
           "cache_hit": "" if m.get("cache_hit") is None else m["cache_hit"],
           # the depth that was MEASURED, not the depth that was asked for: the
           # prompt is sized in characters, so `n_depth` is only the request
           # until a tokenizer has had its say (#11)
           "prompt_tok": m.get("prompt_tok", "") or "",
           # reps whose own request duration could not support the rate they
           # reported, and were therefore left out of tg_tps
           "rejected_reps": int(m.get("rejected_reps", 0) or 0),
           # derivable from the emitted flags, so recorded rather than left to
           # be re-derived by whoever reads the CSV later (KV2)
           "kv_unified": 1 if kv_unified_for(cfg, f) else 0,
           "secs": time.time() - t0, "vram_mib": sampler.peak if sampler else 0}
    if status == "OK":                     # speculative telemetry (F1) — a
        res.update(draft_cols(cfg, f, m.get("draft", {})))  # never sets status
    if status == "OK" and m.get("problem"):   # the wall-clock cross-check (I5)
        # The numbers are zeroed because they cannot be true, but the reason
        # keeps a record of what was rejected. A false reject is otherwise
        # unreviewable from the CSV alone — the evidence for its own appeal is
        # exactly what the rejection destroys (issue #11).
        res.update(status="IMPLAUSIBLE",
                   implausible=f"{m['problem']}; measured pp={pp:.1f} t/s",
                   pp_tps=0.0, tg_tps=0.0)
    # Below a floor the user set. A real measurement they have said they do not
    # want, so it keeps its numbers and is merely not OK — unlike IMPLAUSIBLE,
    # which zeroes them because they cannot be true (T1).
    slow = m.get("too_slow") or (too_slow_reason(cfg, pp, tg)
                                 if status == "OK" else None)
    if status == "OK" and slow:
        res.update(status="SLOW", too_slow=slow)
    # parallel scales the ceiling: the server driver reports throughput
    # aggregated across concurrent streams (I2).
    return validate_measurement(res, parallel=par)


def server_run_one(cfg: Config, f: dict, timeout: int):
    """Standalone server measurement for one config (own session). Used by the
    context probe; the sweep groups configs to reuse sessions instead."""
    par = slots_for(cfg, f)
    prompt_len = cfg.n_prompt + int(f.get("n_depth", 0))
    # K3: a slot gets the full n_ctx under unified KV and n_ctx/slots when split,
    # so only the split regime asks for slots x the per-slot context.
    n_ctx = (prompt_len + cfg.n_gen + 256) * ctx_slots_multiplier(
        f.get("concurrency", str(par) if par > 1 else "1"))
    session = ServerSession(cfg, f, n_ctx, timeout)
    try:
        return measure_in_session(cfg, f, session, timeout)
    finally:
        session.close()


def drive_one(cfg: Config, f: dict, timeout: int):
    """Dispatch to the configured driver (standalone, one process per run)."""
    if cfg.driver == "server":
        return server_run_one(cfg, f, timeout)
    return run_one(cfg, f, timeout)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def score_of(r: dict) -> float:
    """Primary objective score for a run (effective throughput if present)."""
    return float(r.get("eff_tps", r.get("tg_tps", 0.0)))


def measured_ok(rows: list[dict]) -> list[dict]:
    """Rows that both succeeded AND produced a number worth acting on.

    `status == "OK"` alone is not enough, and the gap is reachable: a run that
    completes but generates nothing keeps status OK, because
    `implausible_reason` deliberately passes on `tg <= 0` ("not a measurement;
    other paths own it"). This is that other path.

    It matters wherever a row is SELECTED rather than counted. `longest` keys on
    depth first, so a zero-throughput row at the deepest depth wins outright and
    gets recommended as the max-context config — a command that loads and then
    produces nothing. Same lineage as issue #3: "OK" only ever meant the process
    exited cleanly."""
    return [r for r in rows if r.get("status") == "OK" and score_of(r) > 0]


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Non-dominated set maximizing (context depth, objective score) among OK."""
    ok = measured_ok(rows)
    frontier = []
    for r in ok:
        depth, s = int(r["n_depth"]), score_of(r)
        dominated = any(
            int(o["n_depth"]) >= depth and score_of(o) >= s and o is not r
            and (int(o["n_depth"]) > depth or score_of(o) > s)
            for o in ok
        )
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda r: int(r["n_depth"]))


def verified_depth_of(cfg: Config, rows: list[dict], r: dict) -> int:
    """Deepest n_depth among OK rows sharing r's launch factors (everything but
    n_depth) — the most context this exact config is *known* to load. With the
    server driver those siblings shared one session sized at the group's max
    depth; with bench they each loaded at least their own depth."""
    def key(row):
        return tuple(str(row.get(k)) for k in cfg.factors if k != "n_depth")
    mine = key(r)
    ds = [int(o["n_depth"]) for o in rows
          if o.get("status") == "OK" and key(o) == mine]
    return max(ds, default=int(r["n_depth"]))


def recommended_ctx(cfg: Config, r: dict, verified_depth: int | None = None,
                    rows: list[dict] | None = None) -> int:
    """Context (`-c`) to emit for a winning row.

    Base: the footprint the sweep actually verified for this row — the server
    driver sizes its session at ``n_prompt + n_depth + n_gen + 256`` (times
    ``--parallel``) in ``server_run_one`` — rounded *down* to a tidy multiple;
    rounding up would inflate the KV cache past the verified point and can OOM
    at launch.

    Floor: a server with a tiny context isn't worth pasting (a depth-0 winner
    would emit ``-c 1024``), so the result is raised to at least ``ctx_floor``
    per slot — but never past the footprint of ``verified_depth`` (the deepest
    this launch config measured OK; see ``verified_depth_of``). The floor rides
    on evidence, it never outruns it: with no deeper sibling this returns the
    row's own verified footprint unchanged.
    """
    par = max(1, int(r.get("parallel", cfg.parallel)))

    def footprint(d: int) -> int:
        v = cfg.n_prompt + d + cfg.n_gen + 256
        return v * par if par > 1 else v

    d = int(r["n_depth"])
    want = max(footprint(d), cfg.ctx_floor * par)
    cap = footprint(d if verified_depth is None else max(d, verified_depth))
    # A row's depth is what was ASKED for. The server driver builds its prompt in
    # characters, so when the tokenizer is not 4 chars/token the prompt that ran
    # was a different size — and `-c` must ride on the context the run actually
    # occupied, never on the one it was named after. Issue #11: a row labelled
    # n_depth=32768 measured about 27,000 tokens, and this emitted -c 41472 on
    # the strength of it. Taken across the same siblings `verified_depth` rides
    # on, since it is their evidence the cap is spending.
    if rows is not None:
        measured = verified_footprint(cfg, rows, r, par)
        if measured:
            cap = min(cap, measured)
    return max(256, (min(want, cap) // 256) * 256)


def verified_footprint(cfg: Config, rows: list[dict], r: dict, par: int) -> int:
    """Largest context an OK sibling of `r` is known to have actually occupied.

    Falls back, per sibling, to the footprint its requested depth implies — rows
    written before `prompt_tok` existed, and bench rows, are unknowns and must
    not tighten anything. 0 when there is no evidence at all, which leaves the
    caller's cap alone."""
    def key(row):
        return tuple(str(row.get(k)) for k in cfg.factors if k != "n_depth")

    mine, best = key(r), 0
    for o in rows:
        if o.get("status") != "OK" or key(o) != mine:
            continue
        v = cfg.n_prompt + int(o["n_depth"]) + cfg.n_gen + 256
        best = max(best, measured_footprint(cfg, o, par)
                   or (v * par if par > 1 else v))
    return best


def measured_footprint(cfg: Config, r: dict, par: int) -> int:
    """Context the row's own request actually occupied, or 0 when unrecorded.

    `prompt_tok` is the delivered prompt; the generation and the same 256-token
    margin the session sized itself with sit on top of it. Rows written before
    the column existed, and every bench row, return 0 — an unknown, which leaves
    the recommendation exactly where it was."""
    try:
        tok = int(float(r.get("prompt_tok") or 0))
    except (TypeError, ValueError):
        return 0
    if tok <= 0:
        return 0
    v = tok + cfg.n_gen + 256
    return v * par if par > 1 else v


def ctx_floor_note(cfg: Config, r: dict, ctx: int) -> str | None:
    """One-line note when the emitted -c had to stay below the usable floor
    because the sweep holds no deeper evidence for this config."""
    par = max(1, int(r.get("parallel", cfg.parallel)))
    if ctx >= (cfg.ctx_floor * par) // 256 * 256:
        return None
    return (f"note: -c {ctx} is below the {cfg.ctx_floor} usable-context floor — "
            "the sweep never verified this config any deeper.")


def pick_recommendations(cfg: Config, rows: list[dict]):
    """The report's three picks: (fastest, balanced, longest).

    FASTEST means fastest *usable* — best score among configs whose emitted
    command can hold the usable-context floor (verified evidence, not hope) —
    so the headline is never a config that only shines with an empty KV cache.
    Falls back to the raw fastest when nothing meets the floor (e.g. a model
    whose native context is below it); the floor note flags that. BALANCED is
    the best score *measured at* depth >= the floor (speed while actually deep
    in context); LONGEST is the deepest OK row.
    """
    ok = measured_ok(rows)      # never recommend a config that produced nothing
    if not ok:
        return None, None, None

    def holds_floor(r):
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r), rows)
        return ctx_floor_note(cfg, r, ctx) is None

    pool = [r for r in ok if holds_floor(r)]
    fastest = max(pool or ok, key=score_of)
    deep = [r for r in ok if int(r["n_depth"]) >= cfg.ctx_floor]
    balanced = max(deep, key=score_of) if deep else None
    longest = max(ok, key=lambda r: (int(r["n_depth"]), score_of(r)))
    return fastest, balanced, longest


def kv_downgrade_hint(r: dict) -> str | None:
    """One-line `q8_0` suggestion for rows whose KV cache is heavier than the
    near-lossless `q8_0` floor (i.e. `f32`/`f16`/`bf16`). q8_0 ~halves the KV
    footprint per token for roughly double the context in the same VRAM, so it
    is the cheapest lever back from a memory cliff — surfaced at copy-paste time
    rather than discovered by an OOM. Returns None when nothing is to be gained
    (KV already q8_0 or lossier)."""
    kv = str(r.get("kv_type", ""))
    if kv not in KV_QUALITY or KV_QUALITY.index(kv) >= KV_QUALITY.index("q8_0"):
        return None
    return ("tip: KV cache is " + kv + " — set -ctk q8_0 -ctv q8_0 to ~halve KV "
            "memory (near-lossless) for roughly double the context / more OOM "
            "headroom, at a small decode-speed cost.")


def error_note(r: dict) -> str | None:
    """One-line warning when a config dropped requests, or None.

    Throughput already carries the cost — a round's wall clock covers the failed
    requests too, so a config that serves less scores less, with no extra penalty
    needed. What the number cannot say is *why* it is low. A config that is fast
    but drops one request in eight and a config that is simply slower are the
    same eff_tps and very different things to deploy, so the distinction is
    stated where the command gets copied."""
    try:
        rate = float(r.get("err_rate") or 0.0)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    return (f"warning: {rate * 100:.0f}% of requests FAILED at this config — the "
            "throughput above already reflects the lost work, but a config that "
            "drops requests under load is usually not one to deploy. Re-run it "
            "with --verify-picks before trusting it.")


def ttft_note(r: dict, ctx: int) -> str | None:
    """One-line prefill-cost estimate for the emitted -c: filling the context
    at this config's measured pp speed. A giant -c can mean many minutes before
    the first token — great tg alone doesn't make a config interactive — so the
    cost is stated right where the command gets copied. Prefill slows somewhat
    with depth, so treat the estimate as optimistic."""
    try:
        pp = float(r.get("pp_tps") or 0)
    except (TypeError, ValueError):
        return None
    if pp <= 0 or ctx <= 0:
        return None
    note = (f"prefill cost: a full {ctx}-token prompt ≈ {fmt_dur(ctx / pp)} "
            f"to first token at pp={pp:.0f} t/s")
    if ctx > 16384:
        note += f" (an 8k prompt ≈ {fmt_dur(8192 / pp)})"
    return note


def show_discards(rows: list[dict]) -> None:
    """Say which rows were thrown away and why, wherever a report is printed.

    The sweep says it as each row is rejected (P4: never discard quietly), but a
    report rebuilt from a CSV said nothing at all — so the one command a reporter
    can run without a GPU could not answer the one question a false reject
    raises. It can now: the reason travels in the row (issue #11)."""
    gone = [r for r in rows if r.get("implausible")]
    if not gone:
        return
    print(f"\n### DISCARDED as impossible ({len(gone)} of {len(rows)})")
    for r in gone:
        print(f"  run {r.get('run_id', '?')}: {r['implausible']}")


def report(cfg: Config, rows: list[dict], probe: dict | None = None):
    ok = [r for r in rows if r["status"] == "OK"]
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(ok)}/{len(rows)} configs succeeded")
    print("=" * 70)
    if not ok:
        bad = {}
        for r in rows:
            bad[r["status"]] = bad.get(r["status"], 0) + 1
        print("No successful runs. Status breakdown:", bad)
        _dead = show_dead_levels(cfg, rows)
        if _dead:
            print("\nOUT OF BOUNDS — levels where nothing ever ran:")
            print("\n".join(_dead))
        show_discards(rows)
        return

    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        if cfg.driver == "server":
            print("NOTE: model has an MTP head — these numbers INCLUDE the "
                  "draft-mtp speculative-decoding speedup (server driver).")
        else:
            print("NOTE: model has an MTP head — the server commands enable "
                  "draft-mtp, but the measured t/s below does NOT include that "
                  "boost (bench can't do spec decoding). Use --driver server to "
                  "measure it.")

    if cfg.score == "eff":
        print(f"objective  : effective t/s for a {cfg.profile} request "
              f"({cfg.n_prompt} prompt + {cfg.n_gen} gen tokens)")
    else:
        print("objective  : generation t/s (decode only; pp reported but not "
              "scored — --score eff to blend prefill in)")

    fastest, balanced, longest = pick_recommendations(cfg, rows)

    def show(title, r):
        if not r:
            print(f"\n### {title}: none met the constraint")
            return
        print(f"\n### {title}")
        raw = ((f"tg={r['tg_tps']:.1f}  " if cfg.score == "eff" else "")
               + f"pp={r['pp_tps']:.1f}")
        print(f"  {cfg.score}={score_of(r):.1f} t/s  ({raw})  "
              f"depth={r['n_depth']}  ngl={r['ngl']}  "
              f"t={r['threads']}  kv={r['kv_type']}  ub={r['ubatch']}")
        if r.get("verify_n"):
            print(f"  verified: median of {r['verify_n']} measurements "
                  f"(spread {r['spread_pct']:.0f}%)")
        # size context to what the sweep verified for this config, floored at
        # the usable floor where evidence allows (see recommended_ctx); a
        # bigger -c can OOM at launch.
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r), rows)
        print("  suggested llama-server command:")
        print("    " + server_command(cfg, r, ctx))
        for extra in (error_note(r), kv_downgrade_hint(r),
                      ctx_floor_note(cfg, r, ctx), ttft_note(r, ctx)):
            if extra:
                print("  " + extra)

    show("FASTEST (max speed, usable context)", fastest)
    show(f"BALANCED (best with context >= {cfg.ctx_floor})", balanced)
    show("MAX CONTEXT", longest)

    if probe:
        r, depth, tps = probe["row"], probe["depth"], probe["tg_tps"]
        kind = ("the model's native limit" if probe["at_cap"]
                else "an OOM boundary")
        print("\n### PROBED CEILING (largest -c that loads — beyond the "
              "swept range)")
        print(f"  ~{depth} tokens ({kind})"
              + (f"  tg={tps:.1f} t/s spot-check there" if tps else "")
              + f"  ngl={r['ngl']}  kv={r['kv_type']}  ub={r['ubatch']}")
        print(f"  suggested llama-server command (-c {probe['safe_ctx']}, "
              "~10% headroom under the ceiling):")
        print("    " + server_command(cfg, r, probe["safe_ctx"]))
        for extra in (error_note(r), kv_downgrade_hint(r),
                      ttft_note(r, probe["safe_ctx"])):
            if extra:
                print("  " + extra)

    kind = "effective" if cfg.score == "eff" else "generation"
    print(f"\n### Pareto frontier (context vs {kind} t/s)")
    for r in pareto_frontier(rows):
        raw = f"(tg={r['tg_tps']:5.1f})  " if cfg.score == "eff" else ""
        print(f"  depth={int(r['n_depth']):>6}  {cfg.score}={score_of(r):6.1f} t/s  "
              f"{raw}ngl={r['ngl']:>3}  "
              f"kv={r['kv_type']:>4}  ub={r['ubatch']:>4}")

    # A crash is a measurement of a boundary, not a gap in the data. Balanced
    # designs make it readable: a level that failed in EVERY one of its rows,
    # each beside different partners, is out of bounds rather than unlucky.
    dead = show_dead_levels(cfg, rows)
    if dead:
        print("\n### OUT OF BOUNDS (no run at these levels produced a number)")
        print("\n".join(dead))
        print("  These are dropped from the next --iterate pass. If you expected "
              "them to work,\n  that is the bug to chase — the sweep found the "
              "edge, it did not create it.")

    show_discards(rows)


def factor_level_means(rows: list[dict], factor: str) -> dict:
    """Mean objective score per level of a factor (OK runs only) — the Taguchi
    main effect for a balanced design. A conditional factor is scored only over
    rows where it could act (I3): rows where its gate selected a different
    variant (`active_when`) or switched the feature off entirely (`gated_by`)
    are inert and carry no information about its effect.

    Without this, a knob that did nothing is credited with an effect computed
    from run-to-run noise, in the table the user reads to decide what matters —
    which is worse than the wasted runs, because the runs are at least honest
    about the other factors (issue #16)."""
    ok = [r for r in rows
          if r["status"] == "OK" and is_active(factor, r)
          and not is_inert(factor, r)]
    means = {}
    levels = sorted(set(str(r[factor]) for r in ok if factor in r),
                    key=lambda x: (len(x), x))
    for lvl in levels:
        vals = [score_of(r) for r in ok if str(r.get(factor)) == lvl]
        if vals:
            means[lvl] = sum(vals) / len(vals)
    return means


def refine_numeric(vals: list[int], best: int) -> list[str]:
    """Finer grid of levels bracketing `best` (its neighbours in the current
    grid), for the next refinement pass."""
    vals = sorted(set(vals))
    if len(vals) <= 1:
        return [str(v) for v in vals] or [str(best)]
    i = vals.index(best) if best in vals else min(range(len(vals)),
                                                  key=lambda k: abs(vals[k] - best))
    lo = vals[i - 1] if i > 0 else vals[i]
    hi = vals[i + 1] if i < len(vals) - 1 else vals[i]
    if lo == hi:
        step = max(1, (vals[-1] - vals[0]) // len(vals))
        lo, hi = max(0, best - step), best + step
    return [str(x) for x in five_levels_span(lo, hi)]


# Statuses that mean "this configuration never produced a number", as opposed to
# producing a bad one. SLOW and IMPLAUSIBLE are excluded deliberately: SLOW is a
# real measurement below a floor the user set, and IMPLAUSIBLE is a number we
# refused — both had a working launch, so neither is evidence about bounds.
NO_RESULT_STATUS = {"SIGNAL", "OOM", "ERROR", "TIMEOUT", "CRASH"}


def dead_levels(rows: list[dict], factor: str, min_rows: int = 2) -> dict:
    """Levels of `factor` at which NOTHING ever ran, with why.

    A segfault is not noise — it says a parameter set is out of bounds, and the
    array already visited that region systematically. What makes it usable is the
    design's balance: each level appears in several rows beside *different*
    partners, so "every row at this level failed" is evidence about the level
    rather than about one unlucky combination. One crash is not; that is the
    generalisation this deliberately refuses to make, which is why `min_rows`
    exists and why a level with a single observation is left alone.

    Returns {level: status-counter}. Empty when every level produced something,
    which is the normal case."""
    out = {}
    by_level: dict = {}
    for r in rows:
        if factor not in r:
            continue
        by_level.setdefault(str(r[factor]), []).append(str(r.get("status", "")))
    for lvl, statuses in by_level.items():
        if len(statuses) < min_rows:
            continue                      # too few rows to tell level from luck
        if all(st in NO_RESULT_STATUS for st in statuses):
            tally: dict = {}
            for st in statuses:
                tally[st] = tally.get(st, 0) + 1
            out[lvl] = tally
    return out


def show_dead_levels(cfg: Config, rows: list[dict]) -> list[str]:
    """Lines naming every level that never produced a measurement, or []."""
    lines = []
    for name in cfg.factors:
        dead = dead_levels(rows, name)
        if not dead or len(dead) >= len(cfg.factors[name]):
            # every level dead means the model or the box failed, not this knob
            continue
        for lvl, tally in sorted(dead.items()):
            why = ", ".join(f"{n}x{st}" for st, n in sorted(tally.items()))
            lines.append(f"  {name}={lvl}: no run produced a number ({why})")
    return lines


def refine_factors(cfg: Config, rows: list[dict]) -> dict:
    """Produce the next pass's factor levels: settle low-impact factors at their
    winning level, and refine high-impact factors onto a finer grid around their
    best value (numeric) or their top levels (categorical)."""
    ranges, bests = {}, {}
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if not means:
            ranges[name], bests[name] = 0.0, cfg.factors[name][0]
            continue
        bests[name] = max(means, key=means.get)
        ranges[name] = max(means.values()) - min(means.values()) if len(means) > 1 else 0.0
    max_range = max(ranges.values(), default=0.0) or 1.0

    new = {}
    for name, cur in cfg.factors.items():
        # n_depth is the report's tradeoff axis (speed vs context), not a knob to
        # optimize to one value. Keep its full spread across passes so the final
        # pass still maps the whole curve — otherwise FASTEST/BALANCED/MAX-CONTEXT
        # collapse to a single depth and the three recommendations become identical.
        if name == "n_depth":
            new[name] = cur
            continue
        rng, best = ranges[name], bests[name]
        active = len(cur) > 1 and rng >= 0.25 * max_range
        kind = FACTORS.get(name, {}).get("kind", "cat")
        numeric = kind == "num" and name not in cfg.env_factor_names
        if not active:
            new[name] = [str(best)]                       # settle at the winner
        elif numeric:
            new[name] = refine_numeric([int(x) for x in cur], int(best))  # finer grid
        else:                                             # cat/float/env: keep top few
            means = factor_level_means(rows, name)
            ranked = sorted(means, key=means.get, reverse=True)
            new[name] = ranked[:3] if len(ranked) >= 3 else ranked
        # Levels that never produced a number are out of bounds, not merely bad.
        # Filtered from the OUTPUT rather than the input, because refine_numeric
        # builds a fresh grid spanning the old endpoints and would put a known
        # dead level straight back. Only the levels actually MEASURED dead are
        # dropped — the new neighbours it invented are unknowns, and the next
        # pass is what settles them. Inferring a whole dead band from one dead
        # level is the generalisation this avoids on purpose.
        dead = dead_levels(rows, name)
        if dead and len(dead) < len(cur):
            new[name] = [lv for lv in new[name] if str(lv) not in dead] or new[name]
    # A derived factor's refined grid must still honour its relation (C1). This
    # is the path a level-set-only fix could never have covered: refine_numeric
    # brackets the winner on a grid of its own, per pass, so a conflict-free pass
    # 1 can hand pass 2 an out-of-domain level. Drop the offenders rather than
    # assert — a refinement pass must not abort a sweep that is already hours in.
    for name in derived_names(new):
        kept = [l for l in new[name] if not validate_factor_levels({name: [l]})]
        dropped = [l for l in new[name] if l not in kept]
        if dropped:
            print(f"refine: dropping out-of-domain {name} level(s) {dropped} — "
                  f"{name} is relative to {FACTORS[name]['derived_from'][0]}")
        new[name] = kept or [str(bests[name])]   # the winner is in-domain by construction
    return new


def _svg_pareto(rows: list[dict], vram_total: float = 0,
                ylabel: str = "effective t/s") -> str:
    """Inline SVG: effective t/s (left y) vs context (x), all OK runs as faint dots,
    the Pareto frontier as a highlighted line. If runs carry measured VRAM, overlay
    the VRAM curve on a right-hand axis plus the physical-ceiling line. Theme-neutral."""
    ok = [r for r in rows if r["status"] == "OK" and score_of(r) > 0]
    if len(ok) < 2:
        return ""
    pts = [(int(r["n_depth"]), score_of(r)) for r in ok]
    front = [(int(r["n_depth"]), score_of(r)) for r in pareto_frontier(rows)]
    # measured VRAM per run (only if --vram was used and values are present)
    vpts = sorted((int(r["n_depth"]), float(r.get("vram_mib") or 0))
                  for r in ok if float(r.get("vram_mib") or 0) > 0)
    have_vram = len(vpts) >= 2
    W, H, ml, mt, mb = 680, 340, 62, 14, 46
    mr = 58 if have_vram else 16
    # zoom the x-axis to the data range too (same treatment as y below, clamped
    # at 0 since negative context is meaningless): fixed 0..max scaling stacked
    # every point of a single-depth run on the right edge, and rounded-up ticks
    # could land outside the viewBox.
    xs = [x for x, _ in pts]
    xpad = (max(xs) - min(xs)) * 0.1 or max(max(xs) * 0.05, 1)
    xlo, xhi = max(0, min(xs) - xpad), max(xs) + xpad
    ys = [y for _, y in pts]
    # zoom the left y-axis to 10% of the data *range* beyond each end (NOT 10% of the
    # value, and NOT from 0) so the actual variation fills ~80% of the height —
    # otherwise a high-baseline low-spread curve (e.g. the 270M model, ~450 t/s with
    # ~20 spread) still looks like a flat horizontal line.
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or max(hi * 0.05, 1)
    ylo, yhi = lo - pad, hi + pad
    # right VRAM axis: from 0 to the physical ceiling (or headroom above peak)
    vhi = max([v for _, v in vpts] + [vram_total]) * 1.05 if have_vram else 1

    def sx(x):
        return ml + ((x - xlo) / (xhi - xlo)) * (W - ml - mr)

    def sy(y):
        return H - mb - ((y - ylo) / (yhi - ylo)) * (H - mt - mb)

    def sv(v):                               # right axis -> pixels
        return H - mb - (v / vhi) * (H - mt - mb)

    g = []
    for i in range(5):                       # 5 ticks across the zoomed x-range
        x = xlo + (xhi - xlo) * i / 4
        gx = sx(x)
        g.append(f"<line x1='{gx:.0f}' y1='{mt}' x2='{gx:.0f}' y2='{H - mb}' class='grid'/>")
        g.append(f"<text x='{gx:.0f}' y='{H - mb + 16}' class='ax xt'>{int(x)}</text>")
    for i in range(5):                       # 5 ticks across the zoomed y-range
        y = ylo + (yhi - ylo) * i / 4
        gy = sy(y)
        g.append(f"<line x1='{ml}' y1='{gy:.0f}' x2='{W - mr}' y2='{gy:.0f}' class='grid'/>")
        g.append(f"<text x='{ml - 6}' y='{gy + 4:.0f}' class='ax yt'>{y:.0f}</text>")
    dots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' class='dot'/>"
                   for x, y in pts)
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in front)
    fdots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' class='fdot'/>"
                    for x, y in front)
    vram_svg = ""
    if have_vram:
        for i in range(5):                   # right-axis ticks in GiB
            v = vhi * i / 4
            g.append(f"<text x='{W - mr + 6}' y='{sv(v) + 4:.0f}' class='ax vt'>"
                     f"{v / 1024:.1f}</text>")
        vpoly = " ".join(f"{sx(x):.1f},{sv(v):.1f}" for x, v in vpts)
        vdots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sv(v):.1f}' r='3' class='vdot'/>"
                        for x, v in vpts)
        vram_svg = f"<polyline points='{vpoly}' class='vline'/>{vdots}"
        if vram_total > 0:                   # physical VRAM ceiling
            cy0 = sv(vram_total)
            vram_svg += (
                f"<line x1='{ml}' y1='{cy0:.0f}' x2='{W - mr}' y2='{cy0:.0f}' "
                f"class='ceil'/><text x='{ml + 6}' y='{cy0 - 5:.0f}' class='clbl'>"
                f"VRAM ceiling {vram_total / 1024:.0f} GiB</text>")
    cy = (mt + H - mb) / 2
    vaxis = (f"<text x='{W - 14}' y='{cy:.0f}' class='albl vaxlbl' "
             f"transform='rotate(90 {W - 14} {cy:.0f})'>VRAM used (GiB)</text>"
             if have_vram else "")
    return (
        f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' "
        f"aria-label='effective throughput and VRAM versus context'>"
        "<style>.chart .grid{stroke:#8883} .chart .ax{fill:#888;font:11px system-ui}"
        ".chart .xt{text-anchor:middle} .chart .yt{text-anchor:end} .chart .vt{text-anchor:start}"
        ".chart .dot{fill:#8887} .chart .fline{fill:none;stroke:#2ca88f;stroke-width:2.5}"
        ".chart .fdot{fill:#2ca88f} .chart .albl{fill:#888;font:12px system-ui;"
        "text-anchor:middle}"
        ".chart .vline{fill:none;stroke:#e0993e;stroke-width:2;stroke-dasharray:1}"
        ".chart .vdot{fill:#e0993e} .chart .vaxlbl{fill:#e0993e}"
        ".chart .ceil{stroke:#d1495b;stroke-width:1.5;stroke-dasharray:5 4}"
        ".chart .clbl{fill:#d1495b;font:11px system-ui}</style>"
        f"{''.join(g)}<polyline points='{poly}' class='fline'/>{dots}{fdots}{vram_svg}"
        f"<text x='{(ml + W - mr) / 2:.0f}' y='{H - 6}' class='albl'>context (tokens)</text>"
        f"<text x='16' y='{cy:.0f}' class='albl' transform='rotate(-90 16 {cy:.0f})'>"
        f"{ylabel}</text>{vaxis}</svg>")


def write_html_report(cfg: Config, rows: list[dict], path: Path,
                      probe: dict | None = None):
    import html as _html

    def esc(x):
        return _html.escape(str(x))

    ok = [r for r in rows if r["status"] == "OK"]
    best, balanced, longest = pick_recommendations(cfg, rows)
    pareto = pareto_frontier(rows)

    def card(title, r):
        if not r:
            return f"<div class=card><h3>{esc(title)}</h3><p class=muted>none met the constraint</p></div>"
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r), rows)
        cmd = esc(server_command(cfg, r, ctx))
        ver = (f"verified: median of {r['verify_n']} measurements "
               f"(spread {r['spread_pct']:.0f}%)" if r.get("verify_n") else None)
        tip = "".join(f"<div class=muted>{esc(x)}</div>"
                      for x in (ver, error_note(r), kv_downgrade_hint(r),
                                ctx_floor_note(cfg, r, ctx), ttft_note(r, ctx)) if x)
        raw = ((f"tg {r['tg_tps']:.1f} · " if cfg.score == "eff" else "")
               + f"pp {r['pp_tps']:.1f}")
        return (f"<div class=card><h3>{esc(title)}</h3>"
                f"<div class=big>{score_of(r):.1f} <span class=unit>{cfg.score} t/s</span></div>"
                f"<div class=muted>{raw} · "
                f"depth {esc(r['n_depth'])} · ngl {esc(r['ngl'])} · kv {esc(r['kv_type'])} · "
                f"ub {esc(r['ubatch'])} · t {esc(r['threads'])}</div>"
                f"<pre>{cmd}</pre>{tip}</div>")

    probe_card = ""
    if probe:
        r = probe["row"]
        cmd = esc(server_command(cfg, r, probe["safe_ctx"]))
        kind = "model's native limit" if probe["at_cap"] else "OOM boundary"
        tps = probe["tg_tps"]
        probe_card = (
            f"<div class=card><h3>Probed ceiling (beyond swept range)</h3>"
            f"<div class=big>{probe['depth']:,} <span class=unit>tokens</span></div>"
            f"<div class=muted>{kind}"
            + (f" · tg {tps:.1f} t/s spot-check" if tps else "")
            + f" · ngl {esc(r['ngl'])} · kv {esc(r['kv_type'])} · "
            f"ub {esc(r['ubatch'])} · t {esc(r['threads'])}</div>"
            f"<pre>{cmd}</pre>"
            f"<div class=muted>largest context that loads (binary search); "
            f"speed there is a single measurement, not swept</div>"
            + "".join(f"<div class=muted>{esc(x)}</div>"
                      for x in (ttft_note(r, probe["safe_ctx"]),) if x)
            + "</div>")

    # main-effects bars, factors ordered by range (impact) descending
    effects = []
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if len(means) >= 2:
            effects.append((max(means.values()) - min(means.values()), name, means))
    effects.sort(reverse=True)
    gmax = max((rng for rng, _, _ in effects), default=1) or 1
    fx_html = []
    for rng, name, means in effects:
        vmax = max(means.values()) or 1
        bars = "".join(
            f"<div class=lvl><span class=ll>{esc(l)}</span>"
            f"<span class=bar style='width:{max(2, v / vmax * 100):.0f}%'></span>"
            f"<span class=lv>{v:.1f}</span></div>"
            for l, v in means.items())
        impact = rng / gmax * 100
        fx_html.append(
            f"<div class=fx><div class=fxh><b>{esc(name)}</b>"
            f"<span class=muted>impact {rng:.1f} ({impact:.0f}%)</span></div>{bars}</div>")

    # pareto table (the score IS tg under --score tg: no separate tg column)
    blend = cfg.score == "eff"
    par_rows = "".join(
        f"<tr><td>{int(r['n_depth'])}</td><td>{score_of(r):.1f}</td>"
        + (f"<td>{r['tg_tps']:.1f}</td>" if blend else "")
        + f"<td>{esc(r['ngl'])}</td><td>{esc(r['kv_type'])}</td>"
        f"<td>{esc(r['ubatch'])}</td></tr>" for r in pareto)
    par_head = (f"<th>context</th><th>{cfg.score} t/s</th>"
                + ("<th>tg</th>" if blend else "")
                + "<th>ngl</th><th>kv</th><th>ubatch</th>")

    # all-runs table
    fcols = list(cfg.factors.keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in
                   ["run", *fcols, *(["eff"] if blend else []), "tg", "pp", "status"])
    body = ""
    for r in sorted(rows, key=lambda r: score_of(r), reverse=True):
        cells = "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in fcols)
        cls = "" if r["status"] == "OK" else " class=bad"
        body += (f"<tr{cls}><td>{esc(r.get('run_id',''))}</td>{cells}"
                 + (f"<td>{score_of(r):.1f}</td>" if blend else "")
                 + f"<td>{float(r['tg_tps']):.1f}</td>"
                 f"<td>{float(r['pp_tps']):.1f}</td><td>{esc(r['status'])}</td></tr>")

    meta = (f"{esc(cfg.model.name)} · {cfg.hw.get('n_layers','?')} layers · "
            f"{cfg.hw.get('phys')}c/{cfg.hw.get('logical')}t · "
            f"{cfg.hw.get('vram','?')} MiB VRAM · profile {esc(cfg.profile)} · "
            f"driver {esc(cfg.driver)} · array {esc(cfg.array)}")
    doc = f"""<!doctype html><meta charset=utf-8>
<title>llama-optimize — {esc(cfg.model.name)}</title>
<style>
:root{{color-scheme:light dark}}
body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px;max-width:1100px;
 margin:auto;background:Canvas;color:CanvasText}}
h1{{margin:0 0 4px}} .meta{{color:#888;margin-bottom:20px;font-size:13px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:26px}}
.card{{flex:1;min-width:280px;border:1px solid #8883;border-radius:10px;padding:14px}}
.card h3{{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#888}}
.big{{font-size:30px;font-weight:700}} .unit{{font-size:14px;color:#888;font-weight:400}}
.muted{{color:#888;font-size:13px}}
pre{{background:#8881;padding:10px;border-radius:8px;overflow-x:auto;font-size:12px;margin:10px 0 0}}
h2{{margin:26px 0 12px;font-size:16px;border-bottom:1px solid #8883;padding-bottom:6px}}
.fx{{margin-bottom:14px}} .fxh{{display:flex;justify-content:space-between;margin-bottom:4px}}
.lvl{{display:flex;align-items:center;gap:8px;margin:2px 0}}
.ll{{width:64px;text-align:right;font-size:12px;color:#888}}
.lv{{width:56px;font-size:12px;font-variant-numeric:tabular-nums}}
.bar{{height:14px;background:linear-gradient(90deg,#4a9,#6cf);border-radius:3px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid #8882;font-variant-numeric:tabular-nums}}
th{{color:#888;font-weight:600}} tr.bad{{opacity:.5}}
.chart{{max-width:100%;height:auto;display:block;margin:6px 0 18px}}
</style>
<h1>llama-optimize report</h1>
<div class=meta>{meta}<br>objective: {"effective t/s" if blend else
 "generation t/s (pp reported, not scored)"} for a {esc(cfg.profile)} request
 ({cfg.n_prompt} prompt + {cfg.n_gen} gen tokens) — {len(ok)}/{len(rows)} configs OK</div>
<div class=cards>{card('Fastest (usable)', best)}{card(f'Balanced (≥{cfg.ctx_floor})', balanced)}{card('Max context', longest)}{probe_card}</div>
<h2>What matters (main effects, by impact)</h2>{''.join(fx_html)}
<h2>Pareto frontier (context vs {"effective" if blend else "generation"} t/s)</h2>
{_svg_pareto(rows, cfg.hw.get("vram", 0) if cfg.measure_vram else 0,
             ylabel=("effective t/s" if blend else "generation t/s"))}
<table><tr>{par_head}</tr>{par_rows}</table>
<h2>All runs</h2><table><tr>{head}</tr>{body}</table>
"""
    path.write_text(doc)
    print(f"\nwrote HTML report: {path}")


def taguchi_effects(cfg: Config, exp, rows: list[dict]):
    """Main-effects on the objective. Returns (optimal_levels, predicted_score)
    or (None, None)."""
    if exp is None:
        # direct one-way sweep: no array to analyze (and nothing to confirm —
        # every level was measured directly; the rows ARE the answer)
        print("\n(direct sweep: main-effects analysis and confirmation don't "
              "apply — every level was measured directly)")
        return None, None
    try:
        prepare_taguchi_cli()
        from taguchi import Analyzer
    except Exception:
        return None, None
    # Feed EVERY run: failed configs (OOM/TIMEOUT/ERROR) carry a 0 score as a
    # penalty. The analyzer requires a complete design, and scoring failures as 0
    # is the intended "failure is data" behaviour. (Caveat: a 0 from a timeout is
    # a censored value, so trust the Pareto for the pick and use main-effects only
    # to rank which factors matter — see README.)
    results = {int(r["run_id"]): score_of(r) for r in rows}
    n_failed = sum(1 for r in rows if r["status"] != "OK")
    if len(results) < 3:
        print("\n(not enough runs for main-effects analysis)")
        return None, None
    try:
        with Analyzer(exp, metric_name="eff_tps") as an:
            an.add_results_from_dict(results)
            kind = "effective" if cfg.score == "eff" else "generation"
            print(f"\n### Taguchi main effects ({kind} t/s, higher = better)")
            if n_failed:
                print(f"(note: {n_failed} failed run(s) scored as 0 t/s)")
            print(an.summary())
            opt = an.recommend_optimal(higher_is_better=True)
            print("Predicted-optimal levels:", opt)
            predicted = None
            try:
                p = an.predict_response(opt)
                if isinstance(p, (int, float)):
                    predicted = float(p)
                elif isinstance(p, dict):
                    nums = [v for v in p.values() if isinstance(v, (int, float))]
                    predicted = float(nums[0]) if nums else None
            except Exception:
                pass
            return opt, predicted
    except Exception as e:
        print(f"\n(main-effects analysis skipped: {e})")
        return None, None


# ---------------------------------------------------------------------------
# Stage-2: max-context probe
# ---------------------------------------------------------------------------
def probe_max_context(cfg: Config, base_f: dict, timeout: int, cap: int,
                      baseline_c: float | None = None):
    """Binary-search the largest n_depth that loads (no OOM) for base_f.

    Returns (max_depth, tg_tps_at_max) or None if even depth 0 fails.
    """
    def try_depth(d):
        wait_until_cool(baseline_c)   # each attempt measures; keep it comparable
        f = dict(base_f)
        f["n_depth"] = str(d)
        r = drive_one(cfg, f, timeout)
        return r["status"] == "OK", r

    good, _ = try_depth(0)
    if not good:
        return None
    good, r = try_depth(cap)
    if good:
        return cap, r["tg_tps"]

    lo, hi, best_tps = 0, cap, None
    while hi - lo > 2048:
        mid = ((lo + hi) // 2 // 1024) * 1024
        if mid <= lo:
            break
        good, r = try_depth(mid)
        if good:
            lo, best_tps = mid, r["tg_tps"]
        else:
            hi = mid
    return lo, best_tps


def run_probe_stage(cfg: Config, all_rows: list[dict], timeout: int,
                    thermal_baseline: float | None):
    """Run the max-context probe from the deepest-reaching OK config and package
    the result for the report: dict(row, depth, tg_tps, safe_ctx, at_cap) or
    None. The probe searches BEYOND the swept depth grid (up to the model's
    native context / --max-context), so its ceiling is an extrapolation the
    sweep never benchmarked — the report labels it as such."""
    ok = measured_ok(all_rows)  # a zero-throughput row is a bad probe seed too
    if not ok:
        return None
    # Probe the config that reaches FURTHEST (max measured depth, then
    # fastest) — the memory-lightest good config — so this is the true
    # physical ceiling, not the fastest config's (which may fit less).
    base_row = max(ok, key=lambda r: (int(r["n_depth"]), score_of(r)))
    cap = cfg.hw.get("n_ctx_train") or 131072
    if cfg.max_depth:                          # --max-context caps the search
        cap = min(cap, cfg.max_depth)
    base = {k: base_row[k] for k in cfg.factors}
    print(f"\n### Max-context probe  (config: ngl={base_row.get('ngl')} "
          f"kv={base_row.get('kv_type')} ub={base_row.get('ubatch')}, "
          f"cap={cap})")
    res = probe_max_context(cfg, base, timeout, cap, thermal_baseline)
    if not res:
        print("  even depth 0 failed to load — check the config")
        return None
    depth, tps = res
    print(f"  largest context that loads: ~{depth} tokens"
          + (f"  (tg={tps:.1f} t/s there)" if tps else ""))
    # Turn the ceiling into a usable command: run just under it so there's
    # headroom for runtime allocation/fragmentation (living at the exact edge
    # risks an OOM mid-session), rounded to a tidy size.
    safe = max(4096, int(depth * 0.9) // 1024 * 1024)
    return {"row": base_row, "depth": depth, "tg_tps": tps,
            "safe_ctx": safe, "at_cap": depth >= cap}


# ---------------------------------------------------------------------------
# Stage-3: pick verification (medians beat lucky reps)
# ---------------------------------------------------------------------------
def config_key(cfg: Config, r: dict) -> str:
    """A config's identity across rows and sidecars: its factor values joined
    into one string (JSON-object-key friendly)."""
    return "|".join(f"{k}={r.get(k, '')}" for k in cfg.factors)


def verify_sidecar(results: Path) -> Path:
    """Where a sweep persists its pick-verification medians (JSON) so
    --report-only can re-apply them without a GPU."""
    return Path(str(results) + ".verify.json")


def verify_picks(cfg: Config, all_rows: list[dict], reps: int, timeout: int,
                 thermal_baseline: float | None) -> dict | None:
    """Re-measure each pick candidate `reps` extra times and settle on the
    median of all its measurements. Sweep rows are single measurements carrying
    thermal/run-to-run noise (±25% observed on hot GPUs), and picks made from
    exact scores crown whichever config got the lucky rep — medians make the
    headline numbers reproducible. Returns {config_key: {tg_tps, pp_tps, n,
    spread_pct}} for apply_verification(), or None if nothing to verify."""
    fastest, balanced, longest = pick_recommendations(cfg, all_rows)
    cands, seen = [], set()
    for r in (fastest, balanced, longest):
        if r is None:
            continue
        k = config_key(cfg, r)
        if k not in seen:
            seen.add(k)
            cands.append((k, r))
    if not cands:
        return None
    print(f"\n### Verifying picks  ({len(cands)} config(s) x {reps} extra "
          f"measurement(s); the median becomes the reported number)")
    out = {}
    for k, r in cands:
        f = {name: str(r[name]) for name in cfg.factors}
        tgs, pps = [float(r["tg_tps"])], [float(r["pp_tps"])]
        for i in range(reps):
            wait_until_cool(thermal_baseline)
            prefix = (f"verify ngl={f.get('ngl', '?')} "
                      f"depth={f.get('n_depth', '?')} [{i + 1}/{reps}]")
            res = with_ticker(prefix, timeout,
                              lambda ff=f: drive_one(cfg, ff, timeout))
            if res["status"] == "OK" and res["tg_tps"] > 0:
                tgs.append(res["tg_tps"])
                pps.append(res["pp_tps"])
            else:
                print(f"  (verify rep failed: {res['status']} — keeping the "
                      "other measurements)")
        med_tg = statistics.median(tgs)
        spread = (max(tgs) - min(tgs)) / med_tg * 100 if med_tg > 0 else 0.0
        print(f"  ngl={f.get('ngl', '?')} depth={f.get('n_depth', '?')}: "
              f"tg {float(r['tg_tps']):.1f} -> median {med_tg:.1f} t/s "
              f"({len(tgs)} measurements, spread {spread:.0f}%)")
        out[k] = {"tg_tps": med_tg, "pp_tps": statistics.median(pps),
                  "n": len(tgs), "spread_pct": round(spread, 1)}
    return out


def apply_verification(cfg: Config, rows: list[dict], verify: dict):
    """Overwrite each verified config's measured numbers with its median (and
    tag the row: verify_n / spread_pct) so the picks, Pareto, and report all
    speak the verified value. The results CSV keeps the raw single
    measurements; the medians live in the .verify.json sidecar."""
    for r in rows:
        v = verify.get(config_key(cfg, r))
        if not v:
            continue
        r["tg_tps"], r["pp_tps"] = v["tg_tps"], v["pp_tps"]
        r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
        r["verify_n"], r["spread_pct"] = v["n"], v["spread_pct"]


# ---------------------------------------------------------------------------
# Offline self-test (no GPU): exercises parsing / analysis / factor logic
# ---------------------------------------------------------------------------
def selftest() -> bool:
    try:
        # llama-bench JSON parsing (schema per llama-bench.cpp get_fields())
        sample = json.dumps([
            {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 123.4},
            {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 45.6},
        ])
        assert parse_bench_json(sample) == (123.4, 45.6)
        assert parse_bench_json("not json") == (None, None)
        # what actually ran, recorded per row: the after-the-fact companion to
        # gpu_visibility, and the only form of it that survives into the CSV
        assert parse_bench_backend(
            '[{"backends": "ROCm", "avg_ts": 1.0}]') == "ROCm"
        assert parse_bench_backend(
            '[{"backends": "CPU", "avg_ts": 1.0}]') == "CPU"
        assert parse_bench_backend("not json") == ""
        assert parse_bench_backend("[]") == ""
        assert parse_bench_backend('[{"avg_ts": 1.0}]') == ""   # older build

        # --- a crash is not an error (validated on real hardware) ---
        # Qwen3.8 on gfx906 segfaults during context init above ~64k with NO
        # allocation failure logged, so the OOM patterns do not match and it
        # would land as a generic ERROR. It is genuinely not an OOM: the same
        # model cleanly reports "cudaMalloc failed: out of memory" at 200k, well
        # above where the segfaults begin.
        assert died_on_signal(-11) == 11          # subprocess: negative
        assert died_on_signal(139) == 11          # shell: 128 + N
        assert died_on_signal(-9) == 9 and died_on_signal(137) == 9
        assert died_on_signal(0) is None          # clean exit
        assert died_on_signal(1) is None          # ran, returned an error
        assert died_on_signal(127) is None        # command not found, not a signal
        assert died_on_signal(160) is None        # out of signal range
        assert died_on_signal(None) is None and died_on_signal("x") is None
        # a real OOM message still wins over the signal classification: the
        # process may well be killed after llama.cpp reports the allocation
        # failure, and "OOM" is the more actionable of the two
        assert _OOM_PAT.search("cudaMalloc failed: out of memory")
        assert not _OOM_PAT.search("Segmentation fault")
        assert "backend" in RESULT_COLS          # bookkeeping, never a factor

        # OOM detection
        assert _OOM_PAT.search("ggml_backend_alloc failed: out of memory")
        assert _OOM_PAT.search("ROCm error: hipErrorOutOfMemory")

        # thread + depth level generation
        assert len(thread_levels(8, 16)) >= 3
        # depth_levels respects ctx_floor
        assert depth_levels(65536, floor=0)[0] == 0
        assert depth_levels(65536, floor=32768)[0] >= 32768
        assert depth_levels(65536, floor=32768)[-1] == 65536
        assert depth_levels(65536, floor=999999) == [65536]  # floor > top → single

        # factor-level generation
        assert five_levels_span(0, 64) == [0, 16, 32, 48, 64]
        assert ngl_levels(64)[0] == 0 and ngl_levels(64)[-1] == 64

        # --- spec_type: sweep WHICH head, not MTP on/off (issue #19) ---
        # The field report was "DFlash2 seems a lot better than MTP" on a model
        # that has both. `mtp` is a binary projection of a categorical axis, so
        # the sweep could not put that question. It can now.
        with tempfile.TemporaryDirectory() as _dtd:
            _dfl = Path(_dtd) / "dflash.gguf"
            _dfl.write_bytes(b"x")
            _hw_two = {"phys": 8, "logical": 16, "n_layers": 65,
                       "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1}

            def _cfg_spec(**kw):
                return Config(model=Path("q.gguf"), llama_bench=Path("b"),
                              llama_server=Path("s"), array="auto",
                              ctx_floor=8192, driver="server", emit_mtp=True,
                              hw=dict(_hw_two), **kw)

            # a real GGUF is needed to read the head's architecture, so stub the
            # reader rather than fabricate one
            _real_meta = read_gguf_metadata
            try:
                globals()["read_gguf_metadata"] = \
                    lambda _p: {"general.architecture": "dflash"}
                cfg_two = _cfg_spec(draft_model=_dfl)
                assert spec_type_levels(cfg_two) == ["none", "draft-mtp", "dflash"]
                f_two = build_factors(cfg_two)
                assert f_two["spec_type"] == ["none", "draft-mtp", "dflash"]
                # and `mtp` steps aside -- two columns cannot both own --spec-type
                assert "mtp" not in f_two, f_two

                # each level emits its own thing, and only its own thing
                def _args(lvl):
                    return build_server_args(cfg_two, {"spec_type": lvl,
                                                       "ngl": "99"}, 8080, 4096)
                _none, _mtp, _dfl_a = _args("none"), _args("draft-mtp"), _args("dflash")
                assert "--spec-type" not in _none and "-md" not in _none, _none
                assert "-md" not in _mtp, _mtp        # the head is in the target
                assert _mtp[_mtp.index("--spec-type") + 1] == "draft-mtp", _mtp
                assert "-md" in _dfl_a, _dfl_a        # ...this one is a file
                # ...and no --spec-type with it: llama.cpp reads the type off the
                # draft, and dflash vs dspark is a tensor-level distinction we
                # deliberately do not try to make
                assert "--spec-type" not in _dfl_a, _dfl_a

                # spec_off must not fire on the level that asked for nothing
                assert speculation_requested(cfg_two, {"spec_type": "none"}) is False
                assert speculation_requested(cfg_two, {"spec_type": "dflash"}) is True
                assert draft_cols(cfg_two, {"spec_type": "none"}, {}) == {}
                assert draft_cols(cfg_two, {"spec_type": "dflash"}, {}) == {"spec_off": 1}

                # `--factor spec_type=...` by hand, with no draft model: the
                # column still owns the axis, so the fixed-on MTP default must
                # not fire underneath it and turn a `none` row into draft-mtp
                _hand = build_server_args(_cfg_spec(), {"spec_type": "none",
                                                        "ngl": "99"}, 8080, 4096)
                assert "--spec-type" not in _hand, _hand

                # one head only: nothing to choose between, so the older on/off
                # factor still covers it and the design does not change shape
                cfg_one = _cfg_spec()
                assert spec_type_levels(cfg_one) == ["none", "draft-mtp"]
                f_one = build_factors(cfg_one)
                assert "spec_type" not in f_one and f_one["mtp"] == ["1", "0"]
                # a draft head on a model with NO embedded MTP head is also just
                # two levels, and keeps today's behaviour
                cfg_dm = _cfg_spec(draft_model=_dfl)
                cfg_dm.hw["n_nextn"] = 0
                assert spec_type_levels(cfg_dm) == ["none", "dflash"]
            finally:
                globals()["read_gguf_metadata"] = _real_meta

        # --- a supplied draft model picks the spec type (issue #19) ---
        # llama.cpp reads the type off the draft GGUF (arch `dflash` -> dflash or
        # dspark; an nextn.eh_proj tensor -> mtp). The tool used to emit
        # `--spec-type draft-mtp` whenever the TARGET carried an MTP head, so
        # `--draft-model dflash.gguf` loaded the DFlash2 head and then ran MTP
        # anyway -- measuring the wrong thing without saying so.
        _hw_mtp = {"phys": 8, "logical": 16, "n_layers": 65,
                   "n_ctx_train": 262144, "n_experts": 0, "n_nextn": 1}
        cfg_dm = Config(model=Path("q.gguf"), llama_bench=Path("b"),
                        llama_server=Path("s"), array="auto", ctx_floor=8192,
                        driver="server", emit_mtp=True,
                        draft_model=Path("dflash.gguf"), hw=dict(_hw_mtp))
        _real_meta2 = read_gguf_metadata
        try:
            # a head that names its own kind: llama.cpp infers, we stay quiet
            globals()["read_gguf_metadata"] = \
                lambda _p: {"general.architecture": "dflash"}
            _a_dm = build_server_args(cfg_dm, {"ngl": "99"}, 8080, 4096)
            assert "-md" in _a_dm, _a_dm                 # the draft is loaded...
            assert "--spec-type" not in _a_dm, _a_dm     # ...and decides the type
            assert draft_self_describes(Path("x")) is True
            # an MTP sidecar names itself through its KV, one layer up from the
            # nextn tensor llama.cpp actually looks for
            globals()["read_gguf_metadata"] = lambda _p: {
                "general.architecture": "qwen35", "qwen35.nextn_predict_layers": 1}
            assert draft_self_describes(Path("x")) is True
            assert "--spec-type" not in build_server_args(cfg_dm, {"ngl": "99"},
                                                          8080, 4096)

            # A PLAIN draft model -- the classic speculative setup -- names
            # nothing. llama.cpp infers an empty list, the type stays at its
            # default of `none`, and the draft is loaded, charged to VRAM and
            # never used. Nothing said so, because on a target with no MTP head
            # nothing had requested speculation to begin with.
            globals()["read_gguf_metadata"] = \
                lambda _p: {"general.architecture": "qwen3"}
            assert draft_self_describes(Path("x")) is False
            # A draft we cannot read counts as NOT self-describing, on purpose.
            # llama.cpp reads only the first split, so a sharded draft needs an
            # explicit type by its own account (common/arg.cpp) -- and naming a
            # possibly-wrong type that shows up in the pasted command beats
            # silently speculating with nothing.
            globals()["read_gguf_metadata"] = lambda _p: {}
            assert draft_self_describes(Path("x")) is False
            assert "draft-simple" in build_server_args(cfg_dm, {"ngl": "99"},
                                                       8080, 4096)
            globals()["read_gguf_metadata"] = \
                lambda _p: {"general.architecture": "qwen3"}
            _a_plain = build_server_args(cfg_dm, {"ngl": "99"}, 8080, 4096)
            assert "-md" in _a_plain, _a_plain
            assert _a_plain[_a_plain.index("--spec-type") + 1] == "draft-simple", \
                _a_plain
            # and the pasted command carries the draft too -- it printed a
            # command with no -md at all, which would not reproduce the row
            _cmd_plain = server_command(cfg_dm, {"ngl": "99"}, 4096)
            assert "-md" in _cmd_plain and "draft-simple" in _cmd_plain, _cmd_plain
        finally:
            globals()["read_gguf_metadata"] = _real_meta2
        # with no draft model the target's own MTP head still switches it on
        cfg_no = Config(model=Path("q.gguf"), llama_bench=Path("b"),
                        llama_server=Path("s"), array="auto", ctx_floor=8192,
                        driver="server", emit_mtp=True, hw=dict(_hw_mtp))
        _a_no = build_server_args(cfg_no, {"ngl": "99"}, 8080, 4096)
        assert "--spec-type" in _a_no and "draft-mtp" in _a_no, _a_no
        # and the pasted command agrees with what was run
        assert "--spec-type draft-mtp" not in server_command(cfg_dm, {"ngl": "99"}, 4096)
        assert "--spec-type draft-mtp" in server_command(cfg_no, {"ngl": "99"}, 4096)

        # --- inherited llama.cpp toggles are measured, not assumed ---
        # Each is a documented behaviour switch with no universal answer, and we
        # were taking llama.cpp's default on every run without ever asking. Free
        # in runs: an L125 holds 31 factors, so these ride along at the same 125.
        _hw_tog = {"phys": 8, "logical": 16, "n_layers": 32,
                   "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0}
        _real_supports = supports_flag
        try:
            globals()["supports_flag"] = lambda _b, _f: True
            cfg_tog = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                             llama_server=Path("s"), array="auto", ctx_floor=8192,
                             driver="server", hw=dict(_hw_tog))
            f_tog = build_factors(cfg_tog)
            for _n in ("repack", "no_op_offload", "no_host"):
                assert f_tog[_n] == ["0", "1"], (_n, f_tog.get(_n))
            # ...and they cost nothing: same array as without them
            assert choose_array(f_tog) == choose_array(
                {k: v for k, v in f_tog.items()
                 if k not in ("repack", "no_op_offload", "no_host")})
            # a build too old for a flag does not get a column that cannot be
            # emitted -- every level would be the same run
            globals()["supports_flag"] = lambda _b, _f: False
            cfg_old = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                             llama_server=Path("s"), array="auto", ctx_floor=8192,
                             driver="server", hw=dict(_hw_tog))
            f_old = build_factors(cfg_old)
            assert not ({"repack", "no_op_offload", "no_host"} & set(f_old)), f_old
            # and the bench driver has no spelling for them
            globals()["supports_flag"] = lambda _b, _f: True
            cfg_b = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           driver="bench", hw=dict(_hw_tog))
            assert "repack" not in build_factors(cfg_b)
        finally:
            globals()["supports_flag"] = _real_supports

        # --- swa_full is swept when the model has SWA (issue #15) ---
        # Measured on gemma-3-270m, 15.7k-token prompt, 90% shared prefix: the
        # SECOND rep re-prefills the whole prompt (0% cache hit) because the
        # sliding window has scrolled past the shared prefix. `--swa-full` keeps
        # every rep at 90%. `--ctx-checkpoints` at 0, 32, 128 and 512 changes
        # nothing, and neither does `--cache-ram` -- which is why this is the
        # knob and those are not.
        assert model_swa_window({"gemma3.attention.sliding_window": 512}) == 512
        assert model_swa_window({"gemma4.attention.sliding_window": 1024}) == 1024
        # a globally-attending model has no such key, and llama.cpp disables the
        # flag itself there, so every level would be the same run
        assert model_swa_window({"qwen35.block_count": 64}) == 0
        assert model_swa_window({}) == 0

        # the whole path from metadata key to swept factor, in one hop. Every
        # entry in model_hw gates a factor, so a key that silently reads 0
        # deletes a column from the design and the deletion is indistinguishable
        # from a model that lacks the feature.
        _g3 = model_hw({"gemma3.block_count": 18,
                        "gemma3.context_length": 32768,
                        "gemma3.attention.sliding_window": 512})
        assert _g3 == {"n_layers": 18, "n_experts": 0, "n_ctx_train": 32768,
                       "n_nextn": 0, "n_swa": 512, "ssm_state": 0}, _g3
        _q = model_hw({"qwen35.block_count": 64,
                       "qwen35.context_length": 262144,
                       "qwen35.nextn_predict_layers": 1,
                       "qwen35.ssm.state_size": 128})
        assert _q == {"n_layers": 64, "n_experts": 0, "n_ctx_train": 262144,
                      "n_nextn": 1, "n_swa": 0, "ssm_state": 128}, _q

        # the cache-miss advice is architecture-specific, and every line of it is
        # measured -- the first two versions were wrong in opposite directions
        _swa_adv = " ".join(cache_miss_advice({"n_swa": 512}))
        assert "--swa-full" in _swa_adv and "swept" in _swa_adv, _swa_adv
        _rec_adv = " ".join(cache_miss_advice({"ssm_state": 128}))
        assert "recurrent" in _rec_adv and "--n-depth" in _rec_adv, _rec_adv
        # it must NOT send a recurrent user to --ctx-checkpoints: measured, that
        # changes nothing at 32 or 512
        assert "ctx-checkpoints" not in _rec_adv or "unchanged" in _rec_adv
        # ...nor claim reuse is unavailable outright, which was the old error
        assert "does not have" not in _rec_adv, _rec_adv
        _plain_adv = " ".join(cache_miss_advice({}))
        assert "--n-depth" in _plain_adv and "swa-full" not in _plain_adv
        # and it reaches the design: metadata in, factor out
        assert "swa_full" in build_factors(Config(
            model=Path("g.gguf"), llama_bench=Path("b"), llama_server=Path("s"),
            array="auto", ctx_floor=8192, driver="server",
            hw={**_g3, "phys": 8, "logical": 16}))
        assert "swa_full" not in build_factors(Config(
            model=Path("q.gguf"), llama_bench=Path("b"), llama_server=Path("s"),
            array="auto", ctx_floor=8192, driver="server", emit_mtp=False,
            hw={**_q, "phys": 8, "logical": 16}))

        _hw_swa = {"phys": 8, "logical": 16, "n_layers": 18,
                   "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0}
        cfg_swa = Config(model=Path("g.gguf"), llama_bench=Path("b"),
                         llama_server=Path("s"), array="auto", ctx_floor=8192,
                         driver="server", hw={**_hw_swa, "n_swa": 512})
        assert build_factors(cfg_swa)["swa_full"] == ["0", "1"]
        # not on a model that attends globally...
        cfg_glob = Config(model=Path("q.gguf"), llama_bench=Path("b"),
                          llama_server=Path("s"), array="auto", ctx_floor=8192,
                          driver="server", hw={**_hw_swa, "n_swa": 0})
        assert "swa_full" not in build_factors(cfg_glob)
        # ...and not on the bench driver, which has no spelling for it
        cfg_bench = Config(model=Path("g.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           driver="bench", hw={**_hw_swa, "n_swa": 512})
        assert "swa_full" not in build_factors(cfg_bench)

        # --- ngl grid knows about VRAM (issue #14) ---
        # An even 0..n_layers span spends its slowest rows where the answer
        # cannot be: on a model that fits entirely, every CPU-offload level is a
        # near-certain loser, and ngl=0 is an order of magnitude slower than the
        # rest of the design put together.
        assert ngl_levels(40, 5) == [0, 10, 20, 30, 40]            # unchanged
        assert ngl_levels(40, 5, fits=False) == [0, 10, 20, 30, 40]
        assert ngl_levels(40, 5, fits=None) == [0, 10, 20, 30, 40]  # unknown
        _biased = ngl_levels(40, 5, fits=True)
        assert _biased == [0, 30, 33, 37, 40], _biased
        # the two slowest rows are what it removes
        assert 10 not in _biased and 20 not in _biased
        # NOT clustered at top-1/top-2. Two reasons, and the second is the one
        # that matters at any model size: on a 40-layer model those levels
        # differ by ~7% of the model's compute against a 5-27% measured
        # run-to-run spread; and if the fit verdict is WRONG, a design with
        # nothing between 0 and the top has no level left where a partially
        # offloaded optimum could show up.
        _gaps = [b - a for a, b in zip(_biased[1:], _biased[2:])]
        assert min(_gaps) >= 3, _biased
        # the window widens when a quarter is too narrow to hold levels-1
        # distinct values, so a small model keeps its level COUNT rather than
        # silently losing levels to deduplication
        assert ngl_levels(8, 5, fits=True) == [0, 4, 5, 7, 8]
        assert len(ngl_levels(18, 5, fits=True)) == 5
        # ...but a model with fewer layers than levels cannot, and does not
        # pretend to: every ngl value that exists is already in the span
        assert ngl_levels(4, 5, fits=True) == [0, 1, 3, 4]
        # ngl=0 survives EVERY level count and every verdict -- not for
        # information, but because the fit verdict can be wrong, and it is then
        # the only row that can still produce a measurement instead of an OOM.
        for _lv in (2, 3, 4, 5):
            for _top in (4, 8, 32, 40, 65):
                assert ngl_levels(_top, _lv, fits=True)[0] == 0
                assert ngl_levels(_top, _lv, fits=True)[-1] == _top
        # a model too small to carve a top quarter from keeps the even span
        assert ngl_levels(3, 5, fits=True) == ngl_levels(3, 5)

        # --- a crash is a measurement of a boundary (segfaults are data) ---
        # A Taguchi array visits each level beside DIFFERENT partners, so "every
        # row at this level failed" is evidence about the level. One failure is
        # not, and that is the generalisation this refuses to make.
        _sig = ([{"ngl": "0", "status": "OK"}] * 2
                + [{"ngl": "16", "status": "SIGNAL"}] * 3
                + [{"ngl": "32", "status": "SIGNAL"},
                   {"ngl": "32", "status": "OK"}])
        _d = dead_levels(_sig, "ngl")
        assert set(_d) == {"16"}, _d              # 32 produced a number once
        assert _d["16"] == {"SIGNAL": 3}, _d      # and the tally says why
        # one observation is never enough, however bad it looks
        assert dead_levels([{"ngl": "8", "status": "SIGNAL"}], "ngl") == {}
        # OOM/ERROR/TIMEOUT count too -- all of them mean "produced no number"
        assert set(dead_levels([{"ngl": "9", "status": "OOM"},
                                {"ngl": "9", "status": "TIMEOUT"}], "ngl")) == {"9"}
        # ...but SLOW and IMPLAUSIBLE do NOT: both had a working launch, so
        # neither says anything about bounds
        assert dead_levels([{"ngl": "9", "status": "SLOW"},
                            {"ngl": "9", "status": "SLOW"}], "ngl") == {}
        assert dead_levels([{"ngl": "9", "status": "IMPLAUSIBLE"}] * 2, "ngl") == {}

        # every level dead means the model or the box failed, not this knob --
        # narrowing there would hide the real fault
        _cfg_dead = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                           ctx_floor=8192)
        _cfg_dead.factors = {"ngl": ["0", "16"]}
        assert show_dead_levels(_cfg_dead, [{"ngl": "0", "status": "SIGNAL"}] * 2
                                + [{"ngl": "16", "status": "SIGNAL"}] * 2) == []
        _lines = show_dead_levels(_cfg_dead, [{"ngl": "0", "status": "OK"}] * 2
                                  + [{"ngl": "16", "status": "SIGNAL"}] * 2)
        assert len(_lines) == 1 and "ngl=16" in _lines[0], _lines

        # and the next --iterate pass stops visiting them
        _cfg_ref = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                          ctx_floor=8192)
        _cfg_ref.factors = {"ngl": ["0", "16", "32"], "n_depth": ["0"]}
        _rows_ref = ([{"ngl": "0", "n_depth": "0", "status": "OK", "eff_tps": 10.0}] * 2
                     + [{"ngl": "16", "n_depth": "0", "status": "SIGNAL", "eff_tps": 0.0}] * 2
                     + [{"ngl": "32", "n_depth": "0", "status": "OK", "eff_tps": 40.0}] * 2)
        assert "16" not in refine_factors(_cfg_ref, _rows_ref)["ngl"]

        # --- recurrent models: only two -ngl values run at all (issue #18) ---
        # Measured on Qwen3.6-27B-Q5_K_M (qwen35): -ngl 5 and -ngl 40 both
        # core-dump llama-server after "layer 0 is assigned to device CPU but
        # fused Gated Delta Net (chunked) is assigned to device ROCm0"; -ngl 0
        # and -ngl 99 run. It is the split that kills it, not the amount, so an
        # even span spends three of five levels on rows that abort.
        assert ngl_levels(64, 5, recurrent=True) == [0, 99]
        assert ngl_levels(40, 3, recurrent=True) == [0, 99]
        # 99, not n_layers: -ngl 99 is the spelling that was verified, and
        # -ngl n_layers still leaves the output tensor on the CPU
        assert 64 not in ngl_levels(64, 5, recurrent=True)
        # recurrent wins over the fit bias -- a level that aborts is worse than a
        # level that merely loses
        assert ngl_levels(64, 5, fits=True, recurrent=True) == [0, 99]
        # and it changes nothing for a model without recurrent memory
        assert ngl_levels(64, 5, recurrent=False) == ngl_levels(64, 5)

        # the collapsed grid is a DEFAULT, not a verdict: under --run the box is
        # asked which levels load, and the ones that do come back. Otherwise a
        # crash measured on one architecture and one backend would delete the
        # layers-for-context axis on hardware nobody here has seen.
        _real_sess = ServerSession
        try:
            class _StubSession:
                loads = set()

                def __init__(self, cfg, f, n_ctx, timeout):
                    self.ok = int(f["ngl"]) in self.loads

                def close(self):
                    pass

            globals()["ServerSession"] = _StubSession
            cfg_pr = Config(model=Path("q.gguf"), llama_bench=Path("b"),
                            llama_server=Path("s"), array="auto", ctx_floor=8192,
                            driver="server")
            cfg_pr.factors = {"ngl": ["0", "99"], "threads": ["4"]}
            # a box where the fused op is fine: everything comes back
            _StubSession.loads = {0, 10, 20, 30, 40, 99}
            assert probe_loadable_ngl(cfg_pr, [0, 10, 20, 30, 40, 99], 512, 5) \
                == ([0, 10, 20, 30, 40, 99], [])
            # this box: the measured dead band drops out, the rest stays
            _StubSession.loads = {0, 99}
            live, dead = probe_loadable_ngl(cfg_pr, [0, 10, 20, 30, 40, 99], 512, 5)
            assert live == [0, 99] and dead == [10, 20, 30, 40], (live, dead)
            # nothing loads at all: the caller must not be handed an empty grid
            _StubSession.loads = set()
            assert probe_loadable_ngl(cfg_pr, [0, 99], 512, 5) == ([], [0, 99])
        finally:
            globals()["ServerSession"] = _real_sess

        # WIRING: read off the model's own metadata, like swa_full
        _hw_rec = {"phys": 8, "logical": 16, "n_layers": 64,
                   "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0,
                   "ssm_state": 128}
        cfg_rec = Config(model=Path("q.gguf"), llama_bench=Path("b"),
                         llama_server=Path("s"), array="auto", ctx_floor=8192,
                         driver="server", hw=dict(_hw_rec))
        assert build_factors(cfg_rec)["ngl"] == ["0", "99"]
        assert cfg_rec.ngl_recurrent is True      # so the header can say why
        cfg_plain = Config(model=Path("t.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           driver="server", hw={**_hw_rec, "ssm_state": 0})
        assert build_factors(cfg_plain)["ngl"] != ["0", "99"]
        assert cfg_plain.ngl_recurrent is False

        # the fit probe asks about the WORST cell the design contains, since the
        # levels are generated once while n_depth varies per row: a model that
        # fits at depth 0 and not at 64k must not get an optimistic grid.
        _asked = {}

        def _fake_predict(cfg, f, driver):
            _asked.update(f)
            return True

        _real_predict = predict_fits
        try:
            globals()["predict_fits"] = _fake_predict
            cfg_fit = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                             array="auto", ctx_floor=8192)
            assert full_offload_fits(cfg_fit, ["0", "8192", "65536"],
                                     ["f16", "q8_0"], 40) is True
            assert _asked["n_depth"] == "65536", _asked   # deepest, not first
            assert _asked["ngl"] == "40", _asked          # all layers
            assert _asked["kv_type"] == "f16", _asked     # largest cache
            assert _asked["nkvo"] == "0", _asked          # KV resident on GPU
            # no offload flags: the probe is the un-offloaded footprint, so a
            # positive verdict means it fits without any of them
            assert not ({"ot", "ncmoe", "ncffn", "ffn_place"} & set(_asked))
            # --no-oom-prune means the estimator is not trusted to delete rows;
            # it is not trusted to shape the grid either
            cfg_fit.oom_prune = False
            assert full_offload_fits(cfg_fit, ["65536"], ["f16"], 40) is None
            # an unknown layer count is not a fit verdict
            cfg_fit.oom_prune = True
            assert full_offload_fits(cfg_fit, ["65536"], ["f16"], None) is None

            # WIRING: a level generator that knows about VRAM is worthless if
            # build_factors does not ask it. This is the connection that
            # silently regresses -- the same lesson as the I4/I5 wiring tests.
            _hw14 = {"phys": 8, "logical": 16, "n_layers": 40,
                     "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0}
            cfg_w = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           hw=dict(_hw14))
            globals()["predict_fits"] = lambda c, f, d: True
            assert build_factors(cfg_w)["ngl"] == ["0", "30", "33", "37", "40"]
            assert cfg_w.ngl_biased is True          # so the header can say so
            globals()["predict_fits"] = lambda c, f, d: None
            cfg_u = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           hw=dict(_hw14))
            assert build_factors(cfg_u)["ngl"] == ["0", "10", "20", "30", "40"]
            assert cfg_u.ngl_biased is False
        finally:
            globals()["predict_fits"] = _real_predict
        assert len(thread_levels(8, 16)) >= 3

        # --- conditional-factor core (docs/CONDITIONAL-FACTORS.md) ---
        # the real registry is valid
        assert validate_factor_registry() == [], validate_factor_registry()
        # is_active: unconditional always active; conditional gated on its gate
        assert is_active("ngl", {}) is True                  # unconditional
        assert is_active("does_not_exist", {}) is True       # unknown → active
        synth = {
            "G":     {"server": ("--g",), "kind": "cat", "translate": {"a": "a", "b": "b", "c": "c"}},
            "A":     {"server": ("--a",), "kind": "num"},                    # unconditional
            "Pa":    {"server": ("--pa",), "kind": "num", "active_when": ("G", {"a"})},
            "Pshare":{"kind": "num", "active_when": ("G", {"a", "b"}),
                      "flag_for": lambda v: (f"--p-{v}",)},
        }
        _saved = dict(FACTORS)
        FACTORS.clear(); FACTORS.update(synth)
        try:
            assert is_active("Pa", {"G": "a"}) is True
            assert is_active("Pa", {"G": "b"}) is False       # gate ≠ live value
            assert is_active("Pa", {}) is False               # gate absent
            assert is_active("Pshare", {"G": "b"}) is True
            assert is_active("Pshare", {"G": "c"}) is False
            assert active_factors(["A", "Pa", "Pshare"], {"G": "a"}) == ["A", "Pa", "Pshare"]
            assert active_factors(["A", "Pa", "Pshare"], {"G": "b"}) == ["A", "Pshare"]
            # flag_for resolves the variant-dependent spelling
            assert conditional_flags("Pshare", "b") == ("--p-b",)
            assert conditional_flags("Pa", "a") == ("--pa",)   # static fallback
            assert validate_factor_registry() == []            # synth is valid
            # validator catches: missing gate, bad live value, flag_for gap, cycle
            bad = {"X": {"kind": "num", "active_when": ("Nope", {"z"})}}
            assert validate_factor_registry(bad)
            badval = {"G": synth["G"], "X": {"server": ("--x",), "kind": "num",
                                             "active_when": ("G", {"zzz"})}}
            assert any("not levels" in e for e in validate_factor_registry(badval))
            badff = {"G": synth["G"], "X": {"kind": "num", "active_when": ("G", {"a"}),
                                            "flag_for": lambda v: ()}}
            assert any("non-empty" in e for e in validate_factor_registry(badff))
            cyc = {"G": {"server": ("--g",), "kind": "cat", "active_when": ("H", {"x"})},
                   "H": {"server": ("--h",), "kind": "cat", "active_when": ("G", {"y"})}}
            assert any("cycle" in e for e in validate_factor_registry(cyc))
        finally:
            FACTORS.clear(); FACTORS.update(_saved)

        # MoE metadata
        assert model_expert_count({"llama.expert_count": 8}) == 8
        assert model_expert_count({}) == 0

        # Pareto frontier (maximize depth and tg_tps)
        rows = [
            {"status": "OK", "n_depth": "0", "tg_tps": 50.0},
            {"status": "OK", "n_depth": "16384", "tg_tps": 40.0},
            {"status": "OK", "n_depth": "16384", "tg_tps": 30.0},  # dominated
            {"status": "OK", "n_depth": "4096", "tg_tps": 20.0},   # dominated
            {"status": "OOM", "n_depth": "65536", "tg_tps": 0.0},  # excluded
        ]
        depths = sorted(int(r["n_depth"]) for r in pareto_frontier(rows))
        assert depths == [0, 16384], depths

        # pareto SVG: x-axis zooms to the data range (like y) — a single-depth
        # run must not stack every point on the right edge (regression: a final
        # refinement pass at one depth drew all 25 dots at x=right-margin), and
        # every tick must land on-plot (rounding ticks up past xmax drew them
        # outside the viewBox).
        svg1 = _svg_pareto([{"status": "OK", "n_depth": "49152", "tg_tps": t}
                            for t in (10.0, 12.0, 14.0)])
        cxs = [float(m) for m in re.findall(r"circle cx='([\d.]+)'", svg1)]
        assert cxs and all(200 < c < 500 for c in cxs), cxs   # centered, not edge
        svg2 = _svg_pareto([{"status": "OK", "n_depth": d, "tg_tps": t} for d, t in
                            [("0", 50.0), ("16384", 40.0), ("65536", 20.0)]])
        ticks = [float(m) for m in
                 re.findall(r"<text x='([\d.-]+)' y='\d+' class='ax xt'", svg2)]
        assert ticks and all(62 <= t <= 664 for t in ticks), ticks  # on-plot

        # command builder: flag map, env split, batch clamp
        cfg = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                     ctx_floor=16384, env_factor_names={"GGML_CUDA_FORCE_MMQ"})
        f = {"ngl": "64", "threads": "8", "kv_type": "q4_0", "ubatch": "2048",
             "n_depth": "0", "nkvo": "1", "poll": "50", "GGML_CUDA_FORCE_MMQ": "1"}
        cmd = bench_command(cfg, f)
        assert "-nkvo" in cmd and "--poll" in cmd
        assert "-ctk" in cmd and "-ctv" in cmd
        assert "GGML_CUDA_FORCE_MMQ" not in " ".join(cmd)  # env not on cmdline
        assert run_env(cfg, f)["GGML_CUDA_FORCE_MMQ"] == "1"
        # -b is derived from -ub, so a ratio row can never emit -b < -ub (C1) and
        # ratio 1 lands exactly on -ub — the low-batch regime an absolute floor of
        # 2048 used to hide.
        cmd2 = bench_command(cfg, {"ubatch": "2048", "batch_ratio": "4"})
        assert cmd2[cmd2.index("-b") + 1] == "8192"
        cmd3 = bench_command(cfg, {"ubatch": "128", "batch_ratio": "1"})
        assert cmd3[cmd3.index("-b") + 1] == "128"

        # --- constrained (derived) factors: docs/CONSTRAINED-FACTORS.md ---
        # Rounding must keep the relation unconditional. round() would not:
        # banker's rounding sends 0.5*1 -> 0 but 0.5*3 -> 2, and only floor makes
        # frac 1.0 land exactly on the base.
        for n_max, frac, want in [("1", "0.5", 0), ("1", "1.0", 1), ("1", "0.0", 0),
                                  ("2", "0.5", 1), ("3", "0.5", 1), ("4", "0.5", 2),
                                  ("6", "0.5", 3), ("6", "1.0", 6)]:
            got = derived_value("spec_n_min_frac",
                                {"spec_n_max": n_max, "spec_n_min_frac": frac})
            assert got == want, f"n_max={n_max} frac={frac}: {got} != {want}"
            assert got <= int(n_max)                      # C1
        # offset op, and the llama.cpp default pair (n_min 48, n_max 64)
        assert derived_value("ngram_mod_n_max_off",
                             {"ngram_mod_n_min": "48",
                              "ngram_mod_n_max_off": "16"}) == 64
        assert derived_value("ngram_mod_n_max_off",
                             {"ngram_mod_n_min": "96",
                              "ngram_mod_n_max_off": "0"}) == 96
        # base not swept: derive from that run's fixed value for it
        cfg_d = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, spec_draft_n_max=4)
        assert derived_value("spec_n_min_frac", {"spec_n_min_frac": "0.5"}, cfg_d) == 2
        assert derived_value("batch_ratio", {"batch_ratio": "4"}) == 2048  # -ub 512

        # a conditional derived factor records no absolute where it is inactive
        abscols = derived_abs_cols(cfg_d, {"ngram": "ngram-simple",
                                           "ngram_mod_n_min": "48",
                                           "ngram_mod_n_max_off": "16"})
        assert abscols["ngram_mod_n_max"] == "", abscols

        # registry validation catches a malformed derived declaration
        bad = dict(FACTORS)
        bad["spec_n_min_frac"] = {**FACTORS["spec_n_min_frac"],
                                  "derived_from": ("kv_type", "scale")}
        errs = validate_factor_registry(bad)
        assert any("must be numeric" in e for e in errs), errs
        bad2 = {"a": {"kind": "num", "server": ("-a",),
                      "derived_from": ("b", "scale"), "relation": "at_most"},
                "b": {"kind": "num", "server": ("-b",),
                      "derived_from": ("a", "scale"), "relation": "at_most"}}
        assert any("cycle" in e for e in validate_factor_registry(bad2)), \
            validate_factor_registry(bad2)
        # and level sets that would break the relation are rejected up front
        assert validate_factor_levels({"spec_n_min_frac": ["0.0", "0.5"]}) == []
        assert validate_factor_levels({"spec_n_min_frac": ["2.0"]})
        assert validate_factor_levels({"batch_ratio": ["0"]})
        assert validate_factor_levels({"ngram_mod_n_max_off": ["-16"]})

        # A base a derived factor leaned on but that is NOT swept must be pinned
        # explicitly, or llama.cpp would apply its own default and the relation
        # would have been guaranteed against a number that never reached it.
        cfg_pin = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                         ctx_floor=8192, driver="server", spec_draft_n_max=8,
                         hw={"phys": 8, "logical": 16, "n_layers": 32,
                             "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1})
        ap_ = build_server_args(cfg_pin, {"spec_n_min_frac": "1.0",
                                          "ubatch": "512"}, 8080, 4096)
        assert ap_.count("--spec-draft-n-max") == 1, ap_       # pinned, not doubled
        assert int(ap_[ap_.index("--spec-draft-n-min") + 1]) \
            <= int(ap_[ap_.index("--spec-draft-n-max") + 1]), ap_
        # base swept: no pin needed, still exactly one flag
        bp_ = build_server_args(cfg_pin, {"spec_n_max": "3", "spec_n_min_frac": "1.0",
                                          "ubatch": "512"}, 8080, 4096)
        assert bp_.count("--spec-draft-n-max") == 1, bp_
        assert bp_[bp_.index("--spec-draft-n-min") + 1] == "3", bp_
        # same for -b/-ub: sweeping the ratio without ubatch pins -ub
        cp_ = bench_command(cfg_pin, {"batch_ratio": "4"})
        assert int(cp_[cp_.index("-b") + 1]) >= int(cp_[cp_.index("-ub") + 1]), cp_

        # THE issue-#8 property. Checked as a FULL CROSS-PRODUCT of each derived
        # factor's levels against its base's levels, not over the rows some array
        # happened to draw: an OA can only ever emit cells from this grid, so
        # covering the grid covers every design — including the ones
        # refine_factors invents on later passes, which is what a level-set-only
        # fix could never have guaranteed. Cross-product needs no array binding,
        # so this stays inside --selftest's stdlib-only promise; the same check
        # over real generated rows lives in .github/workflows/binding_smoke.py.
        def _assert_grid_cannot_invert(factor_levels, cfgx):
            cells = 0
            for dn in derived_names(factor_levels):
                base_name = FACTORS[dn]["derived_from"][0]
                bases = factor_levels.get(base_name)
                bases = bases if bases else [str(DERIVED_BASE_FALLBACK[base_name](cfgx))]
                for bl in bases:
                    for dl in factor_levels[dn]:
                        row = {base_name: bl, dn: dl}
                        base, val = derived_base(dn, row, cfgx), derived_value(dn, row, cfgx)
                        if FACTORS[dn]["relation"] == "at_most":
                            assert val <= base, (dn, bl, dl, val, base)
                        else:
                            assert val >= base, (dn, bl, dl, val, base)
                        cells += 1
            return cells

        hw_d = {"phys": 8, "logical": 16, "n_layers": 32,
                "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1}
        cfg_mtp = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                         ctx_floor=8192, driver="server", hw=dict(hw_d))
        fm = build_factors(cfg_mtp)
        # the exact cell issue #8 reported: spec_n_max=1 with the top min level
        assert "1" in fm["spec_n_max"] and "1.0" in fm["spec_n_min_frac"]
        assert derived_value("spec_n_min_frac",
                             {"spec_n_max": "1", "spec_n_min_frac": "1.0"}) == 1
        assert _assert_grid_cannot_invert(fm, cfg_mtp) >= 15
        cfg_tune = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                          ctx_floor=8192, driver="server", ngram=True,
                          ngram_type="ngram-mod",
                          hw={**hw_d, "n_nextn": 0})
        assert _assert_grid_cannot_invert(build_factors(cfg_tune), cfg_tune) >= 25
        # and after a refinement pass rebuilds the numeric grids
        refined = dict(fm)
        refined["spec_n_max"] = refine_numeric([1, 2, 3, 4, 6], 2)
        refined["batch_ratio"] = refine_numeric([1, 4, 16], 1)
        assert _assert_grid_cannot_invert(refined, cfg_mtp) > 0

        # --- speculative telemetry (docs/field-reports.md, F1) ---
        # Acceptance sums over the measured responses.
        assert _draft_totals([{"draft_n": 10, "draft_n_accepted": 6, "predicted_n": 32},
                              {"draft_n": 10, "draft_n_accepted": 4, "predicted_n": 32}]) == \
            {"drafted": 20, "accepted": 10, "generated": 64}
        assert draft_cols(cfg_mtp, {"mtp": "1"},
                          {"drafted": 20, "accepted": 10, "generated": 40}) == \
            {"draft_acc": 0.5, "draft_cov": 0.25}
        # A drafted-but-never-accepted run is a real (bad) result, not the guard:
        # speculation ran, it just never paid. Conflating the two would hide the
        # one case we most want to see.
        assert draft_cols(cfg_mtp, {"mtp": "1"},
                          {"drafted": 40, "accepted": 0, "generated": 64}) == \
            {"draft_acc": 0.0, "draft_cov": 0.0}
        # acc measures draft QUALITY, cov measures how much speculation actually
        # contributed, and acc alone ranks configs BACKWARDS in the case that
        # matters. These are the real measured shapes: a drafter that is always
        # right about the few tokens it dares to guess (acc 1.00) ran at 579 t/s;
        # one that is right 84% of the time but guesses twice as often ran at 846.
        timid = draft_cols(cfg_mtp, {"mtp": "1"},
                           {"drafted": 59, "accepted": 59, "generated": 128})
        bold = draft_cols(cfg_mtp, {"mtp": "1"},
                          {"drafted": 124, "accepted": 104, "generated": 128})
        assert timid["draft_acc"] > bold["draft_acc"], (timid, bold)
        assert timid["draft_cov"] < bold["draft_cov"], (timid, bold)
        # generated missing (older llama.cpp) must not divide by zero
        assert draft_cols(cfg_mtp, {"mtp": "1"},
                          {"drafted": 10, "accepted": 5})["draft_cov"] == 0.0
        # llama.cpp omits BOTH keys when no draft ran, so their absence is the
        # signal and must never be read as "0 tokens drafted".
        assert _draft_totals([{"predicted_n": 64}, {"predicted_n": 64}]) == {}
        # Asked for speculation, drafted nothing: the issue-#8 shape. Flagged,
        # but the measurement stays real — draft_cols returns no status.
        assert draft_cols(cfg_mtp, {"mtp": "1"}, {}) == {"spec_off": 1}
        assert draft_cols(cfg_mtp, {"mtp": "0"}, {}) == {}
        # The gate is not always a factor. MTP fixed-on (NextN head, no --no-mtp)
        # is the common case, and reading only the assignment would miss it.
        assert draft_cols(cfg_mtp, {"ngl": "32"}, {}) == {"spec_off": 1}
        cfg_nospec = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                            ctx_floor=8192, driver="server",
                            hw={**hw_d, "n_nextn": 0})
        assert draft_cols(cfg_nospec, {"ngl": "32"}, {}) == {}
        assert draft_cols(cfg_nospec, {"ngram": "ngram-mod"}, {}) == {"spec_off": 1}
        # Columns appear only where a draft is possible at all (no inert columns).
        cfg_sc = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                        ctx_floor=8192, driver="server", hw=dict(hw_d))
        cfg_sc.factors = build_factors(cfg_sc)
        assert spec_cols_wanted(cfg_sc)
        cfg_nospec.factors = build_factors(cfg_nospec)
        assert not spec_cols_wanted(cfg_nospec)
        cfg_bn = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                        ctx_floor=8192, driver="bench", hw=dict(hw_d))
        cfg_bn.factors = build_factors(cfg_bn)
        assert not spec_cols_wanted(cfg_bn)      # bench cannot speculate
        # A telemetry column is never mistaken for a factor column by the report.
        assert "draft_acc" in RESULT_COLS and "spec_off" in RESULT_COLS
        # llama-bench has no --fit; emitting one makes it exit non-zero, which
        # would fail every bench run. Verified against the binary: it answers
        # "error: invalid parameter for argument: --no-fit".
        assert "--no-fit" not in cmd and "--fit" not in cmd, cmd

        # Capability probe. The boundary matters: llama-bench advertises
        # --fit-target and --fit-ctx but has no --fit, so a substring test would
        # emit a flag it rejects.
        class _B:
            def __init__(self, h): self.h = h
            def __str__(self): return "bin-" + str(id(self))
        _saved_help = dict(_help_cache)
        try:
            _help_cache["srv"] = "-fit,  --fit [on|off]   whether to adjust unset args\n"
            _help_cache["bench"] = ("-fitt, --fit-target <MiB>  fit model to memory\n"
                                    "-fitc, --fit-ctx <n>       minimum ctx\n")
            _help_cache["old"] = "-m, --model FNAME\n"       # predates --fit
            assert supports_flag(Path("srv"), "--fit")
            assert not supports_flag(Path("bench"), "--fit")   # the substring trap
            assert supports_flag(Path("bench"), "--fit-target")
            assert not supports_flag(Path("old"), "--fit")

            # --- load mode (CHANGELOG C1) ---
            # -mmp/--no-mmap are deprecated in favour of --load-mode, and
            # llama.cpp does delete deprecated args. Prefer the new spelling
            # where the binary has it, keep the old where it does not, and never
            # emit both (arg.cpp: "only the last flag takes effect").
            _help_cache["lm"] = "-lm, --load-mode MODE   model loading mode\n"
            assert load_mode_args(Path("lm"), "bench", True) == ["--load-mode", "mmap"]
            assert load_mode_args(Path("lm"), "bench", False) == ["--load-mode", "none"]
            # one spelling, both drivers -- the legacy pair never agreed
            assert load_mode_args(Path("lm"), "server", True) == ["--load-mode", "mmap"]
            assert load_mode_args(Path("lm"), "server", False) == ["--load-mode", "none"]
            # older build: llama-bench takes -mmp 0|1 ...
            assert load_mode_args(Path("old"), "bench", True) == ["-mmp", "1"]
            assert load_mode_args(Path("old"), "bench", False) == ["-mmp", "0"]
            # ... while llama-server has only --no-mmap and mmaps by default, so
            # "on" is the ABSENCE of a flag. Emitting -mmp there would be fatal.
            assert load_mode_args(Path("old"), "server", True) == []
            assert load_mode_args(Path("old"), "server", False) == ["--no-mmap"]
            # R4: swept and fixed emission must not both fire, or llama.cpp
            # takes the last flag and the row measures something else
            cfg_lm = Config(model=Path("m"), llama_bench=Path("lm"),
                            llama_server=Path("lm"), array="auto", ctx_floor=8192,
                            hw={"phys": 8, "logical": 16, "n_layers": 32,
                                "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0})
            c_swept = bench_command(cfg_lm, {"load_mode": "mlock", "ubatch": "512"})
            assert c_swept.count("--load-mode") == 1, c_swept
            assert c_swept[c_swept.index("--load-mode") + 1] == "mlock", c_swept
            c_fixed = bench_command(cfg_lm, {"ubatch": "512"})
            assert c_fixed.count("--load-mode") == 1, c_fixed
            assert c_fixed[c_fixed.index("--load-mode") + 1] == "mmap", c_fixed
            s_swept = build_server_args(cfg_lm, {"load_mode": "dio", "ubatch": "512"},
                                        8080, 4096)
            assert s_swept.count("--load-mode") == 1, s_swept
            assert s_swept[s_swept.index("--load-mode") + 1] == "dio", s_swept

            # F3: llama.cpp defaults --op-offload and --repack ON, so a disabled
            # level must SAY so. Emitting nothing would leave the feature on
            # while the column records 0 -- a row measuring the opposite of its
            # own label, which is the issue-#8 shape in a new place.
            off = _flat(factor_flags(cfg_lm, {"no_op_offload": "0"}, "server"))
            assert off == ["--op-offload"], off
            on = _flat(factor_flags(cfg_lm, {"no_op_offload": "1"}, "server"))
            assert on == ["--no-op-offload"], on
            rp0 = _flat(factor_flags(cfg_lm, {"repack": "0"}, "server"))
            assert rp0 == ["--no-repack"], rp0
            rp1 = _flat(factor_flags(cfg_lm, {"repack": "1"}, "server"))
            assert rp1 == ["--repack"], rp1
            # a default-OFF boolean keeps the old behaviour: absent means absent
            assert _flat(factor_flags(cfg_lm, {"no_host": "0"}, "server")) == []
            assert _flat(factor_flags(cfg_lm, {"no_host": "1"}, "server")) == ["--no-host"]

            # F2: one level, one meaning, on both drivers. The bench spelling is
            # negative (-nopo <0|1>) and the server spelling is a positive/negative
            # pair; naming the factor for the negative is what keeps them aligned.
            b_on = _flat(factor_flags(cfg_lm, {"no_op_offload": "1"}, "bench"))
            assert b_on == ["-nopo", "1"], b_on
            b_off = _flat(factor_flags(cfg_lm, {"no_op_offload": "0"}, "bench"))
            assert b_off == ["-nopo", "0"], b_off

            # F1: every registered knob is reachable by name -- a registry entry
            # --factor would reject is the bug this batch exists to fix.
            for _n in ("load_mode", "no_op_offload", "no_host", "repack",
                       "swa_full", "backend_sampling", "prio", "prio_batch",
                       "ctx_checkpoints", "checkpoint_min_step", "cpu_mask_batch",
                       "cpu_range_batch", "cpu_strict_batch", "poll_batch"):
                assert _n in FACTORS, _n
                assert FACTORS[_n].get("server"), _n     # all reach the server
            assert not validate_factor_registry(), validate_factor_registry()
            for drv in ("bench", "server"):
                for binary in (Path("lm"), Path("old")):
                    for on in (True, False):
                        got = load_mode_args(binary, drv, on)
                        assert not ({"-mmp", "--no-mmap"} & set(got)
                                    and "--load-mode" in got), got
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)

        # effective throughput objective
        assert effective_tps(512, 256, 0, 100) == 0.0
        assert abs(effective_tps(512, 256, 1000.0, 100.0)
                   - 768 / (512 / 1000 + 256 / 100)) < 1e-6
        assert set(PROFILES) == {"single", "agents", "multi"}

        # scoring: tg-only by default — a huge pp can't mask slow decode;
        # --score eff restores the blended request throughput
        cfg_o = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, n_prompt=512, n_gen=256)
        assert cfg_o.score == "tg"
        assert objective_tps(cfg_o, 5000.0, 8.4) == 8.4
        assert objective_tps(cfg_o, 0.0, 100.0) == 100.0  # pp glitch ≠ failed run
        assert objective_tps(cfg_o, 100.0, 0.0) == 0.0
        cfg_o.score = "eff"
        assert abs(objective_tps(cfg_o, 1000.0, 100.0)
                   - effective_tps(512, 256, 1000.0, 100.0)) < 1e-9
        assert objective_tps(cfg_o, 0.0, 100.0) == 0.0    # blend needs both

        # --diff and --merge-results dedup (temp CSVs; offline, no binding)
        with tempfile.TemporaryDirectory() as td:
            cols = ["run_id", "ngl", "n_depth", "pp_tps", "tg_tps", "eff_tps",
                    "status", "secs", "temp_c"]

            def wcsv(name, rws):
                p = Path(td) / name
                with open(p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rws)
                return p

            def rrow(i, ngl, tg, status="OK"):
                return {"run_id": i, "ngl": ngl, "n_depth": "0", "pp_tps": 500.0,
                        "tg_tps": tg, "eff_tps": tg, "status": status,
                        "secs": 5, "temp_c": ""}

            old = wcsv("run.pass1.csv", [rrow(1, "32", 40.0), rrow(2, "16", 30.0),
                                         rrow(3, "48", 0.0, "OOM")])
            new = wcsv("new.csv", [rrow(1, "32", 44.0), rrow(2, "16", 29.0),
                                   rrow(3, "48", 50.0)])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                s = diff_results(old, new)
                assert s["matched"] == 3 and s["compared"] == 2
                assert s["status_changes"] == 1             # ngl=48: OOM -> OK
                assert s["old_winner_still_wins"] is False  # 48 took the lead
                assert abs(s["new_best_tg"] - 50.0) < 1e-9
                assert diff_results(old, Path(td) / "missing.csv") is None

                # merge dedup: a merged row is kept only if it beats every
                # known measurement of that exact config
                cfg_r = Config(model=Path("m"), llama_bench=Path("b"),
                               array="auto", ctx_floor=8192)
                cfg_r.factors = {"ngl": ["16", "32", "48"], "n_depth": ["0"]}
                out = merge_result_rows(cfg_r, [rrow(9, "32", 42.0)], [old])
                assert len(out) == 3                    # 32 deduped; 16+48 added
                got = {r["ngl"]: r for r in out}
                assert got["32"]["tg_tps"] == 42.0      # current row won the dup
                assert got["16"]["run_id"] == "pass1:2"  # tagged with its pass
                assert got["48"]["status"] == "OOM"     # failures still carried
                # a FASTER earlier measurement does get in (never lose a best)
                out = merge_result_rows(cfg_r, [rrow(9, "32", 39.0)], [old])
                assert {r["tg_tps"] for r in out
                        if r["ngl"] == "32"} == {39.0, 40.0}
                # no merge files: rows pass through untouched (no dedup of the
                # array's own intentional replicates)
                reps = [rrow(1, "32", 40.0), rrow(2, "32", 41.0)]
                assert merge_result_rows(cfg_r, reps, []) is reps
            assert "NEW winner: ngl=48" in buf.getvalue()

        # recommended -c never exceeds the verified footprint (regression:
        # a row measured at depth 49152 must NOT be emitted as -c 65536).
        cfgc = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                      n_prompt=512, n_gen=256, ctx_floor=8192)
        assert recommended_ctx(cfgc, {"n_depth": "49152"}) == 50176
        for d in (0, 4096, 49152, 131072):
            ctx = recommended_ctx(cfgc, {"n_depth": str(d)})
            assert ctx <= cfgc.n_prompt + d + cfgc.n_gen + 256      # not inflated
            assert ctx >= d + cfgc.n_prompt + cfgc.n_gen            # covers request
            assert ctx % 256 == 0
        # parallel multiplies the verified session context
        cfgp = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                      ctx_floor=8192, n_prompt=512, n_gen=256, parallel=4)
        assert recommended_ctx(cfgp, {"n_depth": "8192"}) == (512 + 8192 + 256 + 256) * 4

        # usable floor on the emitted -c: raised to ctx_floor when (and only as
        # far as) a sibling verified deeper; explicit lower floors are honored;
        # with no deeper evidence the row's own footprint stands, plus a note.
        assert recommended_ctx(cfgc, {"n_depth": "0"}, verified_depth=49152) == 8192
        assert recommended_ctx(cfgc, {"n_depth": "0"}, verified_depth=4096) == 5120
        assert recommended_ctx(cfgc, {"n_depth": "0"}) == 1024
        cfg_lo = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                        n_prompt=512, n_gen=256, ctx_floor=2048)
        assert recommended_ctx(cfg_lo, {"n_depth": "0"}, verified_depth=49152) == 2048
        assert ctx_floor_note(cfgc, {"n_depth": "0"}, 1024)           # capped => note
        assert ctx_floor_note(cfgc, {"n_depth": "0"}, 8192) is None   # floor met

        # ...and never exceeds the context the run actually OCCUPIED. The server
        # driver sizes its prompt in characters against an assumed 4
        # chars/token; the reporter's model tokenized the battery's prose at
        # 6.05, so a row labelled n_depth=32768 ran ~27,000 tokens and this
        # emitted -c 41472 on the strength of it (issue #11).
        cfg_m = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                       n_prompt=8192, n_gen=256, ctx_floor=8192,
                       factors={"ngl": ["99"], "n_depth": ["32768"]})
        _short = {"n_depth": "32768", "ngl": "99", "status": "OK",
                  "prompt_tok": "27100"}
        assert measured_footprint(cfg_m, _short, 1) == 27100 + 256 + 256
        assert recommended_ctx(cfg_m, _short, 32768, [_short]) == 27392
        # unrecorded is an unknown, not a zero: bench rows and CSVs written
        # before the column existed keep the behaviour they had
        _blind = {"n_depth": "32768", "ngl": "99", "status": "OK"}
        assert measured_footprint(cfg_m, _blind, 1) == 0
        assert (recommended_ctx(cfg_m, _blind, 32768, [_blind])
                == recommended_ctx(cfg_m, _blind, 32768))
        # the cap spends the same siblings verified_depth does, so a deeper row
        # that DID reach its depth still lifts a shallow one
        _deep = {"n_depth": "49152", "ngl": "99", "status": "OK",
                 "prompt_tok": "57600"}
        assert recommended_ctx(cfg_m, _short, 49152, [_short, _deep]) > 27392
        # ...but only a sibling that SUCCEEDED. A row that OOMed at depth is
        # evidence the context does not hold, so letting it raise the cap would
        # paste a -c that is known not to load.
        _oomed = {**_deep, "status": "OOM"}
        assert (recommended_ctx(cfg_m, _short, 49152, [_short, _oomed])
                == recommended_ctx(cfg_m, _short, 49152, [_short]))
        assert verified_footprint(cfg_m, [_oomed], _short, 1) == 0
        # ...and only a sibling of THIS launch config. A different ngl is a
        # different server, and its footprint says nothing about this one.
        _other = {**_deep, "ngl": "40"}
        assert verified_footprint(cfg_m, [_other], _short, 1) == 0
        assert (recommended_ctx(cfg_m, _short, 49152, [_short, _other])
                == recommended_ctx(cfg_m, _short, 49152, [_short]))

        # verified_depth_of: deepest OK sibling sharing the launch factors
        cfg_v = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_v.factors = {"ngl": ["60", "99"], "n_depth": ["0", "16384", "49152"]}
        rows_v = [{"status": "OK", "ngl": "60", "n_depth": "0", "eff_tps": 40.},
                  {"status": "OK", "ngl": "60", "n_depth": "49152", "eff_tps": 30.},
                  {"status": "OOM", "ngl": "60", "n_depth": "16384", "eff_tps": 0.},
                  {"status": "OK", "ngl": "99", "n_depth": "0", "eff_tps": 50.}]
        assert verified_depth_of(cfg_v, rows_v, rows_v[0]) == 49152
        assert verified_depth_of(cfg_v, rows_v, rows_v[3]) == 0   # no deep sibling

        # FASTEST = fastest *usable*: the raw-fastest config (ngl 99, never
        # verified past depth 0) is skipped for the fastest one that holds the
        # floor — no bench-number chasing; falls back when nothing qualifies.
        fast, bal, lng = pick_recommendations(cfg_v, rows_v)
        assert fast is rows_v[0] and bal is rows_v[1] and lng is rows_v[1]
        f2, b2, _ = pick_recommendations(cfg_v, [rows_v[3]])
        assert f2 is rows_v[3] and b2 is None
        assert pick_recommendations(cfg_v, []) == (None, None, None)

        # thermal wait-and-watch: no baseline => immediate no-op (never blocks)
        assert wait_until_cool(None) is None
        # a RISING temperature (post-run heat soak) is not a plateau (regression:
        # `prev - t < 0.5` was true for negatives, exiting at the hottest moment);
        # a genuine stall (cooling < 0.5°C/poll) or reaching baseline+band settles.
        real_temp, calls = gpu_temp_c, []

        def fake_temp(seq):
            it = iter(seq)
            return lambda: (calls.append(1), next(it, seq[-1]))[1]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                globals()["gpu_temp_c"] = fake_temp([60.0, 62.0, 61.9, 55.0])
                wait_until_cool(40.0, band=5.0, cap_s=99, poll_s=0)
                assert len(calls) == 3   # rode out the rise; exited on the stall
                calls.clear()
                globals()["gpu_temp_c"] = fake_temp([60.0, 50.0, 44.0])
                wait_until_cool(40.0, band=5.0, cap_s=99, poll_s=0)
                assert len(calls) == 3   # cooled to <= baseline+band and settled
        finally:
            globals()["gpu_temp_c"] = real_temp

        # q8_0 downgrade hint fires only for KV heavier than the q8_0 floor
        assert kv_downgrade_hint({"kv_type": "f16"})
        assert kv_downgrade_hint({"kv_type": "f32"})
        assert kv_downgrade_hint({"kv_type": "q8_0"}) is None
        assert kv_downgrade_hint({"kv_type": "q4_0"}) is None

        # realistic prompt: varied text of ~n*4 chars, not a repeated token
        assert len(_realistic_prompt(100)) == 400
        assert len(set(_realistic_prompt(200))) > 20  # genuinely varied

        # array auto-selection by factor levels
        assert choose_array({f"f{i}": ["a", "b", "c", "d", "e"] for i in range(5)}) == "L25"
        assert choose_array({f"f{i}": ["a", "b", "c"] for i in range(6)}) == "L27"
        assert choose_array({f"f{i}": ["0", "1"] for i in range(7)}) == "L8"
        # mixed levels ride the largest base's array (the binding has no L18):
        # a 2-level factor among 3-level ones maps onto a 3-level column
        assert choose_array({"a": ["0", "1"], "b": ["x", "y", "z"],
                             "c": ["p", "q", "r"]}) == "L9"
        # 7 varying factors overflow L25's 6 columns -> the 125-run array
        assert choose_array({f"f{i}": ["a", "b", "c", "d", "e"]
                             for i in range(7)}) == "L125"
        # a fixed (1-level) factor among 3-level ones still picks a 3-level array
        assert choose_array({"t": ["8"], "a": ["1", "2", "3"],
                             "b": ["1", "2", "3"]}) == "L9"
        # settled constants don't count: one lone varying factor => no array
        # (direct sweep), not a 25-run L25 to replicate 5 configs 5×.
        assert choose_array({"ngl": ["56", "57", "58", "59", "60"],
                             "d": ["49152"], "t": ["8"], "kv": ["f16"]}) is None
        assert choose_array({}) is None

        # generate_runs: <=1 active factor enumerates directly (N runs, not N×N),
        # constants attached to every row; needs no array binding.
        exp0, runs0 = generate_runs({"ngl": ["56", "58", "60"], "d": ["49152"],
                                     "kv": ["f16"]}, "auto")
        assert exp0 is None and len(runs0) == 3            # one run per ngl level
        assert all(r["factors"]["d"] == "49152" and r["factors"]["kv"] == "f16"
                   for r in runs0)                         # constants ride along
        assert [r["factors"]["ngl"] for r in runs0] == ["56", "58", "60"]
        _, runs1 = generate_runs({"ngl": ["60"], "kv": ["f16"]}, "auto")
        assert len(runs1) == 1                             # 0 active => single config

        # refinement: settle the flat factor, refine the high-impact one
        assert refine_numeric([20, 40, 60], 60) == ["40", "45", "50", "55", "60"]
        cfg_r = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_r.factors = {"ngl": ["20", "40", "60"], "kv_type": ["f16", "q8_0"]}
        rr = [{"status": "OK", "ngl": a, "kv_type": k, "eff_tps": e} for a, k, e in
              [("20", "f16", 10.), ("40", "q8_0", 50.), ("60", "f16", 90.),
               ("20", "q8_0", 11.), ("60", "q8_0", 89.), ("40", "f16", 49.)]]
        ref = refine_factors(cfg_r, rr)
        assert ref["kv_type"] == ["q8_0"]          # flat factor settled at winner
        assert ref["ngl"] == ["40", "45", "50", "55", "60"]  # refined near best (60)
        # n_depth is the tradeoff axis: kept spread across passes, never settled,
        # so the final pass still maps the whole speed/context curve.
        cfg_d = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_d.factors = {"ngl": ["20", "40", "60"],
                         "n_depth": ["0", "16384", "32768", "49152", "65536"]}
        rd = [{"status": "OK", "ngl": "60", "n_depth": d, "eff_tps": e}
              for d, e in [("0", 30.), ("16384", 25.), ("32768", 20.),
                           ("49152", 15.), ("65536", 10.)]]
        assert refine_factors(cfg_d, rd)["n_depth"] == cfg_d.factors["n_depth"]

        # unified registry: driver mapping, server-only, -ot translation, bools
        assert is_server_only("spec_p_min") and not is_server_only("ngl")
        assert FACTORS["threads_batch"]["bench"] is None      # server-only flag
        cfg_s = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server")
        assert factor_flags(cfg_s, {"ot": "none"}, "bench") == []   # none omits
        assert factor_flags(cfg_s, {"ot": "exps_cpu"}, "bench")[0][0] == "-ot"
        assert factor_flags(cfg_s, {"fa": "0"}, "bench") == [["-fa", "0"]]
        assert factor_flags(cfg_s, {"nkvo": "1"}, "server") == [["-nkvo"]]  # bare
        assert factor_flags(cfg_s, {"nkvo": "1"}, "bench") == [["-nkvo", "1"]]

        # MTP as a swept factor: on/off via translate ("" omits), server-only
        assert factor_flags(cfg_s, {"mtp": "1"}, "server") == \
            [["--spec-type", "draft-mtp"]]
        assert factor_flags(cfg_s, {"mtp": "0"}, "server") == []
        assert factor_flags(cfg_s, {"mtp": "1"}, "bench") == []
        cfg_m = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server",
                       hw={"phys": 8, "logical": 16, "n_layers": 32,
                           "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1})
        sa = build_server_args(cfg_m, {"mtp": "0", "ubatch": "512"}, 8080, 4096)
        assert "--spec-type" not in sa      # swept off: automatic flag yields
        sa = build_server_args(cfg_m, {"ubatch": "512"}, 8080, 4096)
        assert "--spec-type" in sa and "draft-mtp" in sa  # fixed on if not swept
        # --fit off is emitted only when the binary advertises --fit, and is
        # never spelled --no-fit (llama-server: "invalid argument: --no-fit").
        _saved_help = dict(_help_cache)
        try:
            _help_cache[str(cfg_m.llama_server)] = "-fit, --fit [on|off]  adjust unset args\n"
            sfit = build_server_args(cfg_m, {"ubatch": "512"}, 8080, 4096)
            assert "--no-fit" not in sfit, sfit
            assert sfit[sfit.index("--fit") + 1] == "off", sfit
            # a build without --fit gets nothing rather than a rejected flag
            _help_cache[str(cfg_m.llama_server)] = "-m, --model FNAME\n"
            sold = build_server_args(cfg_m, {"ubatch": "512"}, 8080, 4096)
            assert "--fit" not in sold and "--no-fit" not in sold, sold
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)
        # default factor set: nkvo/poll/batch always; ffn_place for dense /
        # ncmoe for MoE; threads_batch + the MTP surface only on the server
        # driver; numa only on a multi-node box
        _saved_help = dict(_help_cache)
        try:
            _help_cache[str(cfg_m.llama_server)] = "-m, --model FNAME\n"   # older build
            fs = build_factors(cfg_m)
            assert all(k in fs for k in ("nkvo", "poll", "batch_ratio", "threads_batch",
                                         "mtp", "spec_n_max", "spec_n_min_frac",
                                         "spec_p_min", "spec_p_split"))
            assert "batch" not in fs and "spec_n_min" not in fs   # renamed, now derived
            assert "ncmoe" not in fs                           # dense
            assert "numa" not in fs                            # single NUMA node
            # Pre-b10645: the -ot regimes only. The factor NAME is the same as
            # on a new build — capability changes the level count, not the
            # column, so a CSV stays comparable across a llama.cpp upgrade.
            assert fs["ffn_place"] == ["none", "ffn_up_cpu", "ffn_cpu"], fs["ffn_place"]
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)
        # With -ncffn (llama.cpp b10645) two GRADED levels appear between "none"
        # and "all of it" — the middle -ot cannot express. up_cpu survives: it
        # is a different axis (which tensor, not how many layers), so -ncffn
        # does not subsume it.
        _saved_help = dict(_help_cache)
        try:
            _help_cache[str(cfg_m.llama_server)] = ("-ncffn, --n-cpu-ffn N  "
                                                    "keep dense FFN on CPU\n")
            fs2 = build_factors(cfg_m)
            assert fs2["ffn_place"] == ["none", "ffn_up_cpu", "first_8",
                                        "first_16", "ffn_cpu"], fs2["ffn_place"]
            assert "ot" not in fs2 and "ncffn" not in fs2, fs2   # one column, not three
            # the level picks the flag: -ncffn for graded, -ot for the rest
            assert factor_flags(cfg_m, {"ffn_place": "first_16"}, "server") == \
                [["-ncffn", "16"]]
            assert factor_flags(cfg_m, {"ffn_place": "ffn_up_cpu"}, "bench") == \
                [["-ot", OT_PATTERNS["ffn_up_cpu"]]]
            assert factor_flags(cfg_m, {"ffn_place": "none"}, "server") == []
            # every level must emit something DIFFERENT, or the column is
            # balanced over levels that are the same run. The real bug this
            # caught: a level spelled 'up_cpu' misses the OT_PATTERNS key
            # 'ffn_up_cpu', emits nothing, and silently duplicates 'none'.
            assert validate_factor_levels({"ffn_place": fs2["ffn_place"]}) == []
            dupes = validate_factor_levels(
                {"ffn_place": ["none", "up_cpu", "ffn_cpu"]})
            assert dupes and "both emit" in dupes[0], dupes
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)
        cfg_m.hw["numa_nodes"] = 2
        assert "numa" in build_factors(cfg_m)
        cfg_m.hw["numa_nodes"] = 1
        cfg_m.driver = "bench"
        fs = build_factors(cfg_m)
        assert "nkvo" in fs and "poll" in fs and "batch_ratio" in fs
        assert all(k not in fs for k in ("mtp", "spec_n_max", "spec_n_min_frac",
                                         "spec_p_min", "spec_p_split",
                                         "threads_batch"))  # server-only

        # ngram as a swept factor: translate maps variant names, "none" omits
        assert factor_flags(cfg_s, {"ngram": "ngram-mod"}, "server") == \
            [["--spec-type", "ngram-mod"]]
        assert factor_flags(cfg_s, {"ngram": "none"}, "server") == []
        assert factor_flags(cfg_s, {"ngram": "ngram-simple"}, "server") == \
            [["--spec-type", "ngram-simple"]]
        assert factor_flags(cfg_s, {"ngram": "ngram-mod"}, "bench") == []
        # all ngram factors are server-only
        for nf in ("ngram", "ngram_size_n", "ngram_size_m", "ngram_min_hits",
                   "ngram_mod_n_match", "ngram_mod_n_min", "ngram_mod_n_max_off"):
            assert FACTORS[nf].get("server_only"), f"{nf} should be server_only"

        # conditional ngram knobs: emitted only for their active variant (I2), and
        # the collapsed shared knob resolves the variant-specific flag via flag_for.
        gated = _flat(factor_flags(cfg_s, {"ngram": "ngram-mod",
                                           "ngram_mod_n_match": "16",
                                           "ngram_size_n": "8"}, "server"))
        assert "--spec-ngram-mod-n-match" in gated, f"got {gated}"
        assert "--spec-ngram-simple-size-n" not in gated, f"got {gated}"
        # same shared knob, different variant → different flag spelling
        for variant in ("ngram-simple", "ngram-map-k", "ngram-map-k4v"):
            g = _flat(factor_flags(cfg_s, {"ngram": variant, "ngram_size_m": "32",
                                           "ngram_mod_n_min": "48",
                                           "ngram_mod_n_max_off": "16"}, "server"))
            assert f"--spec-{variant}-size-m" in g, f"{variant}: {g}"
            assert "--spec-ngram-mod-n-max" not in g          # mod knob inactive here
        # ngram=none → variant flag omitted AND every conditional knob suppressed
        assert factor_flags(cfg_s, {"ngram": "none", "ngram_mod_n_match": "16",
                                    "ngram_size_n": "8"}, "server") == []

        # ngram screen build: the gate only (no conditional children — those enter
        # a variant's tuning stage), and NOT spec_n_max (a draft-model knob).
        cfg_n = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server", ngram=True,
                       hw={"phys": 8, "logical": 16, "n_layers": 32,
                           "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0})
        nfs = build_factors(cfg_n)
        assert "ngram" in nfs and len(nfs["ngram"]) == 5, f"ngram not in {list(nfs)}"
        assert "spec_n_max" not in nfs, "spec_n_max must not be an ngram factor"
        assert not any(k.startswith("ngram_") and k != "ngram" for k in nfs), \
            "screen build must exclude conditional children"
        # a pinned variant (tuning stage) includes ONLY its own knobs
        vf_mod = build_factors(replace(cfg_n, ngram_type="ngram-mod"))
        assert vf_mod["ngram"] == ["ngram-mod"]
        assert "ngram_mod_n_max_off" in vf_mod and "ngram_size_n" not in vf_mod
        vf_simple = build_factors(replace(cfg_n, ngram_type="ngram-simple"))
        assert "ngram_size_n" in vf_simple and "ngram_mod_n_max_off" not in vf_simple

        # I3: a conditional knob is scored only over rows where it was active.
        # ngram_size_n really matters within ngram-simple (best at "24"); rows on
        # a different variant carry a DECOY best ("4") that must be ignored.
        cond_rows = (
            [{"status": "OK", "ngram": "ngram-simple", "ngram_size_n": sz,
              "pp_tps": 100.0, "tg_tps": v, "n_prompt": 0, "n_gen": 128}
             for sz, v in (("4", 10.0), ("12", 20.0), ("24", 30.0))] +
            [{"status": "OK", "ngram": "ngram-mod", "ngram_size_n": sz,
              "pp_tps": 100.0, "tg_tps": v, "n_prompt": 0, "n_gen": 128}
             for sz, v in (("4", 90.0), ("12", 5.0), ("24", 5.0))]
        )
        cond_means = factor_level_means(cond_rows, "ngram_size_n")
        assert max(cond_means, key=cond_means.get) == "24", cond_means   # true best
        assert "90.0" not in [f"{v}" for v in cond_means.values()]        # decoy row excluded

        # --- gated factors: a pinned-inert gate must not size the array ---
        # Reported on issue #11: `--factor mtp=0` turns speculation off, but the
        # four speculative-tuning knobs kept their full level sets, so the sweep
        # generated 25 runs of a configuration that could not differ -- and the
        # main-effects table then reported an effect for each of them, computed
        # from nothing but noise.
        _spec = {"spec_n_max": ["1", "2", "3", "4", "6"],
                 "spec_n_min_frac": ["0.0", "0.5", "1.0"],
                 "spec_p_min": ["0.0", "0.25", "0.5", "0.75", "0.9"],
                 "spec_p_split": ["0.1", "0.3", "0.5"]}
        _off, _gone = prune_gated_factors({"ngl": ["99"], "mtp": ["0"], **_spec})
        assert sorted(_gone) == sorted(_spec), _gone
        assert set(_off) == {"ngl", "mtp"}, _off
        # a gate pinned to a LIVE value keeps them -- that is the run being tuned
        _on, _gone_on = prune_gated_factors({"mtp": ["1"], **_spec})
        assert _gone_on == [] and set(_on) == {"mtp", *_spec}
        # ...and a gate still being SWEPT keeps them: one of its levels is live
        _both, _gone_both = prune_gated_factors({"mtp": ["1", "0"], **_spec})
        assert _gone_both == [], _gone_both
        # no gate in the design at all is not evidence of inertness -- a draft
        # model speculates with no `mtp` column, and its knobs must survive
        _nogate, _gone_ng = prune_gated_factors(dict(_spec))
        assert _gone_ng == [] and set(_nogate) == set(_spec)
        # unrelated factors are never touched
        assert prune_gated_factors({"ngl": ["1", "2"]}) == ({"ngl": ["1", "2"]}, [])
        # and with mtp explicitly off, the inert default flag is not pasted either
        cfg_mtp_off = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                             llama_server=Path("s"), array="auto", ctx_floor=8192,
                             driver="server", emit_mtp=True,
                             hw={"phys": 8, "logical": 16, "n_layers": 32,
                                 "n_ctx_train": 32768, "n_experts": 0,
                                 "n_nextn": 1})
        _a_off = build_server_args(cfg_mtp_off, {"mtp": "0", "ngl": "99"}, 8080, 4096)
        assert "--spec-draft-n-max" not in _a_off, _a_off
        assert "--spec-type" not in _a_off, _a_off
        _a_on = build_server_args(cfg_mtp_off, {"mtp": "1", "ngl": "99"}, 8080, 4096)
        assert "--spec-draft-n-max" in _a_on, _a_on

        # --- a gated knob that could not act does not vote (issue #16) ---
        # `gated_by` pruned the DESIGN when the gate is pinned off (#11). When
        # the gate is SWEPT, half the rows still carry the knob at a level that
        # cannot do anything, and those rows were being averaged into its main
        # effect -- crediting a knob that did nothing with an effect computed
        # from run-to-run noise.
        assert is_inert("spec_n_max", {"mtp": "0"}) is True
        assert is_inert("spec_n_max", {"mtp": "1"}) is False
        # an ABSENT gate is not evidence, and this is where is_inert must differ
        # from is_active: --draft-model speculates with no mtp column at all, so
        # treating absence as inertness would delete every draft-model row from
        # the effect and stop emitting the flag.
        assert is_inert("spec_n_max", {}) is False
        assert is_active("spec_n_max", {}) is True      # no active_when on it
        assert is_inert("ngl", {"mtp": "0"}) is False   # ungated, never inert

        # the effect is computed over the rows that could show it
        _mrows = ([{"status": "OK", "mtp": "1", "spec_n_max": str(n),
                    "eff_tps": v} for n, v in ((1, 20.0), (2, 30.0),
                                               (4, 40.0), (6, 50.0))]
                  + [{"status": "OK", "mtp": "0", "spec_n_max": str(n),
                      "eff_tps": 15.0} for n in (1, 2, 4, 6)])
        _eff = factor_level_means(_mrows, "spec_n_max")
        assert _eff == {"1": 20.0, "2": 30.0, "4": 40.0, "6": 50.0}, _eff
        # ...which is the point: averaging the inert rows in halves the apparent
        # range (17.5..32.5 instead of 20..50) and understates the knob
        assert max(_eff.values()) - min(_eff.values()) == 30.0
        # the GATE itself is unconditional and keeps every row -- mtp=0 rows are
        # exactly what its own effect is measured against
        assert factor_level_means(_mrows, "mtp") == {"0": 15.0, "1": 35.0}

        # and the flag is not pasted into a row where it does nothing
        cfg_in = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                        llama_server=Path("s"), array="auto", ctx_floor=8192,
                        driver="server", emit_mtp=True,
                        hw={"phys": 8, "logical": 16, "n_layers": 32,
                            "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1})
        _off = build_server_args(cfg_in, {"mtp": "0", "spec_n_max": "6",
                                          "spec_p_min": "0.9", "ngl": "99"},
                                 8080, 4096)
        assert not [a for a in _off if a.startswith("--spec-")], _off
        _on = build_server_args(cfg_in, {"mtp": "1", "spec_n_max": "6",
                                         "spec_p_min": "0.9", "ngl": "99"},
                                8080, 4096)
        assert "--spec-draft-n-max" in _on and "--spec-draft-p-min" in _on, _on
        # a row with no mtp column keeps its flags: that is the draft-model shape
        _dm = build_server_args(cfg_in, {"spec_n_max": "4", "ngl": "99"},
                                8080, 4096)
        assert "--spec-draft-n-max" in _dm, _dm

        # --- stage planner (docs/CONDITIONAL-FACTORS.md) ---
        # feed the planner the FULL factor set (screen gate + every child) it would
        # decompose; build_factors only ever emits one stage's worth at a time.
        full = dict(nfs); full.update(NGRAM_MAP_LEVELS); full.update(NGRAM_MOD_LEVELS)
        stages = plan_stages(full)
        assert stages[0]["name"] == "screen"
        # screen sweeps the gate at full spread but excludes conditional children
        assert stages[0]["factors"]["ngram"] == full["ngram"]
        assert not any(_active_when(f) for f in stages[0]["factors"])
        # one tuning stage per variant that has children (3 map + mod)
        tune = {s["value"]: s for s in stages if s["gate"] == "ngram"}
        assert set(tune) == {"ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"}
        # a map variant tunes the 3 collapsed knobs; mod tunes its own 3; gate pinned
        assert set(tune["ngram-simple"]["factors"]) == \
            {"ngram", "ngram_size_n", "ngram_size_m", "ngram_min_hits"}
        assert set(tune["ngram-mod"]["factors"]) == \
            {"ngram", "ngram_mod_n_match", "ngram_mod_n_min", "ngram_mod_n_max_off"}
        assert tune["ngram-simple"]["factors"]["ngram"] == ["ngram-simple"]
        # I1 liveness: every factor in every stage is active under the stage's pin
        for s in stages:
            assert all(is_active(f, s["pin"]) for f in s["factors"]), s["name"]
        # F2: each tuning stage is small (gate + 3 knobs = 4 factors → fits L25),
        # versus the flat design's L125
        assert all(len(s["factors"]) <= 4 for s in stages if s["gate"])
        # a subset factor set plans just the screen + the one live variant's stage
        ps = plan_stages({"ngram": ["none", "ngram-mod"],
                          "ngram_mod_n_max_off": ["0", "16"], "ngl": ["0", "64"]})
        assert [s["name"] for s in ps] == ["screen", "tune:ngram=ngram-mod"]
        assert "ngl" in ps[0]["factors"] and \
            "ngram_mod_n_max_off" not in ps[0]["factors"]

        # --- ngram staging helpers (run_ngram_stages) ---
        screen_rows = [
            {"status": "OK", "ngram": "ngram-mod", "pp_tps": 100.0, "tg_tps": 30.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OK", "ngram": "ngram-simple", "pp_tps": 100.0, "tg_tps": 25.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OK", "ngram": "none", "pp_tps": 100.0, "tg_tps": 20.0,
             "ngl": "32", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OOM", "ngram": "ngram-map-k", "pp_tps": 0.0, "tg_tps": 0.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
        ]
        ranked = rank_gate_values(screen_rows, "ngram")
        assert [v for v, _ in ranked][:2] == ["ngram-mod", "ngram-simple"]  # best first
        assert keep_top_gate_values(ranked, 2) == ["ngram-mod", "ngram-simple"]
        assert keep_top_gate_values(ranked, 1) == ["ngram-mod"]             # --ngram-fast
        assert "none" not in keep_top_gate_values(ranked, 5)                # never tune "off"
        held = screen_base_winners(cfg_n, {"ngl": ["32", "64"], "n_depth": ["0", "4096"],
                                           "ngram": nfs["ngram"]}, screen_rows)
        assert "ngram" not in held                                         # gate pinned separately
        assert held["ngl"] == ["64"]                                       # settled at winner
        assert held["n_depth"] == ["0", "4096"]                            # tradeoff axis kept
        # child argv carries --ngram-type for a pinned tuning stage
        class _A:  # minimal args stand-in for build_child_argv
            model = Path("m"); no_mtp = no_shuffle = no_thermal_wait = False
            thermal_baseline = seed = max_depth = None; timeout = 60; cooldown = 0
            confirm = full_ = html = None; no_probe = False; verify_picks = 2
            merge_results = []
        _av = build_child_argv(_A, replace(cfg_n, ngram_type="ngram-mod"),
                               {"ngl": ["64"]}, Path("r.csv"), False, [])
        assert "--ngram-type" in _av and _av[_av.index("--ngram-type") + 1] == "ngram-mod"

        # spec-type merging: mtp + ngram both present → comma-separated
        merged = _merge_spec_type_parts(["--spec-type draft-mtp", "--spec-type ngram-mod"])
        assert merged == ["--spec-type draft-mtp,ngram-mod"], f"got {merged}"
        merged = _merge_spec_type_args(["--spec-type", "draft-mtp", "--spec-type", "ngram-mod"])
        assert merged == ["--spec-type", "draft-mtp,ngram-mod"], f"got {merged}"
        # only one spec-type → unchanged
        assert _merge_spec_type_parts(["--spec-type draft-mtp"]) == ["--spec-type draft-mtp"]
        assert _merge_spec_type_args(["--spec-type", "draft-mtp"]) == ["--spec-type", "draft-mtp"]
        # no spec-type → no change
        assert _merge_spec_type_parts(["-ngl 64"]) == ["-ngl 64"]
        assert _merge_spec_type_args(["-ngl", "64"]) == ["-ngl", "64"]

        # build_server_args with ngram: when ngram swept off, no spec-type;
        cfg_ns = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                        ctx_floor=8192, driver="server",
                        hw={"phys": 8, "logical": 16, "n_layers": 32,
                            "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0},
                        ngram=True)
        sa = build_server_args(cfg_ns, {"ngram": "none", "ubatch": "512"}, 8080, 4096)
        assert "--spec-type" not in sa, f"ngram=none should suppress spec-type, got {sa}"
        sa = build_server_args(cfg_ns, {"ubatch": "512"}, 8080, 4096)
        # ngram fixed on (not swept): should emit --spec-type ngram-mod
        assert "--spec-type" in sa
        idx = sa.index("--spec-type")
        val = sa[idx + 1]
        assert val == "ngram-mod", f"expected ngram-mod, got {val}"
        cfg_m.hw["n_experts"] = 64
        fs = build_factors(cfg_m)
        assert "ncmoe" in fs and "ot" not in fs            # MoE

        # KV quality floor
        assert kv_at_or_above(["f16", "q8_0", "q5_1", "q4_0"], "q8_0") == ["f16", "q8_0"]
        assert kv_at_or_above(["f16", "q8_0", "q4_0"], "any") == ["f16", "q8_0", "q4_0"]
        assert kv_at_or_above(["q4_0"], "q8_0") == ["q8_0"]  # never empty -> floor

        # morris analyze table parsing
        mtxt = ("Factor  mu*  sigma  note\n------  ----  -----  ----\n"
                "ubatch  96  0\nngl  32  0.001  \n\nRanked by mu* ...\n")
        assert parse_morris_analyze(mtxt) == [("ubatch", 96.0, 0.0), ("ngl", 32.0, 0.001)]
        # morris --json: the stable contract, used for decisions. Real shape,
        # captured from `morris analyze ... --json`.
        mjson = json.dumps({
            "tool": "morris", "command": "analyze", "schema": 1,
            "metric": "eff_tps", "trajectories": 2, "all_zero": False,
            "factors": [
                {"factor": "kv_type", "rank": 1, "mu_star": 215.572275,
                 "sigma": 8.066638804, "share": 0.939, "interacting": False},
                {"factor": "n_depth", "rank": 2, "mu_star": 8.8437,
                 "sigma": 12.50688048, "share": 0.038, "interacting": True},
            ]})
        d, why = parse_morris_json(mjson)
        assert why is None and d["all_zero"] is False
        assert [f["factor"] for f in d["factors"]] == ["kv_type", "n_depth"]
        assert d["factors"][0]["mu_star"] == 215.572275
        # an unknown schema must be refused, not guessed at -- upstream bumps it
        # only on a rename or removal, so a field we rely on may have moved
        d2, why2 = parse_morris_json(json.dumps({"tool": "morris", "schema": 2,
                                                 "factors": []}))
        assert d2 is None and "schema" in why2
        # older morris (no --json) prints the table; fall back rather than crash
        d3, why3 = parse_morris_json("Morris elementary effects (metric: x)\n")
        assert d3 is None and why3
        assert parse_morris_json("")[0] is None
        assert parse_morris_json(json.dumps({"tool": "sobol", "schema": 1}))[0] is None
        # all_zero is the tell for the exact defect that started this: a screen
        # where the response never moved, previously indistinguishable from a
        # legitimate empty ranking
        d4, _ = parse_morris_json(json.dumps({"tool": "morris", "schema": 1,
                                              "all_zero": True, "factors": []}))
        assert d4 and d4["all_zero"] is True

        # ...and the same table once morris grew a 95% CI on mu*. The CI is
        # printed glued to the value, so a positional split() reads it as the
        # mu* token and drops every row -- which silently turned --screen into
        # a no-op (rankings empty => "keeping all factors") after paying for
        # the runs. Both spellings, glued and space-separated, must parse.
        mci = ("Factor       mu* [95% CI]   sigma   note\n"
               "------       ------------   -----   ----\n"
               "ngl       4.867[4.4,5.51]   4.809   interacting/nonlinear\n"
               "ubatch    4.831 [4.1,5.5]    5.14   interacting/nonlinear\n"
               "\nRanked by mu* (importance).\n")
        assert parse_morris_analyze(mci) == [("ngl", 4.867, 4.809),
                                             ("ubatch", 4.831, 5.14)]

        # --- every flag we emit must exist on the real binary ---
        # The gap that let --no-fit through: asserting a flag is in OUR argv
        # proves nothing about whether llama.cpp accepts it, and llama.cpp exits
        # non-zero on an unknown argument -- so a bad flag fails every run, not
        # one. Skipped when llama.cpp is absent (CI, and --selftest's promise to
        # need no GPU or model), which is why it cannot be the only guard.
        _lb = resolve_binary("llama-bench", None, None)
        _ls = resolve_binary("llama-server", None, None)
        if _lb.exists() and _ls.exists():
            fcfg = Config(model=Path("m.gguf"), llama_bench=_lb, llama_server=_ls,
                          array="auto", ctx_floor=8192, driver="bench",
                          factors={}, hw={"phys": 8, "logical": 16})
            ff = {"ngl": "99", "n_depth": "0", "kv_type": "f16",
                  "ubatch": "512", "threads": "8"}
            for _which, _cmd, _bin in (
                    ("bench", bench_command(fcfg, ff), _lb),
                    ("server", build_server_args(fcfg, ff, 8080, 4096), _ls)):
                emitted = [t for t in _cmd
                           if isinstance(t, str) and t.startswith("-")
                           and not t[1:2].isdigit()]
                unknown = [x for x in emitted if not supports_flag(_bin, x)]
                assert not unknown, (
                    f"{_which} command emits flags this llama.cpp rejects: "
                    f"{unknown} — it will exit non-zero on every run")

        # --- llama-fit-params output parsing (the OOM pruner's input) ---
        # Real output, captured from llama-fit-params --fit-print on. A wrong
        # number here prunes a config that would have run, and the row never
        # appears to contradict it.
        assert parse_fit_print("ROCm0 271 27 513 \nHost 170 0 12") == 811
        assert parse_fit_print("Host 170 0 12") == 0        # host RAM is not VRAM
        # multi-GPU: every device line counts toward the footprint
        assert parse_fit_print("CUDA0 271 27 513\nCUDA1 100 10 90\nHost 5 0 1") == 1011
        assert parse_fit_print("") == 0
        assert parse_fit_print("garbage\nROCm0 1 2 3") == 6   # skip unparseable
        assert parse_fit_print("ROCm0 a b c") == 0            # non-numeric ignored

        # ...and predict_fits must actually use it. Testing the parser alone
        # leaves the wiring free to regress: stubbing the parse out still
        # passed until this was added, because a footprint of 0 fits anything
        # and the pruner would simply stop pruning.
        _real_run2, _real_exists = subprocess.run, Path.exists
        try:
            class _P:
                returncode, stderr = 0, ""
                def __init__(self, out): self.stdout = out
            fit_cfg = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                             array="auto", ctx_floor=8192, driver="bench",
                             fit_params=Path("llama-fit-params"),
                             hw={"phys": 8, "logical": 16, "vram": 8192})
            Path.exists = lambda self: True          # pretend the binary is there
            row_f = {"ngl": "20", "n_depth": "8192", "kv_type": "f16"}

            # 811 MiB against 8192 - 512 headroom -> fits
            subprocess.run = lambda *a, **k: _P("ROCm0 271 27 513\nHost 170 0 12")
            _fit_cache.clear()
            assert predict_fits(fit_cfg, row_f, "bench") is True
            # 20000 MiB on the same card -> cannot fit
            subprocess.run = lambda *a, **k: _P("ROCm0 19000 500 500\nHost 170 0 12")
            _fit_cache.clear()
            assert predict_fits(fit_cfg, row_f, "bench") is False
            # host-only output means no GPU footprint seen -> must not prune
            subprocess.run = lambda *a, **k: _P("Host 170 0 12")
            _fit_cache.clear()
            assert predict_fits(fit_cfg, row_f, "bench") is True
        finally:
            subprocess.run, Path.exists = _real_run2, _real_exists
            _fit_cache.clear()

        # --- submodule discovery (the reorg that broke us) ---
        # Both are layout-dependent by nature: upstream moved taguchi under
        # optimize/ and stopped building it into taguchi/build, which silently
        # broke the binding. These assert the discovery still lands on a real
        # file, so the next reorg fails here instead of mid-sweep.
        if SUBMODULE_DIR.exists():
            binding = find_taguchi_binding()
            assert (binding / "taguchi" / "__init__.py").exists(), binding
            cli = find_robust_binary("taguchi")
            if cli.exists():               # only meaningful once built
                prepare_taguchi_cli()
                assert os.environ.get("TAGUCHI_CLI_PATH") == str(cli)
                assert str(cli.parent) in os.environ["PATH"].split(os.pathsep)
                # the binding's own fallback is shutil.which, so PATH must work
                assert shutil.which("taguchi")

        # --- multi-GPU VRAM detection (issue #5) ---
        # Both smi tools emit one line/key per device. Reading only the first
        # understates a multi-GPU box, and that number is the OOM pruner's
        # limit -- which it compares against a footprint summed over every GPU
        # llama-fit-params reports. Mismatched scopes prune configs that fit.
        assert parse_nvidia_vram("24564\n24564\n") == [24564, 24564]
        assert sum(parse_nvidia_vram("24564\n24564\n")) == 49128
        assert parse_nvidia_vram("8192\n") == [8192]        # single GPU unchanged
        assert parse_nvidia_vram("") == []                  # nothing detected
        assert parse_nvidia_vram("\n  \n") == []
        assert parse_nvidia_vram("12288.0\n") == [12288]    # float-formatted
        assert parse_nvidia_vram("N/A\n8192\n") == [8192]   # skip unparseable

        # real rocm-smi shape (bytes). NOTE both keys contain "total", so a
        # bare substring match for total also matches the *used* key and the
        # result would depend on key order -- assert we pick the right one.
        rocm1 = ('{"card0": {"VRAM Total Memory (B)": "34342961152", '
                 '"VRAM Total Used Memory (B)": "96665600"}}')
        assert parse_rocm_vram(rocm1, "total") == [32752]
        assert parse_rocm_vram(rocm1, "used") == [92]
        # two cards, and the used key listed first to prove order-independence
        rocm2 = ('{"card0": {"VRAM Total Used Memory (B)": "1048576", '
                 '"VRAM Total Memory (B)": "8589934592"}, '
                 '"card1": {"VRAM Total Memory (B)": "17179869184", '
                 '"VRAM Total Used Memory (B)": "2097152"}}')
        assert parse_rocm_vram(rocm2, "total") == [8192, 16384]
        assert sum(parse_rocm_vram(rocm2, "total")) == 24576
        assert parse_rocm_vram(rocm2, "used") == [1, 2]
        assert parse_rocm_vram('{"card0": {}}', "total") == []

        # --- capacity comes from llama.cpp, not the vendor tool (issue #7) ---
        # An APU runs the model out of GTT, so `rocm-smi --showmeminfo vram`
        # answers with the 2 GiB carve-out while llama.cpp reports the ~30 GiB
        # it can really use. Verbatim from the issue, both tools on the same box:
        rocm_apu = ('{"card0": {"VRAM Total Memory (B)": "2147483648", '
                    '"VRAM Total Used Memory (B)": "1205387264"}}')
        assert parse_rocm_vram(rocm_apu, "total") == [2048]   # what we used to use
        apu = parse_list_devices(
            "Available devices:\n"
            "  ROCm0: AMD Radeon 780M Graphics (30438 MiB, 43542 MiB free)\n")
        assert apu == [{"id": "ROCm0", "name": "AMD Radeon 780M Graphics",
                        "total_mib": 30438, "free_mib": 43542}], apu
        # ~15x the smi figure: the gap that pruned the reporter's whole sweep.
        assert sum(d["total_mib"] for d in apu) == 30438

        # issue #5's mixed pair, verbatim -- CUDA and ROCm lines are the same
        # shape, which is why one parser replaces both vendor paths. Order is
        # llama.cpp's own, the order -ts will index into.
        mixed = parse_list_devices(
            "Available devices:\n"
            "  CUDA0: NVIDIA GeForce RTX 3090 (24117 MiB, 6672 MiB free)\n"
            "  CUDA1: NVIDIA GeForce RTX 3060 (11909 MiB, 9837 MiB free)\n")
        assert [d["id"] for d in mixed] == ["CUDA0", "CUDA1"]
        assert [d["total_mib"] for d in mixed] == [24117, 11909]
        assert [d["free_mib"] for d in mixed] == [6672, 9837]
        assert sum(d["total_mib"] for d in mixed) == 36026
        # free is NOT total here (a display holds the rest of CUDA0); anything
        # deriving a split from capacity must not quietly read one for the other
        assert sum(d["free_mib"] for d in mixed) != sum(d["total_mib"] for d in mixed)

        # a CPU device would be system RAM wearing a device line: dropped, or
        # the GPU budget overstates the machine worse than the bug being fixed
        assert parse_list_devices(
            "  CPU0: AMD Ryzen 9 (64000 MiB, 32000 MiB free)\n"
            "  Vulkan0: Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)\n") == [
            {"id": "Vulkan0", "name": "Radeon RX 7900 XTX",
             "total_mib": 24560, "free_mib": 24000}]
        # a build too old for the flag prints no device lines -> fall back
        assert parse_list_devices("error: invalid argument: --list-devices") == []
        assert parse_list_devices("") == []
        assert parse_list_devices("Available devices:\n") == []
        # backend chatter on the same stream must not parse as a device
        assert parse_list_devices(
            "ggml_cuda_init: found 1 ROCm devices:\n"
            "  Device 0: AMD Radeon 780M Graphics, gfx1103 (0x1103)\n") == []
        # list_devices() with no binary is [] rather than an exception, so
        # detection degrades to the smi path instead of aborting the run
        assert list_devices(None) == []
        assert list_devices(Path("/nonexistent/llama-server")) == []
        # a device list that sums to zero is not an answer -- fall through to
        # the smi path rather than reporting a 0 MiB machine
        assert parse_list_devices("  ROCm0: some accelerator (0 MiB, 0 MiB free)\n") == [
            {"id": "ROCm0", "name": "some accelerator",
             "total_mib": 0, "free_mib": 0}]

        # --- GPU visibility: a CPU-only BUILD vs a CPU-only MACHINE ---
        # detect_vram_mib's source string already carries the diagnosis, which is
        # why this needs no second probe.
        # --- free VRAM: a different question from total ---
        # The dev box's own state when this was written: another project held
        # the card, every run aborted in the CUDA allocator, and the pruner
        # (which compares against TOTAL) passed every one of them first.
        _busy = [{"id": "ROCm0", "total_mib": 32752, "free_mib": 275}]
        assert vram_headroom(_busy) == (275, 32752)
        assert headroom_warning(_busy) is not None
        assert "another process" in headroom_warning(_busy)
        assert "275 of 32752" in headroom_warning(_busy)
        # an idle card says nothing
        assert headroom_warning([{"id": "ROCm0", "total_mib": 32752,
                                  "free_mib": 32000}]) is None
        # a tight but workable fit is not a warning: a model wanting more than
        # half a free card is ordinary, and crying wolf here trains it away
        assert headroom_warning([{"id": "CUDA0", "total_mib": 24576,
                                  "free_mib": 9000}]) is None
        # summed over devices, like detect_vram_mib — one busy card in a pair
        # does not read as "fine" because the other is idle (issue #5 shape)
        _pair = [{"id": "CUDA0", "total_mib": 24117, "free_mib": 300},
                 {"id": "CUDA1", "total_mib": 11909, "free_mib": 400}]
        assert vram_headroom(_pair) == (700, 36026)
        assert "CUDA0 300/24117" in headroom_warning(_pair)
        assert "CUDA1 400/11909" in headroom_warning(_pair)
        # unknown / absent free data must not manufacture a warning
        assert headroom_warning([]) is None
        assert vram_headroom([]) == (0, 0)
        assert headroom_warning([{"id": "ROCm0", "total_mib": 0,
                                  "free_mib": 0}]) is None
        # --- draft model as an INPUT (F2 / issue #12) ---
        _ls, _fp = Path("ls"), Path("fp")
        _hw = {"phys": 8, "logical": 16, "n_layers": 65, "n_ctx_train": 262144,
               "n_experts": 0, "n_nextn": 1, "vram": 24576, "numa_nodes": 1}
        _saved_help = dict(_help_cache)
        _td_fit = tempfile.TemporaryDirectory()
        td_fit = _td_fit.name
        try:
            _help_cache[str(_ls)] = _help_cache[str(_fp)] = "-ncmoe N\n"
            _row = {"ngl": "32", "n_depth": "8192", "kv_type": "f16", "nkvo": "1"}

            # DM1/D2: no draft model => no draft column ANYWHERE. An inert
            # column would read as "draft placement doesn't matter" when it was
            # never tested, which is the failure the whole design guards against.
            _c0 = Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                         llama_server=_ls, array="L125", ctx_floor=8192,
                         fit_params=_fp, driver="server", hw=dict(_hw))
            _f0 = build_factors(_c0)
            assert not [k for k in _f0 if k.startswith("spec_draft_")], _f0
            assert "-md" not in build_server_args(_c0, _row, 8080, 9216)

            # with one: placement factors appear, and -md reaches the command.
            # This is the second route to MTP from issue #12 — the same head can
            # drive speculation from inside the target or as a separate model,
            # and only -md makes the latter expressible.
            _c1 = Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                         llama_server=_ls, array="L125", ctx_floor=8192,
                         fit_params=_fp, driver="server",
                         draft_model=Path("d.gguf"), hw=dict(_hw))
            _f1 = build_factors(_c1)
            assert "spec_draft_ngl" in _f1 and "spec_draft_kv" in _f1, _f1
            _cmd = build_server_args(_c1, _row, 8080, 9216)
            assert "-md" in _cmd and _cmd[_cmd.index("-md") + 1] == "d.gguf"
            assert factor_flags(_c1, {"spec_draft_ngl": "16"}, "server") == \
                [["-ngld", "16"]]
            # kv rides two flags, like the target's kv_type
            assert factor_flags(_c1, {"spec_draft_kv": "q4_0"}, "server") == \
                [["-ctkd", "q4_0"], ["-ctvd", "q4_0"]]

            # The draft KV is NOT held to --min-kv. That floor protects output
            # quality, and the drafter emits no output: a token drafted from a
            # degraded draft cache is verified by the target, then accepted or
            # discarded. Quantising it costs acceptance rate (speed), which is
            # the thing being measured — so the cheap end must stay reachable.
            assert "q4_0" in _f1["spec_draft_kv"], _f1["spec_draft_kv"]

            # DM3: llama-fit-params rejects -md and --mmproj, so it prices the
            # text model alone. We price the rest ourselves rather than giving
            # up on pruning — weights on disk are a hard LOWER bound on weights
            # in VRAM, and a lower bound is the only safe direction: it prunes
            # FEWER rows, so it can admit a config that will not fit (costing
            # time) but never delete one that would have (costing information).
            _tmp = Path(td_fit) / "d.gguf"
            _tmp.write_bytes(b"x" * (8 * 1024 * 1024))       # 8 MiB drafter
            _c2 = Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                         llama_server=_ls, array="L125", ctx_floor=8192,
                         fit_params=_fp, driver="server",
                         draft_model=_tmp, hw=dict(_hw))
            assert resident_extra_mib(_c0, _row) == 0        # nothing extra
            # not swept => llama.cpp offloads the whole drafter
            assert resident_extra_mib(_c2, _row) == 8
            # ...and placement scales it: -ngld 0 leaves it on the CPU
            assert resident_extra_mib(_c2, {**_row, "spec_draft_ngl": "0"}) == 0

            _proj = Path(td_fit) / "mmproj.gguf"
            _proj.write_bytes(b"x" * (4 * 1024 * 1024))      # 4 MiB projector
            _c3 = Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                         llama_server=_ls, array="L125", ctx_floor=8192,
                         fit_params=_fp, driver="server",
                         mmproj=_proj, hw=dict(_hw))
            assert resident_extra_mib(_c3, _row) == 4        # offloaded default
            assert resident_extra_mib(_c3, {**_row, "mmproj_offload": "0"}) == 0
            # both at once, and a missing file prices as 0 rather than raising
            _c4 = Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                         llama_server=_ls, array="L125", ctx_floor=8192,
                         fit_params=_fp, driver="server", draft_model=_tmp,
                         mmproj=_proj, hw=dict(_hw))
            assert resident_extra_mib(_c4, _row) == 12
            assert resident_extra_mib(
                Config(model=Path("t.gguf"), llama_bench=Path("lb"),
                       llama_server=_ls, array="L125", ctx_floor=8192,
                       mmproj=Path("/nonexistent.gguf"), hw=dict(_hw)),
                _row) == 0

            # the per-row footprint must reach the cache key, or two rows
            # differing only in placement inherit each other's OOM verdict —
            # the collision class that has bitten this pruner twice already
            assert _fit_cache_key(_c3, _row, "server") != \
                _fit_cache_key(_c3, {**_row, "mmproj_offload": "0"}, "server")
            assert _fit_cache_key(_c2, _row, "server") != \
                _fit_cache_key(_c2, {**_row, "spec_draft_ngl": "0"}, "server")
            # a projector is swept for placement only when one is loaded
            assert "mmproj_offload" in build_factors(_c3)
            assert "mmproj_offload" not in build_factors(_c0)
            assert "--mmproj" in build_server_args(_c3, _row, 8080, 9216)

            # draft levels come from the DRAFT's geometry, not the target's
            assert len(_f1["spec_draft_ngl"]) == 5
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)
            _td_fit.cleanup()
        assert draft_layer_count(None) is None
        assert draft_layer_count(Path("/nonexistent-draft.gguf")) is None

        # --- setup interview (Q1): answers -> argv is a pure function ---
        assert intent_args(Intent()) == []            # all defaults: say nothing
        assert intent_args(Intent(ctx=8192)) == ["--ctx-size", "8192"]
        assert intent_args(Intent(levels=3)) == ["--levels", "3"]
        assert intent_args(Intent(min_tgs=10.0)) == ["--min-tgs", "10"]
        assert intent_args(Intent(reps="quick")) == ["--quick"]
        assert intent_args(Intent(reps="full")) == ["--full"]
        assert intent_args(Intent(reps="standard")) == []     # the default
        assert intent_args(Intent(ctx=4096, levels=3, min_tgs=2.5,
                                  reps="quick")) == \
            ["--ctx-size", "4096", "--levels", "3", "--min-tgs", "2.5", "--quick"]

        # The interview must never fire where a prompt would hang something.
        _a = SimpleNamespace(model=Path("m.gguf"))
        _tty, sys.stdout.isatty = sys.stdout.isatty, lambda: False
        try:
            assert interview_wanted(_a, []) is None       # piped/redirected
        finally:
            sys.stdout.isatty = _tty
        assert interview_wanted(SimpleNamespace(model=None), []) is None
        # ...nor where the user has already said what they want. Asserted over
        # the whole list rather than a sample, so a flag added to _INTENT_FLAGS
        # without a matching guard cannot pass silently.
        _tty, sys.stdout.isatty = sys.stdout.isatty, lambda: True
        try:
            for flag in _INTENT_FLAGS:
                assert interview_wanted(_a, [flag]) is None, flag
                assert interview_wanted(_a, [flag + "=x"]) is None, flag
            # a bare model path on a TTY is the one case that DOES ask, so it
            # must not be reachable here without stubbing the prompts
            assert "--run" in _INTENT_FLAGS and "--factor" in _INTENT_FLAGS
        finally:
            sys.stdout.isatty = _tty

        # thin_to keeps the extremes: they are the interesting ends of a sweep
        assert thin_to([128, 256, 512, 1024, 2048], 3) == [128, 512, 2048]
        assert thin_to([128, 256, 512, 1024, 2048], 5) == \
            [128, 256, 512, 1024, 2048]
        assert thin_to(["0", "50", "100"], 2) == ["0", "100"]
        assert thin_to([1, 2], 5) == [1, 2]            # never invents levels

        # the cost dial: narrowing must reach a SMALLER array, which is the
        # whole point — trimming one factor at a time never did (choose_array
        # sizes on the widest, so an L125 stays an L125)
        assert n_levels_span(0, 64, 3) == [0, 32, 64]
        assert n_levels_span(0, 64, 5) == [0, 16, 32, 48, 64]
        assert len(ngl_levels(64, 3)) == 3 and len(ngl_levels(64, 5)) == 5
        assert len(depth_levels(32768, levels=3)) == 3
        assert len(cpu_offload_levels(64, 3)) == 3
        _five = {f"f{i}": ["a", "b", "c", "d", "e"] for i in range(7)}
        _three = {f"f{i}": ["a", "b", "c"] for i in range(7)}
        assert choose_array(_five) == "L125" and choose_array(_three) == "L27"

        # --- time budget for one config (T1) ---
        _c = SimpleNamespace(min_tgs=0.0, min_pps=0.0, slow_grace=60,
                             reps=3, n_gen=256)
        # no floor: --timeout is the budget, unchanged
        assert slow_budget_secs(_c, 1200) == 1200
        # a floor shortens it by arithmetic: 4 x 256 tokens at 2 t/s = 512s.
        # This is the whole point — a config that would MEET the floor finishes
        # inside it, so the deadline IS the test, and it works on llama-bench
        # where nothing can be observed mid-run.
        _c.min_tgs = 2.0
        assert slow_budget_secs(_c, 1200) == 512
        # never longer than --timeout: the floor tightens, never loosens
        _c.min_tgs = 0.1
        assert slow_budget_secs(_c, 1200) == 1200
        # ...and never shorter than the grace period, or a small n_gen would
        # derive a budget that measures model load rather than throughput
        _c.min_tgs, _c.n_gen = 1000.0, 8
        assert slow_budget_secs(_c, 1200) == 60
        _c.slow_grace = 10
        assert slow_budget_secs(_c, 1200) == 10

        # floors reject the unwanted, not the impossible — the numbers survive
        _f = SimpleNamespace(min_tgs=5.0, min_pps=100.0)
        assert too_slow_reason(_f, 500.0, 25.0) is None       # meets both
        assert "below --min-tgs" in too_slow_reason(_f, 500.0, 2.0)
        assert "below --min-pps" in too_slow_reason(_f, 50.0, 25.0)
        # a zero is "not measured", not "infinitely slow" — that is other
        # paths' business (measured_ok), and calling it SLOW would be a lie
        assert too_slow_reason(_f, 0.0, 25.0) is None
        assert too_slow_reason(_f, 500.0, 0.0) is None
        # no floors set: never fires
        assert too_slow_reason(SimpleNamespace(), 0.1, 0.1) is None

        # per-request timeout is what is LEFT of the config deadline, capped by
        # --timeout. This is the actual #11-adjacent defect: without a deadline
        # the server driver gave each of (1 warm + reps) requests the full
        # timeout, so a "20 minute" cap allowed 80 minutes.
        _now = time.time()
        assert _left(1200, None) == 1200                    # no deadline: as before
        assert _left(1200, _now + 300) <= 300
        assert _left(100, _now + 3000) == 100               # cap still applies
        assert _left(1200, _now - 5) == 1                   # exhausted, never 0
        assert _expired(_now - 1) and not _expired(_now + 60)
        assert not _expired(None)
        assert gpu_visibility(32752, "llama.cpp: ROCm0", True) == "gpu"
        # the vendor tool sees a card llama.cpp does not: no GPU backend
        assert gpu_visibility(32752, "rocm-smi", True) == "blind"
        assert gpu_visibility(24576, "nvidia-smi", True) == "blind"
        # nobody sees a GPU: a real CPU box, and a legitimate sweep
        assert gpu_visibility(None, "", True) == "cpu-only"
        # a binary too old to ask cannot be caught lying: silence is not evidence,
        # so this must NOT be reported as a broken build
        assert gpu_visibility(32752, "rocm-smi", False) == "unknown"
        assert gpu_visibility(None, "", False) == "unknown"
        # llama.cpp answering wins even when a vendor tool could have too
        assert gpu_visibility(30438, "llama.cpp: ROCm0", False) == "gpu"

        # --- provenance stamp (CHANGELOG "Affects existing results") ---
        assert parse_llama_version(
            "version: 0.3.0-dev (build 10636, commit 4d19b2876)\n"
            "built with Clang 22.0.0 for Linux x86_64") == "build 10636 (4d19b2876)"
        # the banner may be reworded; the build/commit pair is what identifies a run
        assert parse_llama_version(
            "llama.cpp  (build 42, commit abc1234)") == "build 42 (abc1234)"
        assert parse_llama_version("version: 0.3.0-dev (build 10636)") == ""
        assert parse_llama_version("error: invalid parameter: --version") == ""
        assert parse_llama_version("") == ""
        # a stamp is bookkeeping, never a factor column
        assert "tool_version" in RESULT_COLS and "llama_build" in RESULT_COLS
        assert __version__

        _out = io.StringIO()
        with contextlib.redirect_stdout(_out):
            warn_gpu_visibility("blind", "rocm-smi", {"ngl": [], "threads": []})
        _blind = _out.getvalue()
        assert "NO GPU devices" in _blind and "rocm-smi" in _blind, _blind
        assert "ngl" in _blind and "threads" not in _blind, _blind  # only the inert ones
        assert "CMakeCache" in _blind, _blind      # names the actual cause
        # a genuine CPU box gets a note, never the alarm
        _out = io.StringIO()
        with contextlib.redirect_stdout(_out):
            warn_gpu_visibility("cpu-only", "", {"ngl": [], "threads": []})
        _cpu = _out.getvalue()
        assert "!!" not in _cpu and "ngl" in _cpu, _cpu
        # and a CPU sweep with no GPU factors has nothing to warn about at all
        _out = io.StringIO()
        with contextlib.redirect_stdout(_out):
            warn_gpu_visibility("cpu-only", "", {"threads": [], "ubatch": []})
            warn_gpu_visibility("gpu", "llama.cpp: ROCm0", {"ngl": []})
            warn_gpu_visibility("unknown", "rocm-smi", {"ngl": []})
        assert _out.getvalue() == "", _out.getvalue()

        # --- measurement validity (docs/measurement-validity.md) ---
        # Issue #3: a server-driver sweep reported tg=1000000.0 t/s alongside
        # pp=444.1 and crowned it the winner on a box that really does ~25 t/s.
        # Repetition could not catch it -- the fault is deterministic, so
        # verify_picks confirmed it at "spread 0%". Reproducibility is not
        # validity, so the checks are about physical possibility instead.

        # I5, the causal check: a request cannot have produced tokens faster
        # than its own wall clock allows. 128 tokens over a 5s request permits
        # ~26 t/s no matter what the server's internal counter says.
        _rc = RepClock
        assert exceeds_wall_clock(1_000_000.0, [_rc(128, 5.0)])
        assert "wall clock" in exceeds_wall_clock(1_000_000.0, [_rc(128, 5.0)])
        # honest measurements survive, including a rate slightly above the
        # bound (wall time includes HTTP + tokenization, so it always
        # understates pure decode a little)
        assert exceeds_wall_clock(25.0, [_rc(128, 5.0)]) is None
        assert exceeds_wall_clock(30.0, [_rc(128, 5.0)]) is None   # inside margin
        # the kindest rep decides, so one slow round trip cannot condemn a run
        assert exceeds_wall_clock(50.0, [_rc(128, 60.0), _rc(128, 2.4)]) is None
        # nothing to compare against -> no verdict (never invent a rejection)
        assert exceeds_wall_clock(1_000_000.0, []) is None
        assert exceeds_wall_clock(0.0, [_rc(128, 5.0)]) is None

        # ...and the wall it is bounded by is the DECODE part of the request.
        # Issue #11's row, to its own arithmetic: a 23.5s request that spent
        # 17.6s re-prefilling 33,280 tokens and 5.9s generating 256. Against the
        # whole request that reads as 10.9 t/s and rejects an honest 43.1;
        # against the decode it actually claims, it passes.
        _agents = [_rc(256, 23.5, 17.6)]
        assert exceeds_wall_clock(43.1, [_rc(256, 23.5)]) is not None   # the bug
        assert exceeds_wall_clock(43.1, _agents) is None               # the fix
        # and the fix does not cost the check its teeth: issue #3's row still
        # goes, prefill credit or no prefill credit
        assert exceeds_wall_clock(1_000_000.0, _agents) is not None
        # the credit is capped, so a server claiming it spent the ENTIRE request
        # on prefill cannot switch off the check that distrusts it. 10x of
        # slack, against a defect that overshoots by ~1500x.
        _liar = [_rc(256, 23.5, 23.5)]
        assert abs(decode_wall(23.5, 23.5) - 23.5 * 0.1) < 1e-9
        assert exceeds_wall_clock(1_000_000.0, _liar) is not None
        assert exceeds_wall_clock(23.5, _liar) is None
        # a negative or absurd prompt_ms cannot tighten the bound either
        assert decode_wall(5.0, -100.0) == 5.0
        assert decode_wall(0.0, 1.0) == 0.0

        # per-rep screening: a rep is judged against its OWN request, so one
        # broken counter costs one rep rather than the configuration (#11)
        _good = RepSample(45.0, _rc(256, 23.5, 17.6))
        _liar_rep = RepSample(333_362.1, _rc(256, 23.5, 17.6))
        _kept, _gone = screen_reps([_good, _liar_rep, RepSample(46.0,
                                                               _rc(256, 23.5, 17.6))])
        assert [r.tg for r in _kept] == [45.0, 46.0], _kept
        assert [r.tg for r in _gone] == [333_362.1], _gone
        # a rep with no clock cannot be rejected, but it is kept rather than
        # treated as evidence: nothing about it has been checked
        _kept, _gone = screen_reps([RepSample(1_000_000.0, None)])
        assert _kept and not _gone
        # nothing survives -> the reason is built from what failed
        assert _rejected_reason([_liar_rep]) is not None
        assert "wall clock" in _rejected_reason([_liar_rep])
        assert _rejected_reason([RepSample(1e6, None)]) is None
        # the reason is stated against the MEDIAN of the rejected rates, not the
        # worst one: it should describe the number that would have been recorded,
        # and a max would overstate the overshoot in the text that survives
        _r2 = _rejected_reason([RepSample(1000.0, _rc(256, 23.5, 17.6)),
                                RepSample(2000.0, _rc(256, 23.5, 17.6)),
                                RepSample(9000.0, _rc(256, 23.5, 17.6))])
        assert "tg=2000.0" in _r2, _r2

        # the calibration probe must stay cheap -- it runs before every session's
        # first prompt, so it asks for one token, not a generation
        import inspect as _inspect
        _cal_src = _inspect.getsource(ServerSession.calibrate)
        assert "_completion(self.port, probe, 1," in _cal_src, _cal_src

        # what the prompt actually became: run through the model plus spared by
        # the cache. `n_depth` is only ever the request (#11).
        assert prompt_tokens_of({"prompt_n": 819, "cache_n": 7373}) == 8192
        assert prompt_tokens_of({"prompt_n": 8192}) == 8192
        assert prompt_tokens_of({}) == 0

        # delivered cache hit: what the server reused, not what we asked for
        assert delivered_cache_hit([{"cache_n": 7373, "prompt_n": 819}]) == 0.9
        assert delivered_cache_hit([{"cache_n": 0, "prompt_n": 8192}]) == 0.0
        # a server that does not report it leaves an unknown, never a zero --
        # "no cache_n" and "cache never hit" are different facts
        assert delivered_cache_hit([{"prompt_n": 8192}]) is None
        assert delivered_cache_hit([]) is None

        # a discarded row says so wherever a report is printed, not only in the
        # live sweep. --report-only is the one command a reporter can run
        # without a GPU, and it used to answer "why was this rejected" with
        # silence.
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            show_discards([{"run_id": 1, "implausible": "because reasons"},
                           {"run_id": 2}])
        assert "DISCARDED as impossible (1 of 2)" in _buf.getvalue()
        assert "run 1: because reasons" in _buf.getvalue()
        _quiet = io.StringIO()
        with contextlib.redirect_stdout(_quiet):
            show_discards([{"run_id": 1}])       # nothing thrown away, no noise
        assert _quiet.getvalue() == ""

        # I1: decode cannot outrun prefill. The reporter's exact row.
        assert implausible_reason(444.1, 1_000_000.0) is not None
        assert "decode cannot outrun prefill" in implausible_reason(444.1, 1_000_000.0)
        # normal rows: pp is 10-100x tg in prefill's favour
        assert implausible_reason(444.1, 25.0) is None
        assert implausible_reason(500.0, 120.0) is None
        # speculative decoding lifts tg by small factors, never past pp
        assert implausible_reason(300.0, 900.0) is None       # 3x, allowed
        # I2 backstop: absolute ceiling when prefill is unavailable
        assert implausible_reason(0.0, 1_000_000.0) is not None
        assert implausible_reason(0.0, 500.0) is None
        # ...scaled by concurrent streams, which legitimately aggregate
        assert implausible_reason(0.0, 150_000.0) is not None
        assert implausible_reason(0.0, 150_000.0, parallel=64) is None
        # a too-SLOW number is a real measurement and must survive
        assert implausible_reason(444.1, 0.2) is None

        # I3: rejection is total -- status flips and scores are zeroed, so a
        # consumer that reads the number before checking status still sees a
        # non-result rather than a record-breaking one.
        bad = validate_measurement({"status": "OK", "pp_tps": 444.1,
                                    "tg_tps": 1_000_000.0, "secs": 5.0})
        assert bad["status"] == "IMPLAUSIBLE", bad
        assert bad["tg_tps"] == 0.0 and bad["pp_tps"] == 0.0
        assert bad["implausible"]
        assert score_of({**bad, "eff_tps": 0.0}) == 0.0
        # and it is excluded everywhere status == "OK" gates
        assert pareto_frontier([bad]) == []

        # A run can COMPLETE and still generate nothing: implausible_reason
        # deliberately passes on tg <= 0, so status stays OK with a zero score.
        # That row must never be SELECTED — `longest` keys on depth first, so
        # without the guard the deepest zero row is recommended as the
        # max-context config: a command that loads and then produces nothing.
        cfg_pick = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                          ctx_floor=8192)
        _real = {"status": "OK", "n_depth": "8192", "pp_tps": 400.0,
                 "tg_tps": 25.0, "eff_tps": 25.0, "ngl": "32"}
        _empty = {"status": "OK", "n_depth": "65536", "pp_tps": 400.0,
                  "tg_tps": 0.0, "eff_tps": 0.0, "ngl": "32"}
        assert implausible_reason(400.0, 0.0) is None      # the reachable gap
        assert measured_ok([_real, _empty]) == [_real]
        _f, _b, _l = pick_recommendations(cfg_pick, [_real, _empty])
        assert _l is _real, _l          # deepest ZERO row must not win LONGEST
        assert _f is _real and _b is _real
        # and with nothing measurable at all, recommend nothing rather than a zero
        assert pick_recommendations(cfg_pick, [_empty]) == (None, None, None)
        assert pareto_frontier([_real, _empty]) == [_real]
        good = validate_measurement({"status": "OK", "pp_tps": 444.1,
                                     "tg_tps": 25.0, "secs": 5.0})
        assert good["status"] == "OK" and good["tg_tps"] == 25.0
        assert "implausible" not in good
        # a non-OK status is left alone -- OOM rows are not our business
        oom = validate_measurement({"status": "OOM", "pp_tps": 0.0, "tg_tps": 0.0})
        assert oom["status"] == "OOM"

        # I4/I5 wiring: the pure predicates above are worthless if the drivers
        # do not consult them, and that connection is what silently regresses.
        # Drive a fake server through the real measure() path -- one claiming
        # the reporter's 1e6 t/s, one honest -- and check the verdict reaches
        # the result dict. Sleeps are what give the wall clock something to
        # measure; keep them small.
        _real_completion = _completion
        try:
            def _fake(per_second, draft=None):
                def f(port, prompt, n_gen, timeout, cache=False):
                    time.sleep(0.02)
                    t = {"prompt_per_second": 444.1,
                         "predicted_per_second": per_second,
                         "predicted_n": 128, "prompt_n": 8192}
                    t.update(draft or {})     # llama.cpp adds these only when
                    return {"content": "x", "timings": t}   # a draft ran (F1)
                return f

            sess = ServerSession.__new__(ServerSession)   # no server launched
            sess.port, sess.ok, sess.err = 1, True, ""
            sess.cfg = SimpleNamespace(prefix_reuse=1.0)  # historical shape
            fake_cfg = SimpleNamespace(n_prompt=0, n_gen=128, parallel=1, reps=1,
                                       measure_vram=False, factors={},
                                       emit_mtp=False, ngram=False, hw={})

            globals()["_completion"] = _fake(1_000_000.0)
            r_bad = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_bad["status"] == "IMPLAUSIBLE", r_bad
            assert r_bad["tg_tps"] == 0.0 and r_bad["implausible"]

            globals()["_completion"] = _fake(600.0)       # fast but possible
            r_ok = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_ok["status"] == "OK", r_ok
            assert r_ok["tg_tps"] == 600.0
            assert "draft_acc" not in r_ok and "spec_off" not in r_ok, r_ok

            # --- the `agents` request shape reaches measure() (issue #11) ---
            # Every case above runs at reuse 1.0, the pre-4ffa97a shape where
            # each rep is a full cache hit and `wall` is decode plus a round
            # trip. The profiles have not defaulted to that since 4ffa97a, and
            # nothing here noticed: at reuse < 1.0 the reps send DIFFERENT
            # prompts, so each one re-prefills its differing suffix inside the
            # request that I5 then bounds by wall time. Pin the shape the reps
            # actually send, so the request I5 is reasoning about is the request
            # it gets. (Whether the bound itself is still right for this shape
            # is issue #11 and is not settled here.)
            seen = []

            def _recording(port, prompt, n_gen, timeout, cache=False):
                seen.append(prompt)
                time.sleep(0.02)
                # the warm request prefills the lot; the reps hit the cache for
                # all but their differing suffix. Split that way rather than
                # flat, because `prompt_n` and `cache_n` sum to the prompt and a
                # fake where they do not cannot exercise the hit ratio at all
                # (server-common.cpp server_slot_stats::to_json)
                warm = len(seen) == 1
                return {"content": "x",
                        "timings": {"prompt_per_second": 444.1,
                                    "predicted_per_second": 600.0,
                                    "predicted_n": 128,
                                    "prompt_n": 8192 if warm else 819,
                                    "prompt_ms": 18000.0 if warm else 1800.0,
                                    "cache_n": 0 if warm else 7373}}

            sess.cfg = SimpleNamespace(prefix_reuse=0.9)      # the agents shape
            globals()["_completion"] = _recording
            r_shape = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert len(seen) == 2, seen                       # warm + one rep
            assert seen[0] != seen[1], "reps re-use the warm prompt"
            assert 0.85 <= achieved_reuse(seen) <= 0.95, achieved_reuse(seen)
            assert 0.85 <= float(r_shape["reuse"]) <= 0.95, r_shape
            # the server's delivered reuse rides along with the requested one
            assert r_shape["cache_hit"] == 0.9, r_shape

            # --- and the bound is right for that shape (issue #11) ---
            # A request that is mostly prefill: 0.35s on the wall, 0.30s of it
            # spent re-prefilling by the server's own account, 128 tokens
            # generated in the rest. Against the WHOLE request that permits
            # ~366 t/s and an honest 1500 reads as impossible; against the
            # decode it is actually a claim about, it is fine.
            def _prefill_heavy(per_second, prompt_ms=300.0, pps=444.1):
                def f(port, prompt, n_gen, timeout, cache=False):
                    time.sleep(0.35)
                    t = {"prompt_per_second": pps,
                         "predicted_per_second": per_second,
                         "predicted_n": 128, "prompt_n": 8192, "cache_n": 0}
                    if prompt_ms is not None:
                        t["prompt_ms"] = prompt_ms
                    return {"content": "x", "timings": t}
                return f

            # reuse 0.0 — the reporter's own no-profile run, where every rep is
            # MEANT to re-prefill and the cache hit of 0 is correct, not a fault
            sess.cfg = SimpleNamespace(prefix_reuse=0.0)
            globals()["_completion"] = _prefill_heavy(1500.0)
            r_pre = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_pre["status"] == "OK", r_pre      # was IMPLAUSIBLE (#11)
            assert r_pre["tg_tps"] == 1500.0, r_pre
            assert r_pre["cache_hit"] == 0.0, r_pre   # nothing was reused

            # crediting prefill must not buy the server an escape: issue #3's
            # row is still impossible against the decode window
            globals()["_completion"] = _prefill_heavy(1_000_000.0)
            r_liar = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_liar["status"] == "IMPLAUSIBLE", r_liar
            # the reason carries the breakdown AND what was measured, because
            # the rejection zeroes the only other record of it
            assert "prefill" in r_liar["implausible"], r_liar
            assert "measured pp=" in r_liar["implausible"], r_liar

            # a server too old to report prompt_ms falls back to pricing the
            # prefill client-side, off the warm request's own pp. Same shape,
            # same verdict — 8192 tokens at 27306 t/s is the same 0.30s.
            globals()["_completion"] = _prefill_heavy(1500.0, prompt_ms=None,
                                                      pps=27306.0)
            r_old = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_old["status"] == "OK", r_old
            globals()["_completion"] = _prefill_heavy(1_000_000.0, prompt_ms=None,
                                                      pps=27306.0)
            assert measure_in_session(fake_cfg, {"n_depth": "8192"},
                                      sess, 30)["status"] == "IMPLAUSIBLE"

            # --- one bad rep is a bad rep, not a bad configuration (#11) ---
            # The reporter's last run: three reps, one of them reporting
            # 333,362 t/s and the others ~45. Averaged, the outlier decided the
            # verdict and the whole row was zeroed; and because the reason was
            # built from the mean and the kindest rep, the surviving text could
            # not say whether one rep had gone bad or all three had.
            _seq = {"i": 0}

            def _per_rep(rates):
                def f(port, prompt, n_gen, timeout, cache=False):
                    time.sleep(0.05)
                    i = _seq["i"]
                    _seq["i"] += 1
                    # index 0 is the warm request; the reps follow it
                    rate = 600.0 if i == 0 else rates[(i - 1) % len(rates)]
                    return {"content": "x",
                            "timings": {"prompt_per_second": 444.1,
                                        "predicted_per_second": rate,
                                        "predicted_n": 128, "prompt_n": 8192,
                                        "cache_n": 0, "prompt_ms": 1.0}}
                return f

            rep_cfg = SimpleNamespace(n_prompt=0, n_gen=128, parallel=1, reps=3,
                                      measure_vram=False, factors={},
                                      emit_mtp=False, ngram=False, hw={})
            _seq["i"] = 0
            globals()["_completion"] = _per_rep([600.0, 333_362.1, 620.0])
            _rbuf = io.StringIO()
            with contextlib.redirect_stdout(_rbuf):
                r_one = measure_in_session(rep_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_one["status"] == "OK", r_one       # was IMPLAUSIBLE
            assert r_one["rejected_reps"] == 1, r_one
            # ...and the survivors decide the number: the median of 600 and 620,
            # with the impossible rep contributing nothing to it
            assert r_one["tg_tps"] == 610.0, r_one

            # the aggregate is the MEDIAN of the survivors, not their mean: with
            # three believable but skewed reps the two disagree, and the mean
            # walks toward whichever rep was furthest out. 600/610/900 are all
            # possible against their own clocks, so none is screened -- this is
            # the aggregation rule alone, with no rejection involved.
            _seq["i"] = 0
            globals()["_completion"] = _per_rep([600.0, 610.0, 900.0])
            r_med = measure_in_session(rep_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_med["rejected_reps"] == 0, r_med      # nothing rejected
            assert r_med["tg_tps"] == 610.0, r_med         # median, not 703.3

            # every rep impossible is still a rejected configuration, and the
            # reason still travels with it
            _seq["i"] = 0
            globals()["_completion"] = _per_rep([1_000_000.0])
            _abuf = io.StringIO()
            with contextlib.redirect_stdout(_abuf):
                r_all = measure_in_session(rep_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_all["status"] == "IMPLAUSIBLE", r_all
            assert r_all["rejected_reps"] == 3, r_all
            assert "wall clock" in r_all["implausible"], r_all

            # a partial rejection is not allowed to be silent: the row keeps its
            # numbers, which is exactly when a discarded sample would vanish
            assert "1 of 3 reps" in _rbuf.getvalue(), _rbuf.getvalue()
            # a wholly rejected row says so as IMPLAUSIBLE instead; the per-rep
            # notice is for the case where numbers were kept
            assert "reps reported" not in _abuf.getvalue(), _abuf.getvalue()

            # --- the depth that was measured, not the depth that was asked for
            # (#11). The prompt is sized in characters against an assumed 4
            # chars/token; on the reporter's model the real ratio was 6.05, so a
            # row labelled n_depth=32768 had run about 27,000 tokens.
            assert r_one["prompt_tok"] == 8192, r_one

            # and the ratio is measured BEFORE the first prompt is built, not
            # after — on a single-config run the warm request's calibration
            # arrived too late to size anything at all.
            _sizes = []

            def _tokenizer(chars_per_token):
                def f(port, prompt, n_gen, timeout, cache=False):
                    _sizes.append(len(prompt))
                    time.sleep(0.01)
                    return {"content": "x",
                            "timings": {"prompt_per_second": 444.1,
                                        "predicted_per_second": 600.0,
                                        "predicted_n": 128, "cache_n": 0,
                                        "prompt_n": int(len(prompt)
                                                        / chars_per_token)}}
                return f

            fresh = ServerSession.__new__(ServerSession)  # never calibrated
            fresh.port, fresh.ok, fresh.err = 1, True, ""
            # reuse 0.0: a 0% cache hit is the correct outcome for this shape,
            # so the miss warning stays out of the way of what is being tested
            fresh.cfg = SimpleNamespace(prefix_reuse=0.0)
            globals()["_completion"] = _tokenizer(6.05)
            r_cal = measure_in_session(fake_cfg, {"n_depth": "8192"}, fresh, 30)
            assert abs(fresh.cpt - 6.05) < 0.05, fresh.cpt
            # probe first, then a battery sized at the MEASURED ratio: 8192
            # tokens of prompt is ~49.6k characters, not 32.8k
            assert _sizes[0] == ServerSession.CALIBRATION_CHARS, _sizes
            assert abs(_sizes[1] - 8192 * 6.05) < 8192, _sizes
            assert abs(r_cal["prompt_tok"] - 8192) < 100, r_cal
            # one probe per session, whatever it returned
            _sizes.clear()
            measure_in_session(fake_cfg, {"n_depth": "8192"}, fresh, 30)
            assert _sizes[0] != ServerSession.CALIBRATION_CHARS, _sizes

            # The warning, which is the other half of the fix: a rep that was
            # supposed to decode off a cache hit and instead re-prefilled is a
            # real measurement of the WRONG workload, so it is said out loud
            # rather than rejected. Gated on the requested shape — reuse 0.0
            # above expects a 0% hit and must stay quiet.
            def _said(want, hit):
                s = ServerSession.__new__(ServerSession)
                s.cfg = SimpleNamespace(prefix_reuse=want)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    s._warn_cache_miss(hit)
                    s._warn_cache_miss(hit)      # said once, not once per config
                return buf.getvalue()

            assert "delivered 0% reuse" in _said(0.9, 0.0)
            assert _said(0.9, 0.0).count("delivered") == 1
            assert _said(0.0, 0.0) == ""          # asked for nothing, got nothing
            assert _said(0.9, 0.88) == ""         # delivered what it asked for
            assert _said(0.9, None) == ""         # server does not say -> silent
            sess.cfg = SimpleNamespace(prefix_reuse=1.0)      # restore

            # Same wiring for the speculative telemetry (F1): the counters must
            # survive the trip from the response through measure() into the row.
            spec_cfg = SimpleNamespace(n_prompt=0, n_gen=128, parallel=1, reps=1,
                                       measure_vram=False, factors={},
                                       emit_mtp=True, ngram=False,
                                       hw={"n_nextn": 1})
            globals()["_completion"] = _fake(600.0, {"draft_n": 40,
                                                     "draft_n_accepted": 26})
            r_spec = measure_in_session(spec_cfg, {"mtp": "1"}, sess, 30)
            assert r_spec["status"] == "OK", r_spec
            assert r_spec["draft_acc"] == 0.65, r_spec
            assert "spec_off" not in r_spec, r_spec

            # and the guard: MTP asked for, no draft counters in any response
            globals()["_completion"] = _fake(600.0)
            r_off = measure_in_session(spec_cfg, {"mtp": "1"}, sess, 30)
            assert r_off["spec_off"] == 1, r_off
            assert r_off["status"] == "OK" and r_off["tg_tps"] == 600.0, r_off
            assert "draft_acc" not in r_off, r_off

            # --- partial failure is a measurement, not an ERROR (F7) ---
            # One bad request out of `par` used to raise out of the whole round,
            # so "7 of 8 served quickly" was recorded identically to "the model
            # would not load". Those are different configs to deploy.
            _calls = {"n": 0}

            def _flaky(port, prompt, n_gen, timeout, cache=False):
                _calls["n"] += 1
                if _calls["n"] % 4 == 0:          # every 4th request fails
                    raise OSError("simulated request failure")
                time.sleep(0.01)
                return {"content": "x",
                        "timings": {"prompt_per_second": 400.0,
                                    "predicted_per_second": 50.0,
                                    "predicted_n": 64, "prompt_n": 128}}

            par_cfg = SimpleNamespace(n_prompt=0, n_gen=64, parallel=4, reps=1,
                                      measure_vram=False, factors={},
                                      emit_mtp=False, ngram=False, hw={})
            globals()["_completion"] = _flaky
            _calls["n"] = 0
            r_part = measure_in_session(par_cfg, {"n_depth": "0"}, sess, 30)
            assert r_part["status"] == "OK", r_part      # NOT collapsed to ERROR
            assert 0.0 < r_part["err_rate"] < 1.0, r_part
            assert r_part["tg_tps"] > 0, r_part          # the survivors measured
            # throughput is whole-round, so lost work shows up as lower tg
            # rather than needing a separate penalty
            assert error_note(r_part), r_part

            # every request failing IS an error: there is no measurement
            def _dead(port, prompt, n_gen, timeout, cache=False):
                raise OSError("server gone")
            globals()["_completion"] = _dead
            r_dead = measure_in_session(par_cfg, {"n_depth": "0"}, sess, 30)
            assert r_dead["status"] == "ERROR", r_dead
            # a clean config reports no error rate and gets no warning
            globals()["_completion"] = _fake(600.0)
            r_clean = measure_in_session(par_cfg, {"n_depth": "0"}, sess, 30)
            assert r_clean["err_rate"] == 0.0, r_clean
            assert error_note(r_clean) is None
            assert error_note({}) is None and error_note({"err_rate": ""}) is None

            # an error payload is a failure, not a zero-time measurement, and
            # the raise it produces is what measure_in_session turns into ERROR
            def _err(port, prompt, n_gen, timeout, cache=False):
                return raise_for_server_error(
                    {"error": {"message": "context shift is disabled"}})
            globals()["_completion"] = _err
            r_e = measure_in_session(fake_cfg, {"n_depth": "8192"}, sess, 30)
            assert r_e["status"] == "ERROR", r_e
        finally:
            globals()["_completion"] = _real_completion

        # --- prompt battery and prefix reuse (docs/workload-shape-design.md) ---
        # reuse 1.0 must reproduce the historical behaviour EXACTLY: every
        # request byte-identical. Anything else silently redefines every
        # measurement this tool has ever produced.
        same = prompt_battery(256, 5, reuse=1.0)
        assert len(set(same)) == 1, "reuse 1.0 must give identical requests"
        assert achieved_reuse(same) == 1.0
        # reuse 0.0: nothing shared by construction
        none_shared = prompt_battery(256, 6, reuse=0.0)
        assert len(set(none_shared)) > 1, none_shared
        assert achieved_reuse(none_shared) < 0.1, achieved_reuse(none_shared)
        # partial reuse: the shared part is shared CONTENT, not a shared length
        # (W-D3) -- llama.cpp matches on tokens, so a per-request "prefix" would
        # reuse nothing while looking like it should
        part = prompt_battery(400, 8, reuse=0.5)
        assert len(set(part)) > 1, "suffixes must differ"
        pre = part[0][:len(part[0]) // 2]
        assert all(x.startswith(pre) for x in part), "prefix must be shared"
        got = achieved_reuse(part)
        assert 0.4 <= got <= 0.6, got
        # achieved is MEASURED, not echoed back: it reads the prompts we built
        assert achieved_reuse(["abcdef", "abcxyz"]) == round(3 / 6, 4)
        assert achieved_reuse([]) == 0.0 and achieved_reuse(["only"]) == 1.0
        # deterministic for a given seed, so two sweeps are comparable
        assert prompt_battery(256, 4, 0.25) == prompt_battery(256, 4, 0.25)
        # every prompt is about the requested size regardless of reuse
        for _re in (0.0, 0.5, 1.0):
            for _p in prompt_battery(300, 4, _re):
                assert abs(len(_p) - 300 * CHARS_PER_TOKEN) <= CHARS_PER_TOKEN, len(_p)
        # the category mix is drawn from the banks, and the weights sum to 1
        assert abs(sum(w for _, w, _ in PROMPT_BANKS) - 1.0) < 1e-9

        # --- constants that were ours, now derived (docs/constants-audit.md) ---
        # C-B: chars-per-token is measured from llama.cpp's own prompt_n, not
        # assumed to be 4. Getting this wrong means the sweep tests a different
        # n_depth than the column claims, on any model whose tokenizer is not
        # English prose.
        assert calibrate_chars_per_token("x" * 400, 100) == 4.0
        assert calibrate_chars_per_token("x" * 250, 100) == 2.5   # code-ish
        assert calibrate_chars_per_token("x" * 900, 100) == 9.0   # CJK-ish
        # a ratio no real tokenizer produces is a malformed response, not a
        # surprising model: rejecting it matters more than adopting it, because
        # it would size every later prompt
        assert calibrate_chars_per_token("x" * 100, 200) is None  # 0.5
        assert calibrate_chars_per_token("x" * 5000, 100) is None  # 50
        assert calibrate_chars_per_token("x" * 400, 0) is None
        assert calibrate_chars_per_token("x" * 400, None) is None
        assert calibrate_chars_per_token("", 10) is None
        assert calibrate_chars_per_token("x" * 400, "bad") is None
        # a measured ratio actually resizes the battery
        wide = prompt_battery(100, 2, reuse=0.0, chars_per_token=9.0)
        assert all(abs(len(x) - 900) <= 9 for x in wide), [len(x) for x in wide]
        narrow = prompt_battery(100, 2, reuse=0.0, chars_per_token=2.0)
        assert all(abs(len(x) - 200) <= 2 for x in narrow), [len(x) for x in narrow]
        # and None falls back to the bootstrap constant
        assert len(prompt_battery(100, 1, 0.0)[0]) == 100 * CHARS_PER_TOKEN

        # Profile-level reuse defaults. 0.0 everywhere except `agents`, whose
        # name is itself a claim about traffic (a fixed system preamble in front
        # of long tool-use prompts). Asymmetric failure modes decide the rest:
        # overstating speculation is invisible and ships a config that will not
        # deliver, understating it is visible and recoverable.
        assert PROFILES["single"]["prefix_reuse"] == 0.0
        assert PROFILES["multi"]["prefix_reuse"] == 0.0
        assert PROFILES["agents"]["prefix_reuse"] == 0.9

        # --- concurrency / kv_unified (docs/concurrency-kv-design.md) ---
        # llama.cpp couples these: auto means 4 slots AND unified KV, and ANY
        # explicit --parallel disables unified -- including --parallel 1, because
        # the branch tests < 0. One categorical over the states that exist.
        assert concurrency_spec("auto") == (4, True)
        assert concurrency_spec("8") == (8, False)
        assert concurrency_spec("8u") == (8, True)
        assert concurrency_spec("1") == (1, False)      # NOT unified
        assert concurrency_spec("nonsense") == (1, False)
        # KV3: auto must be reachable, and it is the level that emits nothing
        assert concurrency_flags("auto") == []
        assert concurrency_flags("8") == ["--parallel", "8"]
        assert concurrency_flags("8u") == ["--parallel", "8", "--kv-unified"]
        # K3: ask for slots x per-slot context ONLY when the cache is split.
        # A slot gets the full n_ctx under unified KV and n_ctx/slots when split
        # (src/llama-context.cpp), so inverting this silently gives every row a
        # different real context than its ctx_floor claims.
        assert ctx_slots_multiplier("auto") == 1        # unified: no division
        assert ctx_slots_multiplier("8") == 8           # split: divided by 8
        assert ctx_slots_multiplier("8u") == 1
        assert ctx_slots_multiplier("1") == 1
        # KV1: no thresholds -- the level decides, not a `> 1` test
        cfg_cc = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                        ctx_floor=8192, driver="server", parallel=1,
                        hw={"phys": 8, "logical": 16, "n_layers": 32,
                            "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0})
        for lvl, want_np in (("auto", False), ("1", True), ("4", True), ("4u", True)):
            a = build_server_args(cfg_cc, {"concurrency": lvl, "ubatch": "512"},
                                  8080, 4096)
            assert ("--parallel" in a) is want_np, (lvl, a)
            assert ("--kv-unified" in a) is concurrency_spec(lvl)[1] or lvl == "auto", (lvl, a)
        # KV2: the recorded regime matches the flags for every level
        for lvl in ("auto", "1", "4", "4u", "8u"):
            a = build_server_args(cfg_cc, {"concurrency": lvl, "ubatch": "512"},
                                  8080, 4096)
            emitted_unified = "--kv-unified" in a or "--parallel" not in a
            assert kv_unified_for(cfg_cc, {"concurrency": lvl}) == emitted_unified, lvl
        # the older `parallel` factor keeps working, and always splits the cache
        assert kv_unified_for(cfg_cc, {"parallel": "4"}) is False
        assert kv_unified_for(cfg_cc, {"parallel": "1"}) is False
        assert slots_for(cfg_cc, {"parallel": "6"}) == 6
        assert slots_for(cfg_cc, {"concurrency": "6u"}) == 6
        assert slots_for(cfg_cc, {}) == 1
        # nothing emitted at all -> llama.cpp picks auto -> unified
        assert kv_unified_for(cfg_cc, {}) is True
        for _n, _p in PROFILES.items():
            assert 0.0 <= _p["prefix_reuse"] <= 1.0, _n

        # C-F: the sweep estimate must scale with model size, not be flat
        _big = Config(model=Path("/nonexistent-big.gguf"), llama_bench=Path("b"),
                      array="auto", ctx_floor=8192, n_gen=256, reps=3)
        assert estimate_secs_per_run(_big) > 8.0        # missing file: no crash
        class _Sz:
            def __init__(self, n): self.n = n
            def stat(self): return SimpleNamespace(st_size=self.n)
        small = Config(model=_Sz(300 * 1024**2), llama_bench=Path("b"),
                       array="auto", ctx_floor=8192, n_gen=256, reps=3)
        big = Config(model=_Sz(25 * 1024**3), llama_bench=Path("b"),
                     array="auto", ctx_floor=8192, n_gen=256, reps=3)
        assert estimate_secs_per_run(big) > 3 * estimate_secs_per_run(small), (
            estimate_secs_per_run(big), estimate_secs_per_run(small))
        # more reps and more generated tokens both cost time
        more = Config(model=_Sz(300 * 1024**2), llama_bench=Path("b"),
                      array="auto", ctx_floor=8192, n_gen=256, reps=5)
        assert estimate_secs_per_run(more) > estimate_secs_per_run(small)
        assert {n for n, _, _ in PROMPT_BANKS} == {
            "simple_qa", "reasoning", "code", "rag", "long_ctx"}

        # The bench driver needs the same wiring as the server driver above.
        # It reaches validate_measurement by a different route, and llama-bench
        # has the same exposure -- it divides the NOMINAL n_prompt + n_gen by
        # measured time (llama-bench.cpp:1529), guarded upstream only since May
        # 2025, so older builds still report a huge rate for a decode that did
        # nothing. Drive run_one over a faked subprocess to prove the gate is
        # actually on this path too.
        _real_run = subprocess.run

        class _Proc:
            def __init__(self, out):
                self.returncode, self.stdout, self.stderr = 0, out, ""

        def _bench_json(tg):
            return json.dumps([
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 444.1},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": tg},
            ])
        try:
            bench_cfg = Config(model=Path("m.gguf"), llama_bench=Path("llama-bench"),
                               array="auto", ctx_floor=8192, driver="bench",
                               factors={}, hw={"phys": 8, "logical": 16})
            row = {"ngl": "20", "n_depth": "8192", "kv_type": "f16",
                   "ubatch": "1024", "threads": "16"}

            subprocess.run = lambda *a, **k: _Proc(_bench_json(1_000_000.0))
            rb = run_one(bench_cfg, row, 60)
            assert rb["status"] == "IMPLAUSIBLE", rb
            assert rb["tg_tps"] == 0.0 and rb["implausible"]

            subprocess.run = lambda *a, **k: _Proc(_bench_json(25.0))
            rg = run_one(bench_cfg, row, 60)
            assert rg["status"] == "OK" and rg["tg_tps"] == 25.0, rg
        finally:
            subprocess.run = _real_run

        # --- CSV round-trip: rejection must survive, and repair old files ---
        # --report-only never re-measures, so a CSV written before the gate
        # existed would keep crowning its bogus row forever. Re-validating at
        # load is the only thing that can fix issue #3's own results file.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.csv"
            p.write_text(
                "run_id,ngl,n_depth,parallel,pp_tps,tg_tps,eff_tps,status,secs\n"
                "1,20,8192,1,444.1,1000000.0,1000000.0,OK,5.0\n"   # legacy bogus
                "2,20,8192,1,444.1,25.0,25.0,OK,5.0\n"             # honest
                "3,20,8192,1,0.0,0.0,0.0,OOM,0.0\n")               # untouched
            got = load_results_csv(p, {})
            assert got[0]["status"] == "IMPLAUSIBLE", got[0]
            assert got[0]["tg_tps"] == 0.0 and got[0]["eff_tps"] == 0.0
            assert got[1]["status"] == "OK" and got[1]["tg_tps"] == 25.0
            assert got[2]["status"] == "OOM"
            # the repaired row must not win anything
            assert pareto_frontier(got) and all(
                r["status"] == "OK" for r in pareto_frontier(got))
            assert max(got, key=score_of)["tg_tps"] == 25.0   # honest row wins

            # a legitimately high aggregate under --parallel must survive: the
            # ceiling scales with the streams recorded in the row
            p2 = Path(td) / "r2.csv"
            p2.write_text(
                "run_id,ngl,parallel,pp_tps,tg_tps,eff_tps,status,secs\n"
                "1,20,64,900000.0,150000.0,150000.0,OK,5.0\n")
            assert load_results_csv(p2, {})[0]["status"] == "OK"

        # the error check itself, independent of any transport
        try:
            raise_for_server_error({"error": {"message": "boom"}})
            assert False, "error payload was accepted as a result"
        except OSError as e:
            assert "boom" in str(e)
        # a bare-string error, and the shapes that must pass through untouched
        try:
            raise_for_server_error({"error": "kv cache full"})
            assert False, "string error payload was accepted"
        except OSError:
            pass
        ok_body = {"content": "hi", "timings": {"predicted_per_second": 30.0}}
        assert raise_for_server_error(ok_body) is ok_body
        assert raise_for_server_error({"error": None}) == {"error": None}

        # ttft note: prefill-cost estimate on emitted commands
        n = ttft_note({"pp_tps": 100.0}, 235520)
        assert n and "39m" in n and "8k prompt" in n
        assert "8k prompt" not in ttft_note({"pp_tps": 100.0}, 8192)
        assert ttft_note({"pp_tps": 0}, 8192) is None

        # sidecar naming (probe / verify persist next to the results CSV)
        assert probe_sidecar(Path("r/x.csv")).name == "x.csv.probe.json"
        assert verify_sidecar(Path("r/x.csv")).name == "x.csv.verify.json"

        # pick verification: medians overwrite the row, keyed by config
        cfg_pv = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                        ctx_floor=8192,
                        factors={"ngl": ["58"], "n_depth": ["32768"],
                                 "threads": ["5"], "kv_type": ["q8_0"],
                                 "ubatch": ["512"]})
        pv_rows = [{"run_id": "1", "ngl": "58", "n_depth": "32768",
                    "threads": "5", "kv_type": "q8_0", "ubatch": "512",
                    "status": "OK", "pp_tps": 120.0, "tg_tps": 10.8,
                    "eff_tps": 10.8}]
        apply_verification(cfg_pv, pv_rows,
                           {config_key(cfg_pv, pv_rows[0]):
                            {"tg_tps": 8.9, "pp_tps": 118.0,
                             "n": 3, "spread_pct": 25.0}})
        assert pv_rows[0]["tg_tps"] == 8.9 and pv_rows[0]["verify_n"] == 3
        assert pv_rows[0]["eff_tps"] == 8.9      # re-scored under --score tg

        # probe + verification render in the terminal report and the HTML card
        pv_probe = {"row": pv_rows[0], "depth": 262144, "tg_tps": 6.6,
                    "safe_ctx": 235520, "at_cap": True}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(cfg_pv, pv_rows, pv_probe)
        rout = buf.getvalue()
        assert "PROBED CEILING" in rout and "-c 235520" in rout
        assert "median of 3 measurements" in rout and "prefill cost" in rout
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "r.html"
            with contextlib.redirect_stdout(io.StringIO()):
                write_html_report(cfg_pv, pv_rows, hp, pv_probe)
                h = hp.read_text()
                assert "Probed ceiling" in h and "262,144" in h
                write_html_report(cfg_pv, pv_rows, hp)
            assert "Probed ceiling" not in hp.read_text()
    except AssertionError as e:
        print(f"selftest FAILED: {e}")
        return False

    # --- OOM pruning: the fit verdict itself ---
    # Both sides of the comparison had gone untested: the headroom subtracted
    # from capacity, and the artifacts fit-params cannot be told about. Each is
    # a silent admit-or-delete decision on every row, and `full_offload_fits`
    # now rides on the same verdict to shape the ngl grid.
    with tempfile.TemporaryDirectory() as _td:
        _fp = Path(_td) / "llama-fit-params"
        _fp.write_text("#!/bin/sh\n")
        cfg_fit = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                         llama_server=Path("s"), array="auto", ctx_floor=8192,
                         driver="server", fit_params=_fp,
                         fit_headroom_mib=512,
                         hw={"phys": 8, "logical": 16, "n_layers": 32,
                             "vram": 10000, "n_ctx_train": 32768,
                             "n_experts": 0, "n_nextn": 0})
        _real_sub, _real_parse = subprocess, parse_fit_print
        _reported = {"mib": 0}
        try:
            globals()["subprocess"] = SimpleNamespace(
                run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=""),
                TimeoutExpired=_real_sub.TimeoutExpired)
            globals()["parse_fit_print"] = lambda _out: _reported["mib"]

            # capacity is total VRAM MINUS the headroom, not total VRAM: 9600
            # fits under 10000-512=9488? no -- and that is the whole point
            _fit_cache.clear(); _reported["mib"] = 9400
            assert predict_fits(cfg_fit, {"ngl": "32"}, "server") is True
            _fit_cache.clear(); _reported["mib"] = 9600
            assert predict_fits(cfg_fit, {"ngl": "32"}, "server") is False, \
                "headroom is not being subtracted from capacity"

            # and a resident draft model counts against the same budget, since
            # fit-params rejects -md and cannot see it (issue #13)
            _fit_cache.clear(); _reported["mib"] = 9400
            cfg_md = replace(cfg_fit, draft_model=Path(_td) / "d.gguf")
            (Path(_td) / "d.gguf").write_bytes(b"x" * (400 * 1024 * 1024))
            assert predict_fits(cfg_md, {"ngl": "32"}, "server") is False, \
                "artifacts fit-params cannot see are not being priced"
        finally:
            globals()["subprocess"] = _real_sub
            globals()["parse_fit_print"] = _real_parse
            _fit_cache.clear()

    # --- OOM pruning: fit-params flag generation ---
    try:
        class _FC:                        # minimal Config stand-in
            model = Path("m.gguf")
            fit_params = Path("fp")
        f = {"ngl": "32", "n_depth": "32768", "kv_type": "q4_0", "nkvo": "1"}
        bench_flags = _fit_params_flags(_FC, f, "bench")
        assert "-ngl" in bench_flags and "32" in bench_flags
        assert "-c" in bench_flags and "32768" in bench_flags
        assert "-ctk" in bench_flags and "q4_0" in bench_flags
        # nkvo=1 is the level that puts the KV in SYSTEM RAM, and
        # --no-kv-offload is llama.cpp's other name for the flag the drivers
        # emit for it. This assertion was the wrong way round, and with it the
        # estimator priced the opposite placement from the one that would run:
        # measured 32666 MiB vs 26513 on gemma-4-31B at -c 65536 -ctk f16.
        assert "--no-kv-offload" in bench_flags
        server_flags = _fit_params_flags(_FC, f, "server")
        assert "--no-kv-offload" in server_flags  # same for both drivers

        # nkvo=0 keeps the KV on the GPU, so nothing is emitted and the estimate
        # carries the context buffer
        f2 = {"ngl": "32", "n_depth": "8192", "kv_type": "f16", "nkvo": "0"}
        assert "--no-kv-offload" not in _fit_params_flags(_FC, f2, "bench")
        assert "-nkvo" not in _fit_params_flags(_FC, f2, "bench")
        # a row with no nkvo column at all: the drivers emit nothing, so
        # llama.cpp's default applies (KV offload ENABLED, cache on the GPU) and
        # the estimator must price that, not its opposite
        assert "--no-kv-offload" not in _fit_params_flags(
            _FC, {"ngl": "32", "n_depth": "8192"}, "server")

        # the estimator and the drivers must agree about which level means what:
        # whenever a driver emits the flag, so does the estimator
        _CFG_NKVO = Config(model=Path("m.gguf"), llama_bench=Path("b"),
                           llama_server=Path("s"), array="auto", ctx_floor=8192,
                           driver="server",
                           hw={"phys": 8, "logical": 16, "n_layers": 32,
                               "n_ctx_train": 32768, "n_experts": 0,
                               "n_nextn": 0})
        for _lvl in ("0", "1"):
            _row = {"nkvo": _lvl}
            _drv = "-nkvo" in _flat(factor_flags(_CFG_NKVO, _row, "server"))
            _est = "--no-kv-offload" in _fit_params_flags(_FC, _row, "server")
            assert _drv == _est, (_lvl, _drv, _est)

        # ot factor — pattern must be passed literally (no shell quoting)
        f3 = {"ngl": "32", "n_depth": "8192", "ot": "ffn_cpu"}
        s3 = _fit_params_flags(_FC, f3, "bench")
        assert "-ot" in s3
        assert s3[s3.index("-ot") + 1] == OT_PATTERNS["ffn_cpu"]
        # "none" ot → no -ot flag emitted
        assert "-ot" not in _fit_params_flags(_FC, {"ngl": "32", "n_depth": "8192", "ot": "none"}, "bench")

        # ncmoe
        f4 = {"ngl": "32", "n_depth": "8192", "ncmoe": "16"}
        s4 = _fit_params_flags(_FC, f4, "bench")
        assert "-ncmoe" in s4 and "16" in s4

        # ncffn: emitted unconditionally, like every other footprint factor.
        # Whether the estimator can be ASKED is fit_blind_flags' job — keeping
        # the gate out of the flag builder is what lets the cache key below
        # separate rows that differ only in a gated factor.
        f5 = {"ngl": "32", "n_depth": "8192", "ncffn": "16"}
        s5 = _fit_params_flags(_FC, f5, "bench")
        assert "-ncffn" in s5 and "16" in s5

        # --- OOM pruning: the estimator's own capability gate ---
        # These assertions must drive supports_flag from the help text, not from
        # a missing binary: _FC.fit_params does not exist, so supports_flag is
        # False for EVERY flag and a gate asserted that way would pass even if
        # it were inverted.
        _saved_help = dict(_help_cache)
        try:
            # a fit-params that advertises both flags is blind to neither
            _help_cache[str(_FC.fit_params)] = ("-ncmoe, --n-cpu-moe N\n"
                                                "-ncffn, --n-cpu-ffn N\n")
            assert fit_blind_flags(_FC, f5, "bench") == []
            assert fit_blind_flags(_FC, {"ncmoe": "8", "ncffn": "16"}, "bench") == []
            # an older one sees neither, and reports exactly what it is missing
            _help_cache[str(_FC.fit_params)] = "-m, --model FNAME\n"
            assert fit_blind_flags(_FC, f5, "bench") == ["-ncffn"]
            assert fit_blind_flags(_FC, {"ncmoe": "8", "ncffn": "16"}, "bench") == \
                ["-ncffn", "-ncmoe"]
            # a row that sets neither is estimable on any build — the gate must
            # not disable pruning for the whole sweep just because it exists
            assert fit_blind_flags(_FC, {"ngl": "32", "n_depth": "8192"}, "bench") == []
            # ...and a blind row yields None (run it), never a bool. A wrong
            # False here is the silent-prune bug: -ncffn moves weights OFF the
            # GPU, so an estimator that cannot see it reports the un-offloaded
            # footprint and rejects every level of the factor.
            # fit_params must point at a file that EXISTS, or predict_fits bails
            # at its earlier exists() check and this passes without reaching the
            # gate. This script is a convenient stand-in; its help is stubbed
            # above, so nothing is executed.
            _real = Path(__file__).resolve()
            _help_cache[str(_real)] = "-m, --model FNAME\n"      # predates both
            _blind_cfg = Config(model=Path("m.gguf"), llama_bench=Path("lb"),
                                array="L25", ctx_floor=8192,
                                fit_params=_real, hw={"vram": 8192})
            assert _blind_cfg.fit_params.exists()   # else the assert is vacuous
            assert fit_blind_flags(_blind_cfg, f5, "bench") == ["-ncffn"]
            _saved_blind = set(_fit_blind_warned)
            _fit_blind_warned.clear()
            _err = io.StringIO()
            try:
                with contextlib.redirect_stderr(_err):
                    assert predict_fits(_blind_cfg, f5, "bench") is None
                    # once per factor set, not once per row
                    assert predict_fits(_blind_cfg, f5, "bench") is None
            finally:
                _fit_blind_warned.clear()
                _fit_blind_warned.update(_saved_blind)
            # a silently-disabled prune is the thing to avoid: say which flag,
            # and that the rows still run
            assert _err.getvalue().count("OOM prune") == 1, _err.getvalue()
            assert "-ncffn" in _err.getvalue() and "will all run" in _err.getvalue()
            # the same row on a build that HAS the flag is estimable again:
            # the gate opens rather than disabling pruning permanently
            _help_cache[str(_real)] = "-ncmoe, --n-cpu-moe N\n-ncffn, --n-cpu-ffn N\n"
            assert fit_blind_flags(_blind_cfg, f5, "bench") == []
        finally:
            _help_cache.clear()
            _help_cache.update(_saved_help)

        # --- OOM pruning: the fit cache must not merge distinct footprints ---
        # A cached verdict is applied to a row that is then never run, so a
        # collision doesn't produce a wrong number — it silently deletes a
        # valid config from the sweep. Every factor _fit_params_flags forwards
        # must therefore reach the key. Asserted as a property (perturb one
        # factor at a time) rather than by listing names, so a VRAM-relevant
        # factor added later is covered without anyone remembering to.
        base = {"ngl": "32", "n_depth": "8192", "kv_type": "f16",
                "nkvo": "1", "ncmoe": "0", "ot": "none", "ncffn": "0"}
        # each perturbation changes the estimated footprint, so each must
        # change the key (ncmoe/ot were the two the original key dropped).
        # ncffn is here for a second reason: gating its emission on the
        # estimator's help text collapsed every level to one key, so all five
        # inherited a single verdict computed for ncffn=0.
        for knob, other in (("ngl", "16"), ("n_depth", "32768"),
                            ("kv_type", "q4_0"), ("nkvo", "0"),
                            ("ncmoe", "40"), ("ot", "ffn_cpu"),
                            ("ncffn", "32")):
            alt = {**base, knob: other}
            assert _fit_cache_key(_FC, base, "bench") != _fit_cache_key(_FC, alt, "bench"), \
                f"fit cache key ignores {knob}: configs differing only in " \
                f"{knob} would inherit each other's OOM verdict"
        # same config → same key (the memo still has to memoize)
        assert _fit_cache_key(_FC, base, "bench") == _fit_cache_key(_FC, dict(base), "bench")
        # different model, identical flags → different key
        class _FC2:
            model = Path("other.gguf")
            fit_params = Path("fp")
        assert _fit_cache_key(_FC, base, "bench") != _fit_cache_key(_FC2, base, "bench")
    except AssertionError as e:
        print(f"selftest FAILED: {e}")
        return False

    print("selftest: all checks passed")
    return True


# ---------------------------------------------------------------------------
# Iterative refinement: run N passes, each a subprocess of the single-pass tool,
# refining the factor set between passes. Keeps the tested execution path intact.
# ---------------------------------------------------------------------------
def load_results_csv(path: Path, factors: dict) -> list[dict]:
    """Read a results CSV, re-validating every row (docs/measurement-validity.md).

    A row read from disk is a measurement entering the tool again, so the same
    gate applies here as at the drivers (I4). This is also the only thing that
    can repair a CSV recorded *before* the gate existed: a sweep that stored
    tg=1000000 with status OK would otherwise keep winning --report-only
    forever, since re-analysis never re-measures."""
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for c in ("pp_tps", "tg_tps", "eff_tps", "secs"):
                try:
                    r[c] = float(r[c])
                except (KeyError, ValueError, TypeError):
                    r[c] = 0.0
            try:                        # a swept --parallel rides in a column
                par = int(r.get("parallel") or 1)
            except (TypeError, ValueError):
                par = 1
            validate_measurement(r, parallel=par)
            rows.append(r)
    return rows


# results-CSV columns that are measurements/bookkeeping; everything else in a
# header is a factor column (must match the writer's `cols` in main)
RESULT_COLS = {"run_id", "pp_tps", "tg_tps", "eff_tps", "status", "secs",
               "temp_c", "vram_mib", "implausible", "draft_acc", "spec_off",
               "tool_version", "llama_build", "backend", "err_rate", "reuse",
               "draft_cov", "kv_unified", "too_slow", "cache_hit",
               "prompt_tok", "rejected_reps"}


def merge_result_rows(cfg: Config, rows: list[dict],
                      merge_paths: list[Path]) -> list[dict]:
    """Fold rows from earlier results CSVs (--merge-results, e.g. previous
    --iterate passes) into the row set the report/picks/Pareto/probe see, so
    every config ever measured is considered — the final answer can then never
    be worse than an earlier pass's best. A merged row is kept only if it beats
    every already-known measurement of that exact config (same value for every
    factor), so re-measured configs don't pad the Pareto/all-runs tables with
    duplicates. Main-effects/confirm stay on this run's own rows (the analyzer
    needs the balanced array structure; merged rows would skew it)."""
    if not merge_paths:
        return rows

    def key(r):
        return tuple(str(r.get(k, "")) for k in cfg.factors)

    known: dict[tuple, float] = {}
    for r in rows:
        k = key(r)
        known[k] = max(known.get(k, 0.0), score_of(r))
    merged: dict[tuple, dict] = {}
    for mi, mpath in enumerate(merge_paths, 1):
        m = re.search(r"\.(pass\d+)$", mpath.stem)
        tag = m.group(1) if m else f"merge{mi}"
        folded = 0
        for r in load_results_csv(mpath, cfg.factors):
            if not all(str(r.get(k, "")) != "" for k in cfg.factors):
                continue               # foreign CSV missing a factor column
            # re-score under the CURRENT --score mode (same rule as --resume)
            r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
            r["run_id"] = f"{tag}:{r.get('run_id', '')}"
            k = key(r)
            if k in known and known[k] >= score_of(r):
                continue               # this config already measured >= as fast
            if k in merged and score_of(merged[k]) >= score_of(r):
                continue
            merged[k] = r
            folded += 1
        print(f"merged {folded} row(s) from {mpath.name} into the report"
              if folded else
              f"merge: no new rows from {mpath.name} (duplicates or unusable)")
    return rows + list(merged.values())


def probe_sidecar(results: Path) -> Path:
    """Where a sweep persists its max-context probe result (JSON) so
    --report-only can re-show the PROBED CEILING section without a GPU."""
    return Path(str(results) + ".probe.json")


def report_from_results(cfg: Config, args, ap):
    """--report-only: rebuild the terminal report (and --html) from a finished
    sweep's results CSV — no GPU and no llama.cpp binaries. The factor columns
    are whatever the CSV header carries beyond the measurement columns
    (RESULT_COLS), so the picks/Pareto see exactly what the sweep recorded;
    --merge-results folds in further CSVs (e.g. other --iterate passes). The
    PROBED CEILING section appears if the sweep saved a probe sidecar
    (<results>.probe.json); probing anew needs the GPU."""
    path = args.results
    if not path.exists():
        ap.error(f"--report-only: results file not found: {path}")
    rows = load_results_csv(path, {})
    if not rows:
        ap.error(f"--report-only: no rows in {path}")
    names = [c for c in rows[0] if c not in RESULT_COLS]
    if not names:
        ap.error(f"--report-only: no factor columns in {path}")

    def lvl_key(v):
        try:
            return (0, int(v), "")
        except ValueError:
            return (1, 0, v)

    cfg.factors = {n: sorted({str(r.get(n, "")) for r in rows}, key=lvl_key)
                   for n in names}
    cfg.measure_vram = "vram_mib" in rows[0]
    for r in rows:      # re-score under the CURRENT --score mode
        r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
    all_rows = merge_result_rows(cfg, rows, args.merge_results)
    vf = verify_sidecar(path)
    if vf.exists():
        try:
            apply_verification(cfg, all_rows, json.loads(vf.read_text()))
            print(f"applied pick-verification medians from {vf.name}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"(ignoring unreadable {vf.name}: {e})")
    probe = None
    pf = probe_sidecar(path)
    if pf.exists():
        try:
            probe = json.loads(pf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"(ignoring unreadable {pf.name}: {e})")
    print(f"report-only: {len(rows)} row(s) from {path} (no GPU used)"
          + ("" if probe else " — no probe sidecar, PROBED CEILING omitted"))
    report(cfg, all_rows, probe)
    if args.html:
        write_html_report(cfg, all_rows, args.html, probe)


def diff_results(old_path: Path, new_path: Path) -> dict | None:
    """Compare two results CSVs of the same factor space (llama.cpp upgrade,
    driver update, quant swap): match configs on the factor columns both files
    share, then report per-config deltas, status changes, and whether the old
    winner still wins. Compares raw tg_tps (pp alongside) — eff_tps depends on
    the --score mode and request shape a sweep ran with, so it isn't comparable
    across files. Returns a summary dict (for the selftest), None if the files
    can't be compared. Needs no model, GPU, or submodule."""
    def load(path):
        rows = load_results_csv(path, {})
        cols = [c for c in (rows[0].keys() if rows else []) if c not in RESULT_COLS]
        return rows, cols

    old_rows, old_cols = load(old_path)
    new_rows, new_cols = load(new_path)
    for path, rows in ((old_path, old_rows), (new_path, new_rows)):
        if not rows:
            print(f"diff: no rows in {path}")
            return None
    common = [c for c in old_cols if c in new_cols]
    if not common:
        print(f"diff: {old_path.name} and {new_path.name} share no factor columns")
        return None
    ignored = sorted(set(old_cols).symmetric_difference(new_cols))

    def okrow(r):
        return r.get("status") == "OK" and r["tg_tps"] > 0

    def index(rows):
        # several rows can share a key (replicates; or a factor column only the
        # other file has) — keep each config's best measurement
        d: dict[tuple, dict] = {}
        for r in rows:
            k = tuple(str(r.get(c, "")) for c in common)
            if k not in d or r["tg_tps"] > d[k]["tg_tps"]:
                d[k] = r
        return d

    def cfgstr(k):
        return " ".join(f"{c}={v}" for c, v in zip(common, k))

    old_i, new_i = index(old_rows), index(new_rows)
    both = [k for k in old_i if k in new_i]

    print(f"diff: {old_path.name} (old) -> {new_path.name} (new)")
    if ignored:
        print(f"  ignoring factor column(s) not in both files: {', '.join(ignored)}")
    print(f"  configs: {len(old_i)} old, {len(new_i)} new, {len(both)} matched "
          f"({len(old_i) - len(both)} only-old, {len(new_i) - len(both)} only-new)")

    deltas, status_changes = [], []
    for k in both:
        o, n = old_i[k], new_i[k]
        if okrow(o) and okrow(n):
            deltas.append((n["tg_tps"] / o["tg_tps"] - 1.0, k, o, n))
        elif o.get("status") != n.get("status"):
            status_changes.append((k, o.get("status"), n.get("status")))

    if status_changes:
        print(f"\n  status changes ({len(status_changes)}):")
        for k, so, sn in status_changes[:10]:
            print(f"    {so:>7} -> {sn:<7}  {cfgstr(k)}")
        if len(status_changes) > 10:
            print(f"    ... and {len(status_changes) - 10} more")

    summary = {"matched": len(both), "compared": len(deltas),
               "status_changes": len(status_changes), "median_pct": None,
               "old_best_tg": None, "new_best_tg": None,
               "old_winner_still_wins": None}

    if deltas:
        deltas.sort(key=lambda t: t[0])
        pcts = [d[0] for d in deltas]
        n = len(pcts)
        med = pcts[n // 2] if n % 2 else (pcts[n // 2 - 1] + pcts[n // 2]) / 2
        improved = sum(1 for p in pcts if p > 0.02)
        regressed = sum(1 for p in pcts if p < -0.02)
        summary.update(median_pct=med)
        print(f"\n  tg over the {n} config(s) OK in both: median {med:+.1%}, "
              f"{improved} improved / {regressed} regressed (beyond ±2%)")
        worst = [d for d in deltas[:5] if d[0] < -0.02]
        best = [d for d in reversed(deltas[-5:]) if d[0] > 0.02]
        for label, picks in (("largest regressions", worst),
                             ("largest improvements", best)):
            if picks:
                print(f"  {label}:")
                for p, k, o, nw in picks:
                    print(f"    {p:+7.1%}  tg {o['tg_tps']:6.1f} -> {nw['tg_tps']:6.1f}"
                          f"  (pp {o['pp_tps']:.0f} -> {nw['pp_tps']:.0f})  {cfgstr(k)}")

    old_ok = [r for r in old_i.values() if okrow(r)]
    new_ok = [r for r in new_i.values() if okrow(r)]
    if old_ok and new_ok:
        ob = max(old_ok, key=lambda r: r["tg_tps"])
        nb = max(new_ok, key=lambda r: r["tg_tps"])
        ko = tuple(str(ob.get(c, "")) for c in common)
        kn = tuple(str(nb.get(c, "")) for c in common)
        summary.update(old_best_tg=ob["tg_tps"], new_best_tg=nb["tg_tps"],
                       old_winner_still_wins=(ko == kn))
        print(f"\n  best OK config: old {ob['tg_tps']:.1f} tg t/s -> "
              f"new {nb['tg_tps']:.1f} tg t/s")
        if ko == kn:
            print(f"  old winner still wins: {cfgstr(kn)}")
        else:
            now = f"{new_i[ko]['tg_tps']:.1f} tg t/s" if ko in new_i \
                  else "not measured in new"
            print(f"  old winner ({cfgstr(ko)}): {now}")
            print(f"  NEW winner: {cfgstr(kn)}")
    return summary


def build_child_argv(args, cfg: Config, factors: dict, results_path: Path,
                     final: bool, prev_results: list[Path]) -> list[str]:
    """One pass's command line for the single-pass tool (explicit everything so
    the child reproduces the resolved config)."""
    argv = [str(args.model), "--run",
            "--driver", cfg.driver, "--profile", cfg.profile, "--array", "auto",
            "--reps", str(cfg.reps), "--n-prompt", str(cfg.n_prompt),
            "--n-gen", str(cfg.n_gen), "--ctx-floor", str(cfg.ctx_floor),
            "--parallel", str(cfg.parallel), "--score", cfg.score,
            "--timeout", str(args.timeout),
            "--cooldown", str(args.cooldown),
            "--spec-draft-n-max", str(cfg.spec_draft_n_max),
            "--llama-bench", str(cfg.llama_bench),
            "--llama-server", str(cfg.llama_server),
            # results_path already includes the dir; make the child's join a no-op
            "--results-dir", ".", "--results", str(results_path)]
    if args.no_mtp:
        argv.append("--no-mtp")
    if cfg.ngram:
        argv.append("--ngram")
    if cfg.ngram_type:                             # tuning stage: pin the variant
        argv += ["--ngram-type", cfg.ngram_type]
    if cfg.measure_vram:
        argv.append("--vram")
    if args.no_shuffle:
        argv.append("--no-shuffle")
    if args.no_thermal_wait:
        argv.append("--no-thermal-wait")
    if args.thermal_baseline is not None:      # children reuse the parent's idle
        argv += ["--thermal-baseline", str(args.thermal_baseline)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.max_depth is not None:
        argv += ["--max-depth", str(args.max_depth)]
    argv += ["--min-kv", "any"]   # parent already applied the floor; don't re-filter
    # Carry every earlier pass's measurements into this pass's report/picks: a
    # refinement pass can chase a noise-led region, and without the merge the
    # final answer would FORGET a better config pass 1 already measured.
    for rp in prev_results:
        argv += ["--merge-results", str(rp)]
    for name, levels in factors.items():
        flag = "--env" if name in cfg.env_factor_names else "--factor"
        argv += [flag, f"{name}={','.join(str(x) for x in levels)}"]
    if final:
        if args.confirm or args.full:
            argv.append("--confirm")
        if args.html:
            argv += ["--html", str(args.html)]
        if args.no_probe:
            argv.append("--no-probe")
        argv += ["--verify-picks", str(args.verify_picks)]
    else:
        argv.append("--no-probe")  # only probe on the final pass
        argv += ["--verify-picks", "0"]  # only verify final picks
    return argv


def run_iterations(args, cfg: Config):
    base = args.results
    suffix = base.suffix or ".csv"
    factors = dict(cfg.factors)
    prev: list[Path] = []
    for p in range(1, args.iterate + 1):
        final = p == args.iterate
        rp = base.with_name(f"{base.stem}.pass{p}{suffix}")
        print("\n" + "#" * 70)
        print(f"# PASS {p}/{args.iterate}  "
              + ", ".join(f"{k}={'/'.join(str(x) for x in v)}"
                          for k, v in factors.items()))
        print("#" * 70, flush=True)
        argv = build_child_argv(args, cfg, factors, rp, final, prev)
        env = {**os.environ, "LLAMA_OPTIMIZE_CHILD": "1"}
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
        if rc != 0:
            print(f"\npass {p} exited with code {rc}; stopping iteration.")
            return
        if final:
            break
        rows = load_results_csv(rp, factors)
        if not rows:
            print("no results to refine from; stopping.")
            return
        prev.append(rp)
        cfg.factors = factors
        refined = refine_factors(cfg, rows)
        if refined == factors or all(len(v) == 1 for v in refined.values()):
            print("\nfactors converged — stopping refinement early.")
            return
        factors = refined
    print(f"\nAll passes complete. Final report + results (all passes merged): "
          f"{base.with_name(f'{base.stem}.pass{args.iterate}{suffix}')}")


def rank_gate_values(rows: list[dict], gate: str) -> list[tuple]:
    """(gate value, best measured objective) over OK rows, best first. Used to
    rank ngram variants after the screen stage."""
    best: dict[str, float] = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        v = str(r.get(gate, ""))
        if not v:
            continue
        s = score_of(r)
        if v not in best or s > best[v]:
            best[v] = s
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def keep_top_gate_values(ranked: list[tuple], k: int) -> list[str]:
    """The top-k gate values worth tuning: the best `k` real variants (never the
    'none'/off value unless nothing else was measured). At least one when any real
    variant exists."""
    real = [v for v, _ in ranked if v not in ("none", "")]
    return real[:max(1, k)] if real else [v for v, _ in ranked[:1]]


def screen_base_winners(cfg: Config, screen_factors: dict, rows: list[dict]) -> dict:
    """Hold the unconditional knobs at their screen-winning level for the tuning
    stage (so its runs are about the ngram knobs), keeping n_depth's full spread
    (the tradeoff axis, per DESIGN.md). The gate is pinned separately."""
    held: dict = {}
    for name, levels in screen_factors.items():
        if name == "ngram":
            continue
        if name == "n_depth":
            held[name] = levels                        # tradeoff axis: keep spread
            continue
        means = factor_level_means(rows, name)
        best = max(means, key=means.get) if means else str(levels[0])
        held[name] = [str(best)]                        # settle at the screen winner
    return held


def run_ngram_stages(args, cfg: Config):
    """Staged ngram search (docs/CONDITIONAL-FACTORS.md): a screen stage measures
    every variant at default knobs, then one tuning stage per surviving variant
    sweeps only that variant's knobs (gate pinned) with the unconditional knobs
    held at the screen winners. The final answer is the best MEASURED config across
    all stages (each stage merges the earlier ones). Each stage is a child of the
    single-pass tool, exactly like --iterate passes."""
    base = args.results
    suffix = base.suffix or ".csv"

    # Stage 0 — screen: sweep the parent's finalized factor set (base + gate, no
    # conditional children since cfg.ngram_type is unset), passed explicitly so the
    # child inherits the KV floor and any --factor overrides (as --iterate does).
    screen_factors = cfg.factors
    screen_rp = base.with_name(f"{base.stem}.ngram-screen{suffix}")
    print("\n" + "#" * 70)
    print("# NGRAM STAGE 0 — screen variants at default knobs")
    print("#" * 70, flush=True)
    argv = build_child_argv(args, cfg, screen_factors, screen_rp, False, [])
    env = {**os.environ, "LLAMA_OPTIMIZE_CHILD": "1"}
    rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
    if rc != 0:
        print(f"\nngram screen exited with code {rc}; stopping.")
        return
    rows = load_results_csv(screen_rp, screen_factors)
    if not rows:
        print("no screen results; stopping.")
        return

    ranked = rank_gate_values(rows, "ngram")
    kept = keep_top_gate_values(ranked, cfg.ngram_keep)
    print("\nngram screen ranking (best measured objective per variant):")
    for v, s in ranked:
        print(f"  {v:<14} {s:.2f}" + ("   → tuning" if v in kept else ""))
    held = screen_base_winners(cfg, screen_factors, rows)

    # Stage k — tune each surviving variant; the last one produces the final report.
    prev = [screen_rp]
    tunable = [v for v in kept if ngram_child_levels(v)]
    if not tunable:
        print("\nno surviving variant has tunable knobs; screen result stands.")
        return
    for i, v in enumerate(tunable):
        final = i == len(tunable) - 1
        rp = base.with_name(f"{base.stem}.ngram-{v}{suffix}")
        print("\n" + "#" * 70)
        print(f"# NGRAM STAGE {i + 1} — tune {v}")
        print("#" * 70, flush=True)
        cfg_v = replace(cfg, ngram_type=v)             # pin gate ⇒ child sweeps its knobs
        argv = build_child_argv(args, cfg_v, held, rp, final, prev)
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
        if rc != 0:
            print(f"\nngram tuning of {v} exited with code {rc}; stopping.")
            return
        prev.append(rp)
    print(f"\nNgram staging complete. Final report + results (all stages merged): "
          f"{prev[-1]}")


# ---------------------------------------------------------------------------
# Morris screening (funnel stage 1): rank many knobs by importance (mu*) and flag
# interactions (sigma) at ~r*(k+1) runs, using the vendored `robust` morris tool
# for the design + analysis and our own driver (with crash journal) for the runs.
# ---------------------------------------------------------------------------
# Table row: factor, mu*, sigma. morris grew a 95% CI on mu* (upstream E1),
# printed glued to the value as `4.867[4.4,5.51]` — so a plain split()[1] no
# longer parses as a float. Accept mu* with or without a trailing CI, in either
# the glued or space-separated spelling, so we read both old and new output.
_MORRIS_ROW = re.compile(
    r"^(\S+)\s+([-+0-9.eE]+)\s*(?:\[[^\]]*\])?\s+([-+0-9.eE]+)")


# morris --json contract version we know how to read. Upstream bumps this only
# on a rename or removal, never on an addition, so an unknown value means the
# meaning of a field we rely on may have changed — refuse it and fall back
# rather than misread it. Declining a version we do not know is the signal the
# original CI change lacked.
MORRIS_JSON_SCHEMA = 1


def parse_morris_json(text: str):
    """(data, None) for usable morris --json, or (None, reason) to fall back."""
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "output was not JSON (older morris without --json?)"
    if not isinstance(d, dict) or d.get("tool") != "morris":
        return None, "JSON did not come from morris"
    schema = d.get("schema")
    if schema != MORRIS_JSON_SCHEMA:
        return None, (f"unknown --json schema {schema!r} (this build reads "
                      f"{MORRIS_JSON_SCHEMA}); not guessing at its meaning")
    if not isinstance(d.get("factors"), list):
        return None, "JSON had no factors array"
    return d, None


def parse_morris_analyze(text: str):
    """Parse the morris analyze table into [(factor, mu_star, sigma), ...]."""
    out, started = [], False
    for line in text.splitlines():
        if line.startswith("------"):
            started = True
            continue
        if started:
            if not line.strip():
                break
            m = _MORRIS_ROW.match(line)
            if m:
                try:
                    out.append((m.group(1), float(m.group(2)), float(m.group(3))))
                except ValueError:
                    pass
    return out


def morris_screen(cfg: Config, args, ap, trajectories: int):
    """Run a Morris screen; report mu*/sigma; reduce cfg.factors to the ones that
    matter (drop negligible ones, fixed at their best-seen level)."""
    morris = find_robust_binary("morris")
    if not morris.exists():
        ap.error(f"morris binary not found at {morris}; build it: "
                 f"make -C {SUBMODULE_DIR}")
    base = args.results
    space_path = Path(str(base) + ".space")
    res_path = Path(str(base) + ".morris_results.csv")
    journal_path = Path(str(base) + ".journal")

    def is_cat(name):
        return FACTORS.get(name, {}).get("kind") == "cat" or name in cfg.env_factor_names

    # .space: numeric factors as [min,max]; categoricals as [0, n-1] index space
    lines = ["factors:"]
    for name, levels in cfg.factors.items():
        if is_cat(name):
            lines.append(f"  {name}: 0, {max(1, len(levels) - 1)}")
        else:
            nums = sorted(float(x) for x in levels)
            lo, hi = nums[0], (nums[-1] if nums[-1] != nums[0] else nums[0] + 1)
            lines.append(f"  {name}: {lo:g}, {hi:g}")
    seed = args.seed if args.seed is not None else 42
    lines += [f"trajectories: {trajectories}", "grid_levels: 4", f"seed: {seed}"]
    space_path.write_text("\n".join(lines) + "\n")

    out = subprocess.run([str(morris), "sample", str(space_path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        ap.error(f"morris sample failed: {out.stderr.strip()}")
    design = list(csv.DictReader(out.stdout.splitlines()))
    print(f"\n{'#' * 70}\n# MORRIS SCREEN — {trajectories} trajectories, "
          f"{len(design)} runs (r*(k+1)={trajectories}*{len(cfg.factors) + 1})\n{'#' * 70}")
    if cfg.driver == "server":
        print("note: screening on the server driver reloads the model for every "
              "point (slow).\n      For base knobs, screen on the bench driver; "
              "reserve the server driver for MTP/concurrency knobs.")

    def map_point(row):
        f = {}
        for name, levels in cfg.factors.items():
            v = float(row[name])
            if is_cat(name):
                f[name] = levels[max(0, min(len(levels) - 1, int(round(v))))]
            else:
                f[name] = str(min(levels, key=lambda L: abs(float(L) - v)))
        return f

    cache, rows = {}, []
    journal = open(journal_path, "a")
    fh = open(res_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["run_id", "eff_tps"])
    fh.flush()
    try:
        for j, row in enumerate(design, 1):
            rid = row.get("run_id", str(j))
            f = map_point(row)
            ckey = tuple(sorted(f.items()))
            if ckey in cache:
                eff, status = cache[ckey], "OK(cached)"
            else:
                prefix = (f"[screen {j}/{len(design)}] "
                          + " ".join(f"{k}={f[k]}" for k in cfg.factors))
                journal_write(journal, "TRY", "run", f"screen-{rid}", json.dumps(f))
                res = with_ticker(prefix, args.timeout,
                                  lambda ff=f: drive_one(cfg, ff, args.timeout))
                eff = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
                status = res["status"]
                cache[ckey] = eff
                print(f"{prefix} -> {status} {cfg.score}={eff:.1f} t/s", flush=True)
                # the screen decides which knobs get DROPPED — settle it like
                # the sweep so drift doesn't rank the factors
                if args.thermal_baseline is not None:
                    wait_until_cool(args.thermal_baseline)
                elif args.cooldown > 0:
                    time.sleep(args.cooldown)
            rows.append({"status": "OK" if eff > 0 else status, "eff_tps": eff, **f})
            w.writerow([rid, f"{eff:.4f}"])
            fh.flush()
    finally:
        fh.close()
        journal.close()

    base_cmd = [str(morris), "analyze", str(space_path), str(res_path),
                "--metric", "eff_tps"]
    # Decisions read --json, the stable contract. The table is for the log only:
    # parsing it positionally is what silently turned this whole stage into a
    # no-op when upstream glued a confidence interval onto the mu* column.
    js = subprocess.run(base_cmd + ["--json"], capture_output=True, text=True)
    tbl = subprocess.run(base_cmd, capture_output=True, text=True)
    print("\n" + tbl.stdout)
    data, why = (parse_morris_json(js.stdout) if js.returncode == 0
                 else (None, f"morris --json exited {js.returncode}"))
    # Diagnostics (near-tie cuts, overlapping CIs, all-inert results) go to
    # stderr in --json mode too, and the prose there is the useful part.
    for stream in (js.stderr, tbl.stderr):
        if stream and stream.strip():
            print(stream.strip())
            break
    if data:
        rankings = [(f.get("factor", ""), float(f.get("mu_star", 0.0)),
                     float(f.get("sigma", 0.0)))
                    for f in data["factors"] if f.get("factor")]
        # all_zero means the response never moved across the whole design. That
        # is almost always a broken harness, not a genuine "nothing matters" —
        # and it used to be indistinguishable from a real empty ranking.
        if data.get("all_zero"):
            print("\n  every factor measured zero effect — the response never "
                  "moved.\n  That usually means the runs failed rather than that "
                  "nothing matters;\n  check the statuses above before trusting "
                  "this. Keeping all factors.")
            return
    else:
        print(f"(morris --json unusable: {why}; falling back to the table)")
        rankings = parse_morris_analyze(tbl.stdout)
    if not rankings:
        print("(morris analyze returned no rankings — keeping all factors)")
        return

    max_mu = max(mu for _, mu, _ in rankings) or 1.0
    keep = [n for n, mu, _ in rankings if mu >= 0.1 * max_mu]
    if not keep:                       # never drop everything
        keep = [rankings[0][0]]
    # n_depth is the report's tradeoff axis (speed vs context), not a knob to
    # settle — dropping it here would collapse the Pareto/picks to one depth
    # (same guard as refine_factors).
    if "n_depth" in cfg.factors and "n_depth" not in keep:
        keep.append("n_depth")
    dropped = [n for n, _, _ in rankings if n not in keep]
    interacting = [n for n, mu, s in rankings if mu > 0 and s >= mu / 2]

    print(f"KEEP (matter): {', '.join(keep)}")
    if dropped:
        print(f"DROP (negligible, fixed at best): {', '.join(dropped)}")
    if interacting:
        print(f"INTERACTION/nonlinear (σ≥μ*/2): {', '.join(interacting)}")

    best = {}
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if means:
            best[name] = max(means, key=means.get)
    cfg.factors = {name: (levels if name in keep else [str(best.get(name, levels[0]))])
                   for name, levels in cfg.factors.items()}
    print("→ continuing to the Taguchi sweep on the factors that matter.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # flags alphabetized (keep it that way when adding one)
    ap.add_argument("model", type=Path, nargs="?", help="path to the GGUF model")
    ap.add_argument("--array", default="auto",
                    help="orthogonal array; default 'auto' picks the smallest that "
                         "fits your factors. Advanced: force L9/L18/L25/L27/L125/...")
    ap.add_argument("--confirm", action="store_true",
                    help="after the sweep, run the predicted-optimal config to "
                         "verify the model's prediction (implied by --full)")
    ap.add_argument("--cooldown", type=float, default=0,
                    help="fixed seconds to pause between runs so the GPU can cool "
                         "(fallback when no temp sensor; default 0)")
    ap.add_argument("--ctx-scan", action="store_true",
                    help="probe the physical context ceiling FIRST, then set the "
                         "n_depth axis to fractions of it (0, ¼, ½, ¾, 0.9×) so the "
                         "sweep/Pareto span your full usable context range")
    ap.add_argument("--ctx-size", "-c", type=int, default=None,
                    help="tune at a FIXED context size (like llama.cpp -c): "
                         "shorthand for --min-context N --max-context N")
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("OLD.csv", "NEW.csv"),
                    help="compare two results CSVs (e.g. before/after a llama.cpp "
                         "upgrade): per-config tg deltas over the factor columns "
                         "both share, status changes, and whether the old winner "
                         "still wins. Needs no model or GPU; exits after the report")
    ap.add_argument("--driver", choices=["bench", "server"], default=None,
                    help="benchmark driver (default: from profile). 'server' "
                         "measures real generation incl. MTP and concurrency")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="sweep an environment variable as a factor (repeatable), "
                         "e.g. --env GGML_CUDA_FORCE_MMQ=0,1")
    ap.add_argument("--factor", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="override/add a sweepable factor (repeatable), e.g. "
                         "--factor ngl=56,60,64 --factor nkvo=0,1 --factor poll=0,50")
    ap.add_argument("--full", action="store_true",
                    help="thorough: 5 reps per config (steadier numbers, slower)")
    ap.add_argument("--html", type=Path, default=None,
                    help="also write a visual HTML report (Pareto + main effects)")
    ap.add_argument("--iterate", type=int, default=1, metavar="N",
                    help="run N refinement passes: each settles the low-impact "
                         "factors at their winner and refines the high-impact ones "
                         "onto a finer grid (screen -> refine -> ...). The final "
                         "report/picks merge ALL passes' results, so extra passes "
                         "can only add information, never lose pass 1's best. "
                         "default 1")
    ap.add_argument("--llama-bench", type=Path, default=None,
                    help="explicit path to the llama-bench binary")
    ap.add_argument("--llama-cpp", type=Path, default=None,
                    help="path to your llama.cpp (its root or build/bin dir). "
                         "Also read from $LLAMA_CPP or $PATH. Required if the "
                         "binaries aren't auto-found.")
    ap.add_argument("--llama-server", type=Path, default=None,
                    help="explicit path to the llama-server binary")
    ap.add_argument("--max-context", "--max-depth", type=int, default=None,
                    dest="max_depth",
                    help="cap the context axis and the ceiling probe at this "
                         "many tokens (don't explore above it)")
    ap.add_argument("--merge-results", action="append", type=Path, default=[],
                    metavar="CSV",
                    help="fold rows from an earlier results CSV into this run's "
                         "report/picks/Pareto without re-running them (repeatable; "
                         "--iterate uses this to carry every pass into the final "
                         "report). Main-effects stay on this run's own balanced "
                         "design.")
    ap.add_argument("--min-context", "--ctx-floor", type=int, default=None,
                    dest="ctx_floor",
                    help="minimum context you need — BALANCED targets it, FASTEST "
                         "only considers configs verified to hold it, and emitted "
                         "-c is floored at it where the sweep has evidence "
                         "(default: from profile)")
    ap.add_argument("--min-kv", default="q8_0", metavar="TYPE",
                    help="KV-cache quality floor: never consider a KV type lossier "
                         "than this (default q8_0, near-lossless). 'any' explores "
                         "all; e.g. --min-kv q4_0 to allow aggressive quantization")
    ap.add_argument("--n-gen", type=int, default=None,
                    help="generated tokens per measurement (default: from profile)")
    ap.add_argument("--n-prompt", type=int, default=None,
                    help="prompt tokens per measurement (default: from profile)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="don't add draft-mtp flags to the emitted server command "
                         "even if the model has an MTP head")
    ap.add_argument("--ngram", action="store_true",
                    help="enable ngram self-speculative decoding (server only): a "
                         "screen stage measures every variant, then the top variants "
                         "get their parameters tuned (staged so the search stays "
                         "clean and affordable — see docs/CONDITIONAL-FACTORS.md)")
    ap.add_argument("--ngram-type", default=None,
                    choices=["ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v"],
                    help="pin the ngram variant and tune only its parameters in one "
                         "pass (skips the variant screen)")
    ap.add_argument("--ngram-keep", type=int, default=2, metavar="K",
                    help="how many top variants from the screen to tune (default 2)")
    ap.add_argument("--ngram-fast", action="store_true",
                    help="greedy ngram search: tune only the single best-screened "
                         "variant (--ngram-keep 1)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the max-context probe (which runs by default after "
                         "the sweep: binary-searches the physical context ceiling)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="run configs in array order (default: randomized to "
                         "decorrelate thermal/background drift from factors)")
    ap.add_argument("--no-thermal-wait", action="store_true",
                    help="disable the default 'wait and watch' settle that pauses "
                         "between runs until GPU temp returns near its idle "
                         "baseline (keeps measurements thermally comparable)")
    ap.add_argument("--prefix-reuse", type=float, default=None, metavar="PCT",
                    help="workload SHAPE: percent of each prompt that is a "
                         "prefix shared across requests (0-100). Describes your "
                         "traffic, it is not tuned. Defaults to the profile's "
                         "shape (0, or 90 for `agents`). 100 = every request "
                         "identical, which is the pre-4ffa97a behaviour and "
                         "INFLATES n-gram speculation (see CHANGELOG)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="concurrent request streams for the server driver "
                         "(default: from profile)")
    ap.add_argument("--probe-ctx", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: the probe is now default
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="workload profile (request shape + objective): single "
                         "(interactive), agents (big-context tool use), multi "
                         "(concurrent serving). Usually set via --use-case; "
                         "default: single")
    ap.add_argument("--quick", action="store_true",
                    help="fast screen: 1 rep per config (noisier, ~1/3 the time)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from an existing results CSV — no GPU, "
                         "no llama.cpp: reads --results (default: the model's CSV in "
                         "--results-dir), folds in --merge-results, honors --score "
                         "and --html. PROBED CEILING is included when the sweep "
                         "saved its probe result (<results>.probe.json).")
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per config (default: 3, or --quick=1/--full=5)")
    ap.add_argument("--results", type=Path, default=None,
                    help="results CSV name, in --results-dir (default: the model's "
                         "name, e.g. <model>.csv; journal/HTML/pass files land beside)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="directory for all output (default: results/)")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in --results (rows are "
                         "saved incrementally, so an interrupted sweep resumes)")
    ap.add_argument("--retry-crashed", action="store_true",
                    help="on resume, also retry configs that were started but never "
                         "finished (suspected machine crash/hang); default skips them")
    ap.add_argument("--run", action="store_true",
                    help="actually execute the benchmark sweep (uses the GPU)")
    ap.add_argument("--score", choices=["tg", "eff"], default="tg",
                    help="objective for stats/fits/picks: 'tg' (default) ranks by "
                         "generation speed alone (pp is measured and reported but "
                         "can't sway the pick); 'eff' ranks by blended effective "
                         "t/s for the profile's request (prefill + decode)")
    ap.add_argument("--screen", type=int, nargs="?", const=6, default=None, metavar="R",
                    help="Morris pre-screen with R trajectories (default 6) to rank "
                         "knobs by importance and drop the negligible ones before the "
                         "sweep — cheap (~R*(k+1) runs). Great with many --factor knobs.")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for execution order (reproducibility)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline logic checks and exit (no GPU, no model)")
    ap.add_argument("--server-start-timeout", type=int, default=180,
                    help="max seconds to wait for llama-server to load before "
                         "giving up on a config (default 180)")
    ap.add_argument("--spec-draft-n-max", type=int, default=2,
                    help="--spec-draft-n-max for MTP speculative decoding (default 2)")
    ap.add_argument("--thermal-baseline", type=float, default=None,
                    help=argparse.SUPPRESS)  # internal: parent hands the idle
    #                                          baseline to --iterate child passes
    ap.add_argument("--thinking", action="store_true",
                    help="tune for a reasoning/thinking workload — long generations "
                         "(decode-heavy). Sets n_gen to a reasoning length (~2048); "
                         "default (no flag) is non-thinking / short answers")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="wall-clock budget for ONE config (default 1200), "
                         "covering warm-up and every rep together. Previously "
                         "the server driver applied this per HTTP REQUEST, so a "
                         "config could take (1+reps)x this; it is now a deadline "
                         "on both drivers")
    ap.add_argument("--draft-model", "-md", type=Path, default=None, metavar="GGUF",
                    help="draft model for speculative decoding (server driver). "
                         "An INPUT, not a factor — a sweep either has one or it "
                         "does not. Adding it unlocks the draft-side placement "
                         "factors (-ngld, -ctkd/-ctvd), which llama.cpp ignores "
                         "without it. Note a model whose MTP head is embedded "
                         "can speculate either way: without -md from its own "
                         "head, with -md from a separate head file")
    ap.add_argument("--mmproj", type=Path, default=None, metavar="GGUF",
                    help="multimodal projector (server driver). An INPUT, like "
                         "--draft-model: it occupies VRAM from load whether or "
                         "not image traffic arrives, so a sweep that will serve "
                         "with one must measure with one. Adds --mmproj-offload "
                         "as a factor. OOM pruning turns OFF for these rows — "
                         "llama-fit-params cannot be told a projector exists")
    ap.add_argument("--levels", type=int, default=5, metavar="N",
                    help="levels per auto-generated numeric factor (default 5). "
                         "This is the sweep's cost dial: the orthogonal array is "
                         "sized by the WIDEST factor, so narrowing one knob "
                         "changes nothing while another still has 5. --levels 3 "
                         "narrows them together and drops L125 to L27 — 125 runs "
                         "to 27. Explicit --factor values are untouched")
    ap.add_argument("--min-tgs", type=float, default=0.0, metavar="TPS",
                    help="abandon a config generating slower than TPS. A config "
                         "that would MEET the floor finishes within "
                         "tokens/TPS seconds, so this shortens the per-config "
                         "budget instead of waiting out the full --timeout — the "
                         "reason it saves time on llama-bench too, where nothing "
                         "can be watched mid-run. Rows are marked SLOW, keep "
                         "their numbers, and are excluded from the picks")
    ap.add_argument("--min-pps", type=float, default=0.0, metavar="TPS",
                    help="as --min-tgs, for prefill. On the server driver this "
                         "is answerable the moment the warm-up returns, so a "
                         "config failing it skips its decode reps entirely")
    ap.add_argument("--tgs-timeout", type=int, default=60, metavar="SECS",
                    help="never judge a config on less than this many seconds "
                         "(default 60), so a small --n-gen cannot derive a "
                         "budget that measures model load instead of throughput")
    ap.add_argument("--use-case", choices=list(USE_CASES), default=None,
                    metavar="{app,single,agents,multi-user}",
                    help="high-level runbook that bundles driver+profile+concurrency: "
                         "app (general/embedded llama.cpp via llama-bench), single "
                         "(llama-server, one worker/user), agents (several concurrent "
                         "agents, long tool-use prompts), multi-user (many concurrent "
                         "chat users). --driver/--profile/--parallel override the "
                         "runbook.")
    ap.add_argument("--verify-picks", type=int, default=None, metavar="R",
                    help="re-measure each pick candidate R extra times and report "
                         "the MEDIAN of all its measurements (default: 2, or "
                         "--quick=0/--full=3; 0 disables) — guards the headline "
                         "numbers against thermal/run-to-run noise. Medians persist "
                         "to <results>.verify.json for --report-only; the CSV keeps "
                         "the raw sweep measurements")
    ap.add_argument("--vram", action="store_true",
                    help="measure actual peak VRAM used per run (polls "
                         "rocm-smi/nvidia-smi); records vram_mib and draws the "
                         "VRAM curve + physical ceiling on the Pareto chart")
    ap.add_argument("--no-oom-prune", action="store_true",
                    help="don't skip configs predicted to exceed VRAM (by default, "
                         "llama-fit-params estimates each config's footprint before "
                         "running it, and skips those certain to OOM — saving a "
                         "model load + timeout per doomed row)")
    ap.add_argument("--fit-headroom", type=int, default=512, metavar="MiB",
                    help="safety margin under detected VRAM for OOM pruning; "
                         "configs using more than (vram - headroom) are skipped "
                         "(default: 512)")
    args = ap.parse_args()
    # Q1: a bare invocation on a terminal is someone finding out what this does.
    # Anything more specific than that — a flag they chose, a non-sweep action,
    # or a redirected stdout (a script) — is left exactly as it was.
    asked = interview_wanted(args, sys.argv[1:])
    args = apply_intent(ap, args, asked)

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.diff:
        sys.exit(0 if diff_results(*args.diff) is not None else 1)
    if not args.model:
        ap.error("model path is required (or use --selftest / --diff)")
    if not args.model.exists():
        ap.error(f"model not found: {args.model}")

    # default results name from the model; place relative names in --results-dir
    if args.results is None:
        args.results = Path(f"{args.model.stem}.csv")
    if not args.results.is_absolute():
        prefixed = args.results_dir / args.results
        # --report-only reads an existing file: a CWD-relative path that exists
        # wins over the --results-dir prefix (users paste the path they see)
        if not (args.report_only and args.results.exists()
                and not prefixed.exists()):
            args.results = prefixed
    if args.html is not None and not args.html.is_absolute():
        args.html = args.results_dir / args.html
    if args.run:  # ensure the output directory exists
        args.results.parent.mkdir(parents=True, exist_ok=True)
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)

    meta = read_gguf_metadata(args.model)
    mhw = model_hw(meta)
    n_layers = mhw["n_layers"]
    n_experts = mhw["n_experts"]
    n_ctx_train = mhw["n_ctx_train"]
    n_nextn = mhw["n_nextn"]
    phys = detect_physical_cores()
    logical = detect_logical_cores()
    # Resolved before hardware detection: capacity now comes from llama.cpp
    # itself where possible, so detect_vram_mib needs the binary. resolve_binary
    # never raises — a missing binary is a best-guess path the caller validates.
    llama_bench = resolve_binary("llama-bench", args.llama_bench, args.llama_cpp)
    llama_server = resolve_binary("llama-server", args.llama_server, args.llama_cpp)
    fit_params = resolve_binary("llama-fit-params", None, args.llama_cpp)
    detected = detect_vram_mib(llama_server, llama_bench)
    vram, vram_src = detected if detected else (None, "")

    if args.quick and args.full:
        ap.error("--quick and --full are mutually exclusive")

    if args.prefix_reuse is not None and not 0.0 <= args.prefix_reuse <= 100.0:
        ap.error("--prefix-reuse is a percentage: 0-100")

    # --ctx-size N: fix context at N (min == max), like llama.cpp -c
    if args.ctx_size is not None:
        args.ctx_floor = args.ctx_size
        args.max_depth = args.ctx_size

    # Resolve the workload with precedence: built-in default < use-case runbook <
    # explicit flag. The use-case (if any) supplies defaults for profile/driver/
    # parallel; any flag the user set on the command line still wins.
    uc = USE_CASES.get(args.use_case) or {}
    profile = args.profile or uc.get("profile") or "single"
    prof = PROFILES[profile]
    n_prompt = args.n_prompt if args.n_prompt is not None else prof["n_prompt"]
    # thinking = long reasoning generation (decode-heavy); non-thinking = short
    n_gen = (args.n_gen if args.n_gen is not None
             else 2048 if args.thinking else prof["n_gen"])
    ctx_floor = args.ctx_floor if args.ctx_floor is not None else prof["ctx_floor"]
    prefix_reuse = (args.prefix_reuse / 100.0 if args.prefix_reuse is not None
                    else prof.get("prefix_reuse", 0.0))
    driver = args.driver or uc.get("driver") or prof["driver"]
    parallel = (args.parallel if args.parallel is not None
                else uc.get("parallel", prof.get("parallel", 1)))
    reps = args.reps if args.reps is not None else (1 if args.quick else 5 if args.full else BENCH_REPS)
    if args.verify_picks is None:
        args.verify_picks = 0 if args.quick else 3 if args.full else 2

    # A model with an MTP/NextN head defaults to the server driver: llama-bench
    # cannot do speculative decoding, so on bench the MTP speedup is neither
    # measured nor tunable (its knobs are server-only). An explicit --driver or
    # --use-case still wins, as does --no-mtp; needs llama-server built.
    # ngram self-speculation is also server-only.
    if args.ngram_fast:
        args.ngram_keep = 1
    args.ngram = args.ngram or args.ngram_type is not None   # --ngram-type implies ngram
    if (driver == "bench" and args.driver is None and uc.get("driver") is None
            and (n_nextn or 0) > 0 and not args.no_mtp and llama_server.exists()):
        driver = "server"
        print("note: model has an MTP head — driver auto-switched to server so "
              "the sweep measures and tunes MTP (--driver bench to override)")
    if (driver == "bench" and args.driver is None and uc.get("driver") is None
            and args.ngram and llama_server.exists()):
        driver = "server"
        print("note: ngram speculation requested — driver auto-switched to server "
              "(--driver bench to override)")

    cfg = Config(
        model=args.model.resolve(),
        llama_bench=llama_bench,
        llama_server=llama_server,
        array=args.array,
        ctx_floor=ctx_floor,
        reps=reps,
        n_prompt=n_prompt,
        n_gen=n_gen,
        max_depth=args.max_depth,
        emit_mtp=not args.no_mtp,
        # request shape, from the profile unless --prefix-reuse says otherwise
        prefix_reuse=prefix_reuse,
        ngram=args.ngram,
        ngram_type=args.ngram_type,
        ngram_keep=args.ngram_keep,
        spec_draft_n_max=args.spec_draft_n_max,
        profile=profile,
        driver=driver,
        parallel=parallel,
        score=args.score,
        draft_model=args.draft_model,
        mmproj=args.mmproj,
        levels=max(2, args.levels),
        min_tgs=max(0.0, args.min_tgs),
        min_pps=max(0.0, args.min_pps),
        slow_grace=max(1, args.tgs_timeout),
        measure_vram=args.vram,
        oom_prune=not args.no_oom_prune,
        fit_headroom_mib=args.fit_headroom,
        fit_params=fit_params,
        server_start_timeout=args.server_start_timeout,
        hw={**mhw, "phys": phys, "logical": logical, "vram": vram,
            "numa_nodes": detect_numa_nodes()},
    )
    cfg.factors = build_factors(cfg)
    if args.ctx_size is not None:            # fixed context: don't sweep n_depth
        cfg.factors["n_depth"] = [str(args.ctx_size)]

    # apply --factor overrides / additions
    for spec in args.factor:
        name, _, vals = spec.partition("=")
        name = name.strip()
        levels = [v.strip() for v in vals.split(",") if v.strip()]
        if name in RENAMED_FACTORS:
            new, why = RENAMED_FACTORS[name]
            ap.error(f"--factor {name}: '{name}' is now derived from a sibling "
                     f"factor and was renamed to '{new}'.\n  {why}.")
        if name not in FACTORS:
            ap.error(f"--factor: unknown factor '{name}' "
                     f"(sweepable: {', '.join(sorted(FACTORS))})")
        if is_server_only(name) and cfg.driver != "server":
            ap.error(f"--factor {name} requires the server driver "
                     "(--driver server or --profile multi)")
        if FACTORS[name].get("bench") is None and cfg.driver == "bench":
            ap.error(f"--factor {name} isn't supported by the bench driver; "
                     "use --driver server")
        if not levels:
            ap.error(f"--factor {name}: no levels given")
        lvl_errors = validate_factor_levels({name: levels})
        if lvl_errors:
            ap.error("--factor " + "; ".join(lvl_errors))
        cfg.factors[name] = levels

    # Overrides can pin a gate to a value that makes its knobs inert -- most of
    # all `--factor mtp=0`, which turns speculation off while leaving four
    # speculative-tuning columns sweeping full level sets (issue #11).
    cfg.factors, _inert = prune_gated_factors(cfg.factors)
    if _inert:
        _gates = sorted({FACTORS[n]["gated_by"][0] for n in _inert})
        print(f"not swept: {', '.join(_inert)} — inert at the pinned "
              f"{', '.join(_gates)}")

    # apply --env: each becomes an orthogonal factor that sets a process env var
    for spec in args.env:
        name, _, vals = spec.partition("=")
        name = name.strip()
        levels = [v.strip() for v in vals.split(",") if v.strip()]
        if not name or not levels:
            ap.error(f"--env expects NAME=v1,v2,... (got '{spec}')")
        cfg.factors[name] = levels
        cfg.env_factor_names.add(name)

    # DM2: a draft model is a server-driver concept. llama-bench cannot speculate
    # at all, so accepting it there would emit a flag the binary rejects — or,
    # worse, silently sweep draft factors that never reach anything.
    if cfg.draft_model:
        if cfg.driver != "server":
            ap.error("--draft-model needs the server driver (llama-bench cannot "
                     "speculate): add --driver server, or a --use-case that "
                     "implies it")
        if not cfg.draft_model.exists():
            ap.error(f"--draft-model not found: {cfg.draft_model}")
    if cfg.mmproj:
        if cfg.driver != "server":
            ap.error("--mmproj needs the server driver (llama-bench has no "
                     "multimodal path): add --driver server")
        if not cfg.mmproj.exists():
            ap.error(f"--mmproj not found: {cfg.mmproj}")

    # apply the KV quality floor (quality is essentially only KV-type deep here;
    # MTP is lossless, other knobs don't affect quality)
    if "kv_type" in cfg.factors:
        kept = kv_at_or_above(cfg.factors["kv_type"], args.min_kv)
        dropped = [l for l in cfg.factors["kv_type"] if l not in kept]
        cfg.factors["kv_type"] = kept
        if dropped:
            print(f"KV quality floor --min-kv {args.min_kv}: dropping {dropped} "
                  f"(keeping {kept})")

    # --report-only: rebuild the report from the results CSV and exit — no GPU
    if args.report_only:
        if args.run:
            ap.error("--report-only doesn't run anything; drop --run")
        report_from_results(cfg, args, ap)
        return

    # Thermal baseline: capture the idle GPU temperature ONCE, before any GPU
    # work (--ctx-scan/--screen/pass 1 all heat the card), and thread it through
    # child passes via --thermal-baseline — a child capturing its own "idle" at
    # the start of pass 2 would bake a hot GPU into the target and neuter the
    # settle for that whole pass.
    if args.run and not args.no_thermal_wait and args.thermal_baseline is None:
        args.thermal_baseline = gpu_temp_c()

    # --ctx-scan: probe the physical ceiling first, then make the context axis
    # fractions of it, so the sweep spans the full usable range on THIS hardware.
    if args.ctx_scan and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--ctx-scan needs --run")
        needed = cfg.llama_server if cfg.driver == "server" else cfg.llama_bench
        if not needed.exists():
            ap.error(f"{needed.name} not found ({needed}); pass --llama-cpp")
        # base config = full offload + most context-efficient allowed KV (lossiest
        # allowed = smallest KV = furthest reach) + first level of the rest
        base = {k: v[0] for k, v in cfg.factors.items()}
        if "ngl" in cfg.factors:
            base["ngl"] = max(cfg.factors["ngl"], key=lambda x: int(x))
        if "kv_type" in cfg.factors:
            base["kv_type"] = max(cfg.factors["kv_type"],
                                  key=lambda k: KV_QUALITY.index(k) if k in KV_QUALITY else 0)
        cap = cfg.hw.get("n_ctx_train") or 131072
        if args.max_depth:                       # --max-context caps the scan
            cap = min(cap, args.max_depth)
        print(f"### Context scan — probing the physical ceiling first "
              f"(ngl={base.get('ngl')} kv={base.get('kv_type')}, cap={cap})...")
        res = probe_max_context(cfg, base, args.timeout, cap, args.thermal_baseline)
        if not res:
            ap.error("--ctx-scan: base config failed to load even at depth 0")
        ceiling = res[0]
        lo = args.ctx_floor or 0                  # --min-context sets the low end
        depths = sorted({max(0, (lo + int((ceiling - lo) * fr)) // 1024 * 1024)
                         for fr in (0.0, 0.25, 0.5, 0.75, 0.9)})
        cfg.factors["n_depth"] = [str(d) for d in depths]
        print(f"physical ceiling ~{ceiling} tokens → n_depth axis "
              f"[{lo}..{ceiling}]: {depths}\n")

    # funnel stage 1: Morris pre-screen (reduces cfg.factors to the ones that
    # matter) before the Taguchi sweep / iterate. Runs in the parent, not children.
    if args.screen and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--screen needs --run")
        morris_screen(cfg, args, ap, args.screen)

    # ngram staged search: a screen stage measures every variant, then the top-K
    # get their knobs tuned (gate pinned). Only the unpinned parent orchestrates —
    # a --ngram-type child (pinned gate) runs as a normal single-variant sweep.
    if (cfg.ngram and cfg.ngram_type is None and cfg.driver == "server"
            and not os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--ngram needs --run")
        run_ngram_stages(args, cfg)
        return

    # iterative refinement: orchestrate N passes as subprocesses of this tool
    if args.iterate > 1 and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--iterate needs --run")
        run_iterations(args, cfg)
        return

    # Ask the box which -ngl levels load, rather than trusting a verdict measured
    # somewhere else. Only where there is reason to doubt (recurrent memory: see
    # ngl_levels) and only under --run, so plan-only keeps its promise to touch no
    # GPU. What loads is a property of the model, the backend and the build, and
    # none of those is knowable from here (issue #18).
    if (args.run and getattr(cfg, "ngl_recurrent", False)
            and cfg.driver == "server" and cfg.factors.get("ngl")):
        cands = sorted({int(x) for x in
                        (getattr(cfg, "ngl_candidates", None) or cfg.factors["ngl"])}
                       | {int(x) for x in cfg.factors["ngl"]})
        probe_ctx = cfg.n_prompt + cfg.n_gen + 256
        print(f"probing which of -ngl {', '.join(map(str, cands))} load on this "
              f"box (recurrent model; {len(cands)} launches)...", flush=True)
        live, dead = probe_loadable_ngl(cfg, cands, probe_ctx, args.timeout)
        if live:
            cfg.factors["ngl"] = [str(x) for x in live]
            if dead:
                print(f"  -ngl {', '.join(map(str, dead))} did not load and are "
                      f"dropped; sweeping {', '.join(map(str, live))}")
            else:
                print(f"  all {len(live)} load — sweeping the full span")
        else:
            print("  none loaded; leaving the grid alone so the sweep records "
                  "what happens rather than deciding for you")

    # resolve the array now that the factor set is final
    if str(cfg.array).lower() == "auto":
        cfg.array = choose_array(cfg.factors) or "auto"

    print("=" * 70)
    print("llama-optimize")
    print("=" * 70)
    print(f"model      : {cfg.model.name}")
    arch = meta.get("general.architecture", "?")
    moe = f"MoE ({n_experts} experts)" if n_experts else "dense"
    print(f"arch       : {arch}   layers: {n_layers if n_layers else '?'}   {moe}")
    print(f"CPU        : {phys} physical / {logical} logical cores")
    # The source is printed because getting it wrong is silent otherwise: a
    # too-small number just prunes runs away (issue #7), and the line that says
    # where it came from is what makes that diagnosable from a paste.
    print(f"VRAM       : {vram} MiB ({vram_src})" if vram
          else "VRAM       : (undetected)")
    # Free VRAM is a different question from total, and the one that decides
    # whether a sweep started now will measure anything. Printed whenever the
    # device list carries it, so the header records the state the run began in.
    _devs = list_devices(cfg.llama_bench) or list_devices(cfg.llama_server)
    _free, _total = vram_headroom(_devs)
    if _total > 0:
        print(f"VRAM free  : {_free} of {_total} MiB "
              f"({100.0 * _free / _total:.0f}%) at start")
    # A build that cannot see the GPU produces plausible numbers, not an error,
    # so this is the only place it can be caught before a whole sweep is wasted.
    warn_gpu_visibility(
        gpu_visibility(vram, vram_src,
                       supports_flag(cfg.llama_bench, "--list-devices")
                       or supports_flag(cfg.llama_server, "--list-devices")),
        vram_src, cfg.factors)
    warn_vram_headroom(_devs)
    if cfg.oom_prune and vram and cfg.fit_params.exists():
        print(f"OOM prune  : on (llama-fit-params, {cfg.fit_headroom_mib} MiB headroom)")
    elif cfg.oom_prune:
        what = "no fit-params binary" if not cfg.fit_params.exists() else "VRAM undetected"
        print(f"OOM prune  : off ({what})")
    if cfg.draft_model:
        d_layers = draft_layer_count(cfg.draft_model)
        print(f"draft model: {cfg.draft_model.name}"
              + (f"   layers: {d_layers}" if d_layers else "")
              + "  — draft placement swept (-ngld, -ctkd/-ctvd)")
    if cfg.mmproj:
        print(f"projector  : {cfg.mmproj.name}  — resident from load; OOM "
              "pruning off for these rows")
    if n_ctx_train:
        print(f"native ctx : {n_ctx_train}")
    if n_nextn:
        emit = ("swept as factors (mtp on/off, spec_n_max)" if "mtp" in cfg.factors
                else "will add --spec-type draft-mtp to server cmd" if cfg.emit_mtp
                else "disabled (--no-mtp)")
        print(f"MTP        : yes ({n_nextn} NextN layer(s)) — {emit}")
        if cfg.driver == "bench":
            print("             hint: add --driver server to MEASURE the MTP "
                  "speedup (bench can't); otherwise it's only emitted")
    if cfg.ngram:
        if cfg.ngram_type:
            emit = f"tuning {cfg.ngram_type} knobs (variant pinned)"
        elif "ngram" in cfg.factors:
            emit = "screening variants (tuning follows for the top ones)"
        else:
            emit = "will add --spec-type ngram-mod to server cmd"
        print(f"ngram      : yes — {emit}")
    print(f"profile    : {cfg.profile}  (request {cfg.n_prompt} prompt + "
          f"{cfg.n_gen} gen tokens; driver={cfg.driver})")
    if cfg.driver == "server":
        shape = (f"{cfg.prefix_reuse * 100:.0f}% shared prefix across requests")
        warn = ("  <-- identical requests; INFLATES n-gram speculation, see "
                "CHANGELOG" if cfg.prefix_reuse >= 1.0 else
                "  (set --prefix-reuse to match your traffic)")
        print(f"workload   : {shape}{warn}")
    print("objective  : " + ("eff (effective t/s: blends pp + tg)"
                             if cfg.score == "eff" else
                             "tg (generation t/s; pp reported, not scored)"))
    mode = "quick" if args.quick else "full" if args.full else "standard"
    print(f"mode       : {mode}  ({cfg.reps} rep{'s' if cfg.reps != 1 else ''}/config)")
    print(f"array      : {cfg.array}   ctx floor: {cfg.ctx_floor}")
    print("\nfactors:")
    for name, levels in cfg.factors.items():
        print(f"  {name:10s}: {', '.join(levels)}")
    # Say when the ngl grid was reshaped, and why. A level set that silently
    # depends on an estimate is one the reader cannot check (issue #14).
    if getattr(cfg, "ngl_recurrent", False):
        print("  note: ngl is 0 or 99 only — this model has recurrent (SSM) "
              "memory, and every partial offload core-dumped llama-server when "
              "measured (issue #18), so those levels would abort rather than "
              "produce a number. Measured on qwen35/qwen35moe + ROCm; if partial "
              "offload works for you, --factor ngl=0,16,32,48,64 restores it.")
    elif getattr(cfg, "ngl_biased", False):
        print(f"  note: ngl levels bias to full offload — every layer fits in "
              f"VRAM at depth {max(int(d) for d in cfg.factors['n_depth'])}. "
              f"ngl=0 is kept in case that verdict is wrong "
              f"(--no-oom-prune restores an even span).")
    fixed_bits = [f"mmap {'on' if FIXED_MMAP else 'off'}"]
    if "fa" not in cfg.factors:
        fixed_bits.insert(0, f"flash-attn {'on' if FIXED_FA else 'off'}")
    if "batch_ratio" not in cfg.factors:
        fixed_bits.append(f"batch {FIXED_BATCH}")
    print(f"fixed      : {', '.join(fixed_bits)}  "
          f"(p={cfg.n_prompt} n={cfg.n_gen} reps={cfg.reps})")

    try:
        exp, runs = generate_runs(cfg.factors, cfg.array)
    except Exception as e:
        ap.error(f"can't build the design for array '{cfg.array}' with "
                 f"{len(cfg.factors)} factors: {e}\n"
                 "  Try --array auto (default) to let it pick a fitting array.")
    print(f"\ngenerated {len(runs)} runs "
          + (f"(array={getattr(exp, 'array_type', cfg.array)})" if exp is not None
             else "(direct sweep — <=1 varying factor, no array needed)"))

    if not args.run:
        print("\n--- PLAN ONLY (no GPU used). Re-run with --run to execute. ---")
        print(f"\nSample command (run 1, {cfg.driver} driver):")
        f0 = runs[0]["factors"]
        if cfg.driver == "server":
            n_ctx = cfg.n_prompt + cfg.n_gen + 256
            print("  " + " ".join(build_server_args(cfg, f0, 8080, n_ctx)))
        else:
            print("  " + " ".join(bench_command(cfg, f0)))
        per = estimate_secs_per_run(cfg)
        print(f"\nAll {len(runs)} runs would execute sequentially "
              f"(~{fmt_dur(len(runs) * per)} at a rough {per:.0f}s/run, "
              f"scaled from model size, reps and n_gen — a guess, not a "
              f"measurement).")
        if asked is not None:
            # The estimate above is the "then show the cost" half of the
            # interview: the numbers are on screen before anything is spent.
            derived = [str(args.model)] + intent_args(asked) + ["--run"]
            if offer_to_run(derived):
                sys.exit(subprocess.call([sys.executable, str(Path(__file__))]
                                         + derived))
            print("Not run. Copy the command above when you are ready.")
        return

    needed = cfg.llama_server if cfg.driver == "server" else cfg.llama_bench
    if not needed.exists():
        ap.error(
            f"{needed.name} not found (looked at: {needed}).\n"
            "  Need the path to your llama.cpp build. Pass --llama-cpp "
            "/path/to/llama.cpp\n  (its build/bin dir), set $LLAMA_CPP, put it on "
            f"$PATH, or pass --{needed.name} directly.")

    # preflight: confirm the binary actually runs (catches missing GPU libs / a
    # wrong build) in a couple of seconds, instead of failing deep in the sweep.
    ok, why = preflight(needed)
    if not ok:
        ap.error(f"{needed.name} at {needed} won't run: {why}\n"
                 "  Is it built for your GPU? Check its ROCm/CUDA/Metal libraries "
                 "are on the loader path (e.g. LD_LIBRARY_PATH), or rebuild llama.cpp.")

    # Idle thermal baseline for the "wait and watch" settle between runs —
    # captured up front (before ctx-scan/screen) or inherited from the parent.
    thermal_wait = not args.no_thermal_wait
    thermal_baseline = args.thermal_baseline if thermal_wait else None
    if thermal_wait and thermal_baseline is not None:
        print(f"thermal    : idle baseline {thermal_baseline:.0f}°C — settle to "
              f"≤{thermal_baseline + THERMAL_BAND_C:.0f}°C between runs "
              f"(cap {THERMAL_CAP_S:.0f}s; --no-thermal-wait to disable)")
    elif thermal_wait:
        print("thermal    : no GPU temp sensor — "
              + (f"using fixed --cooldown {args.cooldown:.0f}s between runs"
                 if args.cooldown > 0 else "no settle between runs (see --cooldown)"))

    # --- execute sweep ---
    # Derived factors contribute two columns: the relative level they were swept
    # at, and the absolute value it materialized to (C3).
    abs_cols = [derived_abs_name(n) for n in derived_names(cfg.factors)]
    cols = (["run_id"] + list(cfg.factors.keys()) + abs_cols
            + ["pp_tps", "tg_tps", "eff_tps", "status", "secs", "temp_c"]
            + (["vram_mib"] if cfg.measure_vram else [])
            + (["draft_acc", "draft_cov", "spec_off"]
               if spec_cols_wanted(cfg) else [])
            + (["backend"] if cfg.driver == "bench"
               else ["err_rate", "reuse", "cache_hit", "prompt_tok",
                     "rejected_reps", "kv_unified"])
            # only when floors are in play, so ordinary sweeps keep their shape
            + (["too_slow"] if (cfg.min_tgs > 0 or cfg.min_pps > 0) else [])
            # why a row was rejected, on the row. It costs one mostly-empty
            # column and it is the difference between a bug report that can be
            # decided on sight and one that takes five round trips (issue #11).
            + ["implausible", "tool_version", "llama_build"])

    # Resume keys on run_id (unique per array row), not config values, because
    # orthogonal arrays can repeat a config across rows (intentional replication).
    # Assumes resume uses the same factors/array so run_ids line up.
    rows, done = [], set()
    if args.resume and args.results.exists():
        with open(args.results, newline="") as fh:
            for r in csv.DictReader(fh):
                for c in ("pp_tps", "tg_tps", "eff_tps", "secs"):
                    try:
                        r[c] = float(r[c])
                    except (KeyError, ValueError, TypeError):
                        r[c] = 0.0
                # re-score under the CURRENT --score mode, not whatever mode
                # wrote the CSV — a resumed sweep must fit one objective
                r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
                if r.get("run_id"):
                    rows.append(r)
                    done.add(str(r["run_id"]))
        print(f"resuming: {len(done)} run(s) already in {args.results}, "
              "skipping them")

    # Provenance stamped on every row rather than in a sidecar: a results CSV
    # gets copied, merged (--merge-results) and mailed around on its own, and the
    # question it must answer later — "what produced this?" — travels with it.
    stamp = {"tool_version": __version__,
             "llama_build": llama_build(cfg.llama_server, cfg.llama_bench)}

    fresh = not (args.resume and args.results.exists())
    fh = open(args.results, "w" if fresh else "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    # Crash journal (see helpers): configs started but never finished on a prior
    # run likely hung/rebooted the machine — mark CRASH and skip, don't retry.
    journal_path = Path(str(args.results) + ".journal")
    if args.resume and not args.retry_crashed:
        tried_load, ok_load, tried_run = read_journal(journal_path)
        crashed_loads = set(tried_load) - ok_load
        # screen runs journal into the same file but are not sweep rows — a
        # completed screen must not resurface as phantom CRASH rows on resume
        crashed = {str(rid): fac for rid, fac in tried_run.items()
                   if str(rid) not in done and not str(rid).startswith("screen-")}
        for run in runs:                       # runs whose server load crashed
            rid = str(run.get("run_id"))
            if rid not in done and load_key_str(cfg, run["factors"]) in crashed_loads:
                crashed.setdefault(rid, run["factors"])
        for rid, fac in crashed.items():
            crow = {"run_id": rid, **{k: fac.get(k, "") for k in cfg.factors},
                    **derived_abs_cols(cfg, fac),
                    "pp_tps": 0.0, "tg_tps": 0.0, "eff_tps": 0.0,
                    "status": "CRASH", "secs": 0.0, **stamp}
            rows.append(crow)
            writer.writerow(crow)
            done.add(rid)
        fh.flush()
        if crashed:
            ids = sorted(crashed, key=lambda x: (len(x), x))
            print(f"⚠  {len(crashed)} config(s) were started but never finished "
                  "on a prior run — suspected machine crash/hang.")
            print(f"   Marked CRASH and NOT retrying: runs {ids}")
            print("   (use --retry-crashed to attempt them again once addressed)")
    journal = open(journal_path, "w" if fresh else "a")

    # Execution plan. The server driver groups configs that share load-time
    # params (only the request — prompt length via n_depth — differs) so one
    # server serves the whole group. The bench driver runs each config solo.
    if cfg.driver == "server":
        groups: dict = {}
        order = []
        for run in runs:
            k = load_key(cfg, run["factors"])
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(run)
        plan = [groups[k] for k in order]
        reused = len(runs) - len(plan)
        if reused > 0:
            print(f"server reuse: {len(plan)} server launch(es) for {len(runs)} "
                  f"runs ({reused} reload(s) saved)")
    else:
        plan = [[run] for run in runs]

    # Randomize execution order to decorrelate slow drift (GPU thermal throttling,
    # background load) from the factors — standard DOE practice. For the server
    # driver we shuffle whole groups so reuse still holds. --no-shuffle keeps
    # array order; --seed makes it reproducible.
    if not args.no_shuffle:
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        random.Random(seed).shuffle(plan)
        print(f"execution order: randomized (seed={seed}) to decorrelate drift")

    sweep_start = time.time()
    i = 0
    try:
        for group in plan:
            session = None
            pending = [r for r in group if str(r.get("run_id", "")) not in done]
            if cfg.driver == "server" and pending:
                launch = pending[0]["factors"]
                par = int(launch.get("parallel", cfg.parallel))
                max_depth = max(int(r["factors"].get("n_depth", 0)) for r in pending)
                n_ctx = (cfg.n_prompt + max_depth + cfg.n_gen + 256) * \
                    ctx_slots_multiplier(launch.get(
                        "concurrency", str(par) if par > 1 else "1"))
                lp = (f"server launch: ngl={launch['ngl']} kv={launch['kv_type']} "
                      f"ub={launch['ubatch']} ctx={n_ctx}")
                lk = load_key_str(cfg, launch)
                journal_write(journal, "TRY", "load", lk, json.dumps(launch))
                session = with_ticker(
                    lp, args.timeout,
                    lambda lf=launch, nc=n_ctx: ServerSession(cfg, lf, nc, args.timeout))
                if getattr(session, "ok", False):
                    journal_write(journal, "OK", "load", lk)  # load survived
            try:
                for run in group:
                    i += 1
                    f = run["factors"]
                    rid = run.get("run_id", i)
                    if str(rid) in done:
                        print(f"[{i}/{len(runs)}] run {rid}: already done, skipping")
                        continue
                    nl = cfg.hw.get("n_layers") or "?"
                    prefix = (f"[{i}/{len(runs)}] run {rid}: "
                              f"{f['ngl']}/{nl} layers on GPU, {f['threads']} threads, "
                              f"{f['kv_type']} KV cache, {f['n_depth']}-token context, "
                              f"ubatch {f['ubatch']}")
                    journal_write(journal, "TRY", "run", rid, json.dumps(f))  # durable
                    # OOM pruning: estimate VRAM footprint before loading the model.
                    # Skip rows certain to exceed physical VRAM (ROADMAP item 2).
                    if cfg.oom_prune:
                        est = predict_fits(cfg, f, cfg.driver)
                        if est is False:
                            res = {"status": "SKIP_PRED", "pp_tps": 0.0,
                                   "tg_tps": 0.0, "secs": 0.0, "vram_mib": 0}
                            res["eff_tps"] = 0.0
                            row = {"run_id": rid, **f, **derived_abs_cols(cfg, f),
                                   **res, "temp_c": "", **stamp}
                            rows.append(row)
                            writer.writerow(row)
                            fh.flush()
                            journal_write(journal, "OK", "run", rid)  # close journal
                            print(f"{prefix} -> SKIP_PRED (predicted OOM, skipped)")
                            continue
                    temp0 = gpu_temp_c()   # start temp: thermal comparability is
                    #                        checkable in the CSV, not assumed
                    if cfg.driver == "server":
                        res = with_ticker(
                            prefix, args.timeout,
                            lambda ff=f, ss=session: measure_in_session(
                                cfg, ff, ss, args.timeout))
                    else:
                        res = run_with_progress(cfg, f, args.timeout, prefix)
                    res["eff_tps"] = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
                    row = {"run_id": rid, **f, **derived_abs_cols(cfg, f), **res,
                           "temp_c": f"{temp0:.0f}" if temp0 is not None else "",
                           **stamp}
                    rows.append(row)
                    writer.writerow(row)   # incremental save: survive a crash/kill
                    fh.flush()
                    elapsed = time.time() - sweep_start
                    eta = (elapsed / i) * (len(runs) - i)
                    raw = (f"tg={res['tg_tps']:.1f} pp={res['pp_tps']:.1f}"
                           if cfg.score == "eff" else f"pp={res['pp_tps']:.1f}")
                    print(f"{prefix} -> {res['status']} "
                          f"{cfg.score}={res['eff_tps']:.1f} t/s ({raw}) ({res['secs']:.0f}s)  "
                          f"[{i}/{len(runs)} done, elapsed {fmt_dur(elapsed)}, "
                          f"ETA ~{fmt_dur(eta)}]", flush=True)
                    if res.get("implausible"):      # P4: never discard quietly
                        print(f"  discarded: {res['implausible']}", flush=True)
                    if res.get("spec_off"):        # F1: the row is not measuring
                        print("  speculation did not run: this config asks for "
                              "it, but llama.cpp drafted no tokens", flush=True)
                    if i < len(runs):                  # settle before the next run
                        if thermal_wait and thermal_baseline is not None:
                            wait_until_cool(thermal_baseline)
                        elif args.cooldown > 0:
                            time.sleep(args.cooldown)  # fixed fallback, no sensor
            finally:
                if session:
                    session.close()
    finally:
        fh.close()
        journal.close()
    print(f"\nwrote {args.results}")

    all_rows = merge_result_rows(cfg, rows, args.merge_results)

    # Stage-3 verification: the pick candidates get re-measured and their
    # medians become the reported numbers (before the probe, so the probe's
    # base-config choice also sees the settled scores). Persisted to a sidecar
    # so --report-only re-applies it; the CSV keeps the raw measurements.
    if args.verify_picks > 0:
        verify = verify_picks(cfg, all_rows, args.verify_picks, args.timeout,
                              thermal_baseline)
        if verify:
            verify_sidecar(args.results).write_text(json.dumps(verify, indent=1))
            apply_verification(cfg, all_rows, verify)

    # Stage-2 probe runs BEFORE the report so its ceiling appears alongside
    # the picks (fixed context via --ctx-size: no ceiling search). The result
    # is persisted next to the results CSV so --report-only can re-show it.
    probe = None
    if not args.no_probe and args.ctx_size is None:
        probe = run_probe_stage(cfg, all_rows, args.timeout, thermal_baseline)
        if probe:
            probe_sidecar(args.results).write_text(json.dumps(probe, indent=1))

    report(cfg, all_rows, probe)
    opt, predicted = taguchi_effects(cfg, exp, rows)

    if args.html:
        write_html_report(cfg, all_rows, args.html, probe)

    if (args.confirm or args.full) and opt:
        # Run the predicted-optimal config directly to check the additive model
        # (Taguchi best practice). A big predicted-vs-actual gap => interactions.
        f = {k: cfg.factors[k][0] for k in cfg.factors}
        f.update({k: str(v) for k, v in opt.items() if k in cfg.factors})
        print("\n### Confirmation run (predicted-optimal config)")
        # the prediction came from settled runs — measure the check settled too,
        # or a hot GPU masquerades as "interactions"
        wait_until_cool(thermal_baseline)
        prefix = "confirm: " + " ".join(f"{k}={f[k]}" for k in cfg.factors)
        res = with_ticker(prefix, args.timeout,
                          lambda: drive_one(cfg, f, args.timeout))
        actual = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
        raw = (f"tg={res['tg_tps']:.1f} pp={res['pp_tps']:.1f}"
               if cfg.score == "eff" else f"pp={res['pp_tps']:.1f}")
        print(f"  predicted {cfg.score}: "
              + (f"{predicted:.1f} t/s" if predicted else "(n/a)"))
        print(f"  measured  {cfg.score}: {actual:.1f} t/s  "
              f"({raw}, status={res['status']})")
        if predicted and actual > 0:
            err = abs(actual - predicted) / predicted * 100
            verdict = ("additive model holds — trust the prediction" if err <= 15
                       else "LARGE gap: interactions likely — trust the Pareto pick")
            print(f"  prediction error: {err:.0f}%  → {verdict}")


if __name__ == "__main__":
    main()
