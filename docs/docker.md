# Running llama-optimize against a containerised llama.cpp

Answers the common case of "I don't build llama.cpp locally — I run the official
image". The short version: **run llama-optimize inside the container**, not
against it from the host.

## Origin

Asked by [@mendhak](https://github.com/mendhak) in
[issue #2](https://github.com/bigattichouse/llama-optimize/issues/2), and
answered by [@fboudra](https://github.com/fboudra), whose recipe this document
is based on. Credit for the approach is theirs; this file exists so the answer
doesn't stay buried in an issue thread.

## Why inside the container

llama-optimize is not a client. It doesn't talk to a running server over HTTP
and read numbers off it — it *launches* `llama-bench` or `llama-server` itself,
once per configuration, with a different set of flags each time, and tears each
one down. Two consequences:

- It needs to **execute the binaries**, so they must be on the same filesystem
  it runs on. Pointing it at a container's exposed port is not enough; there is
  no flag for "drive a server someone else started".
- It needs to **control the launch flags**, since the flags *are* the experiment.
  A container started with a fixed command line has already frozen the variables
  llama-optimize exists to vary.

Running the tool inside the container satisfies both, and costs nothing extra:
the official image already ships the build dependencies (git, gcc) that the
`robust` submodule needs.

## Recipe

Start the image with an interactive shell instead of its normal entrypoint:

```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  -v /path/to/your/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda13
```

`--entrypoint /bin/bash` is the key part — it overrides the image's default of
launching `llama-server` directly. Mount your model directory so the sweep has
GGUFs to work on, and keep whatever GPU flags you normally use (`--gpus all` for
CUDA; on ROCm images the equivalent `--device`/`--group-add` flags).

Then, inside the container, install as usual:

```bash
git clone --recurse-submodules https://github.com/bigattichouse/llama-optimize
cd llama-optimize
make -C robust
```

The llama.cpp binaries live in `/app` in the official images, which is not where
llama-optimize looks by default, so point it there:

```bash
python3 llama-optimize.py --llama-cpp /app -m /models/your-model.gguf
```

`--llama-cpp` accepts a directory and searches it, plus `build/bin` and `bin`
underneath — so `/app` resolves `llama-bench` and `llama-server` without naming
each binary. If your image lays them out differently, `--llama-bench` and
`--llama-server` take explicit paths.

## Things worth knowing before a long sweep

**Results die with the container.** A sweep is many runs and can take hours.
`--rm` deletes the container on exit, taking the CSV, the HTML report, and the
crash journal with it. Mount a volume for output, or drop `--rm`.

**The crash journal wants to survive a restart too.** llama-optimize records
"about to try X" before each risky launch, so a config that hangs or reboots the
box is skipped rather than retried on resume. That protection is only worth
anything if the journal outlives the container — another reason to write results
to a mounted path.

**Measurements are only as isolated as the container.** Thermal settling
(`--cooldown`, `--thermal-baseline`) and VRAM measurement read the *host's* GPU.
They work from inside a container with GPU access, but nothing stops other
workloads on the same host from perturbing the numbers. A sweep competing with
other GPU tenants will produce a confident ranking of noise.

**`--numa` and CPU-affinity factors may not behave as they do on the host.**
Container CPU limits and cpuset restrictions change what the thread and affinity
knobs can actually reach. If the container is pinned to a subset of cores, the
thread levels llama-optimize derives from the detected core count will be wrong
for the host — treat CPU-side conclusions from inside a constrained container
with suspicion.
