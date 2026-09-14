# glm-ocr-hu-env

Container image for post-training [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)
on Hungarian, built by GitHub Actions and published to GHCR.

```
ghcr.io/HoneyMonster7/glm-ocr-hu-env:latest
```

This repo holds **only an environment specification** — no data, no model code, no
measurements. The project it serves is private; nothing here is worth keeping private, which
is what makes free Actions minutes and free public-package hosting available.

## Why an image at all

GPU billing starts before your code does. Dependency resolution, wheel downloads and the
2 GB base checkpoint are all things that can happen off the clock, and this image is where
they happen. A rented pod should go from boot to first training step without resolving
anything.

The development machine is an arm64 Mac with no CUDA, so the image cannot be meaningfully
built or tested locally — CI is the only sane build path. `make build-local` cross-builds
under QEMU but the result is untestable here.

## What is in it

One Python environment holding **both** the trainer and the inference server:

| | version | for |
|---|---|---|
| torch | 2.13.0 | |
| transformers | 5.8.0 | carries the `glm_ocr` architecture |
| vllm | 0.28.0 | `ocr-pt predict` against the OpenAI-compatible API |
| LLaMA-Factory | `100e9a4` (0.9.6.dev0) | `ocr-pt train` shells out to `llamafactory-cli` |
| GLM-OCR weights | baked at `/opt/hf` | `BAKE_WEIGHTS=0` to omit |

Plus `ocr-post-training`'s own runtime deps, so the private repo installs on the pod with
`pip install -e . --no-deps` and resolves nothing.

### One environment, not two — and the pin that makes it possible

The obvious design is two venvs, mirroring the project's own local `.venv` / `.venv_training`
split. It is not necessary, and avoiding it saves a second ~3 GB CUDA torch stack:

```
LLaMA-Factory 100e9a4  needs  transformers >=4.55.0,<=5.8.0,!=5.6.0
vllm 0.28.0            needs  transformers >=5.5.3       + torch ==2.13.0
vllm 0.29.0            needs  transformers >=5.10.4      <- no overlap
```

So **0.28.0 is the newest vLLM that can share an environment with this trainer**, and
transformers 5.8.0 sits in the intersection. Bumping vLLM past 0.28.0 is not a version bump;
it is a decision to split the image. `requirements.in` says so at the point of the pin.

### No flash-attn, deliberately

The generated `train.yaml` sets no `flash_attn` key, so LLaMA-Factory resolves attention to
`sdpa`. flash-attn is therefore optional — and it is the single most painful thing to install
(PyPI ships a source distribution only; a mismatched wheel URL silently falls back to a
multi-hour CUDA compile). It is left out until something measures that it is worth the
trouble. vLLM brings its own kernels via flashinfer regardless.

### Base image

`python:3.12-slim-trixie`, pinned by digest — not `pytorch/pytorch` or `nvidia/cuda`. Nothing
here compiles: torch, vLLM and flashinfer all install as wheels carrying their own CUDA
runtime, so a CUDA base image would add a toolkit that the wheels then shadow. Only the
*driver* has to come from the host, and the NVIDIA container runtime injects that.

Pinned by digest because tags are mutable, and a silent base refresh that moves the Python
patch level or the C++ ABI breaks the wheels weeks later with an `undefined symbol` error
that looks like anything except a base-image change.

## The one constraint when renting: driver r580+

The wheels are **CUDA 13** builds. CUDA is only minor-version compatible, so a host on a
r5xx driver below 580 cannot load them — and RunPod's Community cloud does still carry older
hosts. Filter for CUDA 13 when picking a pod.

`scripts/preflight.sh` checks this and exits non-zero with the reason. Run it as the first
command on a fresh pod:

```bash
bash /opt/app/preflight.sh     # ~30 s, checks driver, GPU, and the real imports
```

It exists because CI structurally cannot do this part: a hosted runner has no GPU, so the
build can only `find_spec("vllm")` rather than import it, and cannot see a driver at all.

## Build

CI builds on push to `main` when `Dockerfile`, `requirements.txt`, `scripts/` or the workflow
change; pull requests build without pushing. `workflow_dispatch` offers a `bake_weights`
toggle.

**After the first successful build, set the GHCR package to public** (Packages → the package
→ Package settings → Change visibility). Otherwise the pod needs registry credentials and
every pull is metered.

Images are tagged `latest` and `sha-<commit>`. Reference the SHA tag once you care which
image produced which checkpoint.

### Changing dependencies

Edit `requirements.in`, then:

```bash
make lock      # recompiles for linux/amd64 + py3.12, NOT for this Mac
git commit -am "..." && git push
```

`make lock` passes `--python-platform x86_64-unknown-linux-gnu` explicitly. Without it the
resolver picks macOS wheels and produces a lock that cannot install in the image at all.

## Verification

`scripts/verify_env.py` runs as a build step, so a broken image fails in CI rather than on a
rented GPU. It asserts, rather than prints:

- `glm_ocr` is in transformers' `CONFIG_MAPPING_NAMES` — support landed inside LLaMA-Factory's
  legal transformers range, so a lower resolution produces an image that trains nothing
- `glm_ocr` is in LLaMA-Factory's `TEMPLATES` — every generated `train.yaml` names it, and a
  wrong template trains against the wrong prompt format silently
- `nltk` / `jieba` / `rouge-chinese` import — these are first touched at `eval_steps: 200`,
  i.e. over an hour into a billed run
- the baked checkpoint's `config.json` really says `model_type: glm_ocr` — a partial or
  redirected download leaves files that look fine to `ls`

## Layout

```
Dockerfile              the image
requirements.in         unpinned intent, with the reasoning at each pin
requirements.txt        compiled lock, committed
scripts/verify_env.py   build-time assertions (also runnable on the pod)
scripts/preflight.sh    pod-side checks CI cannot do (driver, real import)
Makefile                make lock / build-local / check
.github/workflows/      the build
```

## Related

The training procedure, what to ship to the box, and the cost model live with the private
project: `docs/cloud_training.md`, `docs/training_setup.md`, `docs/RUNPOD-HANDOFF.md`.
