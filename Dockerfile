# syntax=docker/dockerfile:1.7

# Base is a plain Python image, NOT pytorch/pytorch or nvidia/cuda, on purpose.
# Nothing here is compiled -- torch, vllm and flashinfer all install as wheels
# that carry their own CUDA runtime (the nvidia-* packages in requirements.txt).
# A CUDA base image would add ~3 GB of toolkit that the wheels then shadow.
# The only thing that must come from the host is the driver, which the NVIDIA
# container runtime injects at `docker run` time.
#
# Pinned by digest, not by tag. Tags are mutable, and a silent base refresh that
# moves the Python patch or the C++ ABI breaks the wheels weeks later with an
# "undefined symbol" error that looks like anything but a base-image change.
# Digest is the multi-arch index, resolved 2026-09-14; the build pins
# --platform linux/amd64, so it lands on the amd64 manifest beneath it.
FROM python:3.12-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# git: pip installs LLaMA-Factory from a git ref.
# curl, ca-certificates: dataset pull on the pod.
# build-essential: NOT for building anything here -- triton JIT-compiles CUDA kernels at RUN
#   time and shells out to `cc`, so without it the first training step dies with
#   "Failed to find C compiler" after the model has loaded. Measured on an L4, 2026-09-16:
#   the run reached `trainable params: 3,735,552` and then failed at step 0. This is the one
#   thing the CUDA-less base image costs, and it cannot be caught by any GPU-less CI check,
#   because nothing compiles a kernel until there is a device to compile it for.
# libnuma1, libgomp1: vllm's and torch's threading/NUMA paths dlopen these.
# openssh-server: RunPod's ssh/scp access needs a real sshd in the container. Their documented
#   recipe apt-installs it at pod start; baking it trades ~40 MB of image for ~20 s of billed
#   time and one fewer network dependency at the moment you can least afford one.
# rsync: scp'ing ~1 GB of pages over a flaky link, resumably.
# No libGL: the resolve lands on opencv-python-headless, which does not need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libnuma1 libgomp1 openssh-server rsync \
 && rm -rf /var/lib/apt/lists/*

# Shell tooling, so the pod is usable from RunPod's web terminal without apt-getting first.
# A few MB against a 6.5 GB image; the debugging time it saves is worth more than the bytes.
#
# NOT coreutils or procps -- the slim base already has both, so `tail`, `head`, `ps`, `top` and
# `watch` all work. What it actually lacks is a pager and an editor. Checked on a live pod
# rather than guessed (2026-09-16).
#
# tmux is the one that changes the workflow: a training run inside it survives a dropped ssh
# connection, which on a multi-hour run is the difference between reconnecting and re-renting.
# nvtop watches the GPU the way top watches the CPU -- the fastest way to see whether a run is
# saturating the card or waiting on the dataloader.
RUN apt-get update && apt-get install -y --no-install-recommends \
        less vim-tiny tmux htop nvtop jq tree \
 && ln -sf /usr/bin/vim.tiny /usr/bin/vim \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

# HF_HOME under /opt, like everything else: if a RunPod network volume is ever
# attached it mounts over /workspace and anything baked there vanishes.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /opt/app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Expose the CUDA toolkit that the nvidia-* wheels already installed.
#
# `nvidia-cuda-nvcc` puts a complete toolkit -- bin/, include/, lib/, nvvm/ -- under
# site-packages, but every library that shells out to a CUDA compiler looks for `nvcc` on PATH
# or at $CUDA_HOME, defaulting to /usr/local/cuda. A symlink costs nothing and makes the
# toolkit findable, where a nvidia/cuda base image would have added ~3 GB of duplicate.
RUN ln -sfn /usr/local/lib/python3.12/site-packages/nvidia/cu13 /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

# Do not let vllm JIT-compile flashinfer's sampling kernels on startup.
#
# Measured on an L4, 2026-09-16: with nvcc reachable it still fails, because pip's split CUDA
# packages disagree -- nvidia-cuda-nvcc is 13.4.59 while nvidia-cuda-runtime is 13.0.96, and
# flashinfer's build stops at
#   "CUDA compiler and CUDA toolkit headers are incompatible".
#
# Disabling it costs this project nothing measurable. `predict` decodes at temperature 0 with
# a fixed seed (models/glm_ocr_vllm.py sends `seed` because temperature 0 alone was not
# reproducible over this transport), so a faster top-k/top-p sampler optimises work we never
# do -- the cost is the prefill of a ~2,500-token page image. The AOT alternative
# (`python -m flashinfer.aot` across four target arches at build time) was considered and
# rejected: real build complexity to precompute kernels for a sampler this project does not use.
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# Bake the base weights (~2 GB). They are public and immutable, and pulling them
# on the pod is billed GPU-idle time plus one more thing that can fail mid-rental.
#
# Downloaded whole, with no --include filter: the repo is one model.safetensors
# plus small files, and a filter is how you lose chat_template.jinja -- which is
# not matched by any obvious "*.json / *.safetensors" pattern and whose absence
# would misformat every training prompt rather than raise.
ARG BAKE_WEIGHTS=1
ENV BAKE_WEIGHTS=${BAKE_WEIGHTS}
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        hf download zai-org/GLM-OCR --exclude ".eval_results/*" ; \
    fi

# Fail the build, not the rental. Each of these has a known way of going wrong
# silently, so they are assertions rather than prints.
COPY scripts/ /opt/app/
RUN chmod +x /opt/app/preflight.sh /opt/app/start.sh && python /opt/app/verify_env.py

# Sanity: the trainer's entrypoint must exist on PATH, since `ocr-pt train`
# shells out to it by name and would otherwise fail after dataset preprocessing.
RUN llamafactory-cli version

# Not ["/bin/bash"]: a pod whose container process exits is a pod that stops, and bash with
# no TTY exits immediately. start.sh brings up sshd and blocks. See the script.
EXPOSE 22
CMD ["/opt/app/start.sh"]
