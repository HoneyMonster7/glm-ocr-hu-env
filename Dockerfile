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
# libnuma1, libgomp1: vllm's and torch's threading/NUMA paths dlopen these.
# openssh-server: RunPod's ssh/scp access needs a real sshd in the container. Their documented
#   recipe apt-installs it at pod start; baking it trades ~40 MB of image for ~20 s of billed
#   time and one fewer network dependency at the moment you can least afford one.
# rsync: scp'ing ~1 GB of pages over a flaky link, resumably.
# No libGL: the resolve lands on opencv-python-headless, which does not need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libnuma1 libgomp1 openssh-server rsync \
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
