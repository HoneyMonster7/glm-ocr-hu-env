#!/usr/bin/env bash
# First command to run on a freshly rented pod. Everything it checks is
# something CI structurally cannot: there is no GPU on a hosted runner.
#
# Costs ~30 seconds of billed time and is worth it -- the failure it is looking
# for (driver too old for the CUDA 13 wheels) otherwise surfaces as a torch
# import error after you have already uploaded the dataset.
set -euo pipefail

echo "=== host ==="
nproc
free -g | head -2
df -h / /opt | tail -3

echo
echo "=== driver ==="
nvidia-smi
# The wheels in this image are CUDA 13 builds, which need driver r580 or newer.
# CUDA is only minor-version compatible, so r5xx < 580 fails -- and RunPod's
# Community cloud does carry older hosts. Filter for CUDA 13 when renting.
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
major=${driver%%.*}
if [ "$major" -lt 580 ]; then
    echo "FATAL: driver $driver is older than r580; the CUDA 13 wheels will not load." >&2
    echo "Rent a host advertising CUDA 13 support, or rebuild the image on cu12 wheels." >&2
    exit 1
fi
echo "driver $driver -- ok for CUDA 13"

echo
echo "=== the real imports (the part CI cannot do) ==="
python - <<'PY'
import torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("device         ", torch.cuda.get_device_name(0))
print("capability     ", torch.cuda.get_device_capability(0))
print("vram GB        ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

import vllm  # the import CI skips: it dlopens libcuda.so.1
print("vllm           ", vllm.__version__)

import llamafactory
from llamafactory.data.template import TEMPLATES
print("llamafactory   ", llamafactory.__version__ if hasattr(llamafactory, "__version__") else "ok")
assert "glm_ocr" in TEMPLATES
print("glm_ocr template ok")
PY

echo
echo "=== baked weights ==="
python /opt/app/verify_env.py

echo
echo "preflight passed"
