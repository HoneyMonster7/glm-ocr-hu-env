"""Build-time assertions for the training/serving image.

Every check here corresponds to a failure that is otherwise discovered on a
*billed* GPU, usually late: after the dataset has been preprocessed, or at the
first eval step, or per-page during inference. A build failure is free; a
rental failure is not. That asymmetry is the whole rationale for this file.

Runs on a GPU-less CI runner, so nothing may touch the driver. Anything that
needs `libcuda.so.1` is checked with `find_spec` (does the wheel exist and is
it importable-by-path) rather than a real import.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

FAILURES: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - reporting every failure beats stopping at the first
        FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
        return
    print(f"ok    {label}" + (f"  ({detail})" if detail else ""))


# --- the stack imports at all ------------------------------------------------


def _torch():
    import torch

    assert torch.__version__.startswith("2.13."), f"unexpected torch {torch.__version__}"
    return torch.__version__


def _transformers():
    import transformers

    return transformers.__version__


def _llamafactory():
    from llamafactory.extras.env import VERSION

    return VERSION


def _vllm_present():
    # Not a real import: vllm loads compiled extensions that dlopen libcuda.so.1,
    # which does not exist on a CI runner. find_spec confirms the wheel landed;
    # the pod-side preflight in README.md does the real import once, on GPU.
    spec = importlib.util.find_spec("vllm")
    assert spec is not None, "vllm not installed"
    import importlib.metadata as md

    return md.version("vllm")


# --- the two things this project has already been bitten by ------------------


def _glm_ocr_architecture():
    """transformers must know the architecture.

    LLaMA-Factory's pin is `transformers>=4.55.0,<=5.8.0`, and GLM-OCR support
    arrived inside that range -- so a resolver that legally picks a lower
    version produces an image that trains nothing. Documented as a manual step
    in docs/training_setup.md; asserted here so it cannot be skipped.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "glm_ocr" in CONFIG_MAPPING_NAMES, "transformers has no glm_ocr architecture"
    return "glm_ocr in CONFIG_MAPPING_NAMES"


def _glm_ocr_template():
    """LLaMA-Factory must carry the `glm_ocr` chat template.

    `template: glm_ocr` is written into every generated train.yaml. An
    unregistered template name is a late failure with a confusing message, and
    a *wrong* one silently trains against the wrong prompt format.
    """
    from llamafactory.data.template import TEMPLATES

    assert "glm_ocr" in TEMPLATES, f"no glm_ocr template; have {len(TEMPLATES)} templates"
    return "glm_ocr in TEMPLATES"


# --- deps that fail late rather than at import -------------------------------


def _c_compiler():
    """triton JIT-compiles kernels at run time and shells out to a C compiler.

    Nothing in this image is built at build time, which is what lets the base be a plain
    python image rather than a CUDA one -- but triton compiles a C extension the first time a
    kernel is launched, and `python:3.12-slim` ships no compiler. Measured on an L4
    (2026-09-16): the run loaded the model, reported the correct trainable-param count, and
    then died at step 0 with "Failed to find C compiler".

    The *need* only appears on a GPU, so CI cannot catch it by running anything. The
    compiler's presence is checkable anywhere, so it is checked here.
    """
    import sysconfig

    compiler = shutil.which("cc") or shutil.which("gcc")
    assert compiler, "no C compiler on PATH; triton cannot build its kernels"

    # triton builds a CPython extension, so the headers have to be there too.
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    assert header.exists(), f"no {header}; triton's extension build would fail"
    return f"{compiler}, {header.name} present"


def _eval_metrics():
    """nltk / jieba / rouge-chinese are imported at the first eval step.

    train.yaml sets `eval_strategy: steps` with `eval_steps: 200`, so a missing
    one of these crashes the run ~200 optimizer steps in -- well over an hour
    of billed time on the full page set.
    """
    import jieba  # noqa: F401
    import nltk  # noqa: F401
    import rouge_chinese  # noqa: F401

    return "nltk, jieba, rouge-chinese"


def _ocr_pt_deps():
    """ocr-post-training's runtime deps, so the private repo installs --no-deps.

    Resolving these on the box would be a few minutes of GPU-idle billing and
    one more network dependency mid-rental.
    """
    import fitz  # noqa: F401  (pymupdf)
    import fontTools  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import pydantic_settings  # noqa: F401
    import rich  # noqa: F401
    import typer  # noqa: F401

    return "typer, pydantic-settings, rich, pillow, pymupdf, fonttools, numpy"


# --- baked weights -----------------------------------------------------------


def _weights():
    """The base checkpoint, if this image was built with BAKE_WEIGHTS=1.

    Checks model_type rather than mere presence: a partial or redirected
    download leaves files on disk that look fine to `ls`.
    """
    import json

    hub = Path(os.environ.get("HF_HOME", "/opt/hf")) / "hub"
    snaps = sorted(hub.glob("models--zai-org--GLM-OCR/snapshots/*"))
    assert snaps, f"no GLM-OCR snapshot under {hub}"
    snap = snaps[-1]

    config = snap / "config.json"
    assert config.exists(), f"no config.json in {snap}"
    model_type = json.loads(config.read_text()).get("model_type")
    assert model_type == "glm_ocr", f"model_type is {model_type!r}, expected 'glm_ocr'"

    weights = list(snap.glob("*.safetensors"))
    assert weights, f"no .safetensors in {snap}"
    size_gb = sum(w.stat().st_size for w in weights) / 1e9
    return f"{len(weights)} safetensors, {size_gb:.2f} GB, model_type=glm_ocr"


def main() -> int:
    check("torch", _torch)
    check("transformers", _transformers)
    check("llamafactory", _llamafactory)
    check("vllm wheel", _vllm_present)
    check("C compiler for triton", _c_compiler)
    check("glm_ocr architecture", _glm_ocr_architecture)
    check("glm_ocr chat template", _glm_ocr_template)
    check("eval metric deps", _eval_metrics)
    check("ocr-pt runtime deps", _ocr_pt_deps)

    if os.environ.get("BAKE_WEIGHTS", "1") == "1":
        check("baked GLM-OCR weights", _weights)
    else:
        print("skip  baked GLM-OCR weights (BAKE_WEIGHTS=0)")

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
