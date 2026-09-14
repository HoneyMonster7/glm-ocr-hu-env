.PHONY: lock build-local shell check

# Recompile the lock. Targets linux/amd64 + py3.12 explicitly, because this is
# normally run on an arm64 Mac and the resolver would otherwise pick macOS
# wheels -- producing a lock that cannot install in the image at all.
lock:
	uv pip compile requirements.in -o requirements.txt \
		--python-platform x86_64-unknown-linux-gnu \
		--python-version 3.12 \
		--no-header

# Cross-build under QEMU. Slow (tens of minutes) and the result is UNTESTABLE
# here -- no NVIDIA container runtime on macOS, so nothing CUDA can run. Useful
# only to see whether the layers assemble before spending a CI round trip.
build-local:
	docker build --platform linux/amd64 --build-arg BAKE_WEIGHTS=0 -t glm-ocr-hu-env:dev .

shell:
	docker run --rm -it --platform linux/amd64 glm-ocr-hu-env:dev bash

# Pull the published image and re-run the assertions. Needs a CUDA host.
check:
	docker run --rm --gpus all ghcr.io/HoneyMonster7/glm-ocr-hu-env:latest \
		python /opt/app/verify_env.py
