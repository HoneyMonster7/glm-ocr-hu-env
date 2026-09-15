#!/usr/bin/env bash
# Container entrypoint. Starts sshd and stays alive.
#
# Both halves matter on RunPod. A pod whose container process exits is a pod that stops, so
# `CMD ["/bin/bash"]` -- fine for `docker run -it` -- terminates the instant it starts with no
# TTY attached, and the rental dies before you can look at it. And RunPod's SSH access needs
# a real sshd listening on 22 inside the container; the web terminal goes through their agent,
# but `ssh` and `scp` (which is how the ~1 GB bundle arrives) do not.
#
# RunPod's documented recipe installs openssh-server at pod start. That is ~20 s of billed
# time, a network dependency at the worst moment, and an apt repo that has to be up. It is
# baked into the image instead; this script only does the parts that need the pod's own
# environment.
set -euo pipefail

# RunPod injects the account's public key here. Absent on a plain `docker run`, which is fine:
# that path uses `docker exec`, not ssh.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Host keys are generated here rather than baked: an image-wide host key would be shared by
# every pod anyone ever launches from it, which is both a warning on every connection and a
# genuine impersonation risk on a public image.
ssh-keygen -A
mkdir -p /run/sshd
/usr/sbin/sshd -D -e &

echo "glm-ocr-hu-env ready. Next: bash /opt/app/preflight.sh"

# Keep the container alive. `wait` rather than `sleep infinity` so a dying sshd surfaces
# instead of leaving a pod that looks healthy and refuses every connection.
wait -n
