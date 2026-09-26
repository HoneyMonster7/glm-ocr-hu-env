#!/usr/bin/env bash
# Install RunPod's injected public keys into /root/.ssh/authorized_keys, and SAY what was
# installed. Called by start.sh; separate so it can be tested without a pod.
#
#   AUTH_KEYS=/tmp/x PUBLIC_KEY='...' bash install_ssh_keys.sh
#
# Written after the 2026-09-26 release run, where the pod refused the one key it should have
# accepted and nothing on the box said why. RunPod's docs name two ways that happens, and this
# handles both instead of assuming the variable holds exactly one well-formed key:
#
# - **Several keys pasted into account settings without a newline between them.** RunPod: "only
#   the first key will work". The second key lands inside the first key's comment field, where
#   sshd never looks. Any key-type token appearing mid-line starts a new line here.
# - **The per-pod `SSH_PUBLIC_KEY` override**, which replaces the account key for one pod. The old
#   start.sh read only `PUBLIC_KEY`, so an override was silently dropped. Both are read now.
#
# Every installed key's fingerprint goes to the container log, which the RunPod console shows
# without ssh -- so a refused key can be diagnosed from the console by comparing fingerprints
# with `ssh-keygen -lf ~/.ssh/<key>.pub` locally.
set -euo pipefail

AUTH_KEYS="${AUTH_KEYS:-/root/.ssh/authorized_keys}"
RAW="$(printf '%s\n%s\n' "${PUBLIC_KEY:-}" "${SSH_PUBLIC_KEY:-}")"

# Literal "\n" (two characters) as a separator, then key types run together on one line.
KEYS="$(printf '%s\n' "$RAW" \
    | sed 's/\\n/\n/g' \
    | sed -E 's/[[:space:]]+(ssh-ed25519|ssh-rsa|ssh-dss|ecdsa-sha2-[a-z0-9-]+|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-[a-z0-9@.-]+)[[:space:]]+/\n\1 /g' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
    | grep -E '^(ssh-|ecdsa-|sk-)' || true)"

if [ -z "$KEYS" ]; then
    echo "install_ssh_keys: WARNING no PUBLIC_KEY or SSH_PUBLIC_KEY -- ssh will refuse every key." \
         "Add one under RunPod Settings -> SSH Public Keys, or paste it via the web terminal." >&2
    exit 0
fi

mkdir -p "$(dirname "$AUTH_KEYS")"
chmod 700 "$(dirname "$AUTH_KEYS")"
touch "$AUTH_KEYS"
# Append only keys not already present, so a container restart does not pile up duplicates.
while IFS= read -r key; do
    grep -qxF "$key" "$AUTH_KEYS" || printf '%s\n' "$key" >> "$AUTH_KEYS"
done <<< "$KEYS"
chmod 600 "$AUTH_KEYS"

count=$(grep -cE '^(ssh-|ecdsa-|sk-)' "$AUTH_KEYS" || true)
echo "install_ssh_keys: $count key(s) in $AUTH_KEYS:"
ssh-keygen -lf "$AUTH_KEYS" | sed 's/^/  /'
