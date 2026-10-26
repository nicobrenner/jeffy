#!/usr/bin/env bash
# Deploy the current PyPI release to the production playground.
#
# This is step 7 of a release and it is not optional. PyPI going to a21 while
# the droplet kept serving a19 meant support_ticket_v2 returned 404 to the one
# user who had been told it existed.
#
#   scripts/deploy_droplet.sh            # deploy whatever pyproject says
#   scripts/deploy_droplet.sh 0.1.0a21   # deploy an explicit version
#
# Demo engines (vizdoom, flygym, mujoco) are installed separately and are NOT
# touched here; MUJOCO_GL=osmesa lives in the systemd unit because the droplet
# has no GPU and the EGL path fails on it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROPLET="${JEFFY_DROPLET:-142.93.5.94}"
PLAYGROUND="https://playground.jeffyclassify.com"
VENV=/opt/jeffy/venv

TARGET="${1:-$(grep -m1 '^version' "$REPO/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')}"
echo "target version: $TARGET"

# 1. It has to exist on PyPI first. Deploying something unpublished would make
#    prod the only place the code runs, which is how you lose it.
pypi=$(curl -fsS -m 20 https://pypi.org/pypi/jeffy-classify/json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')
echo "pypi latest:    $pypi"
if ! curl -fsS -m 20 "https://pypi.org/pypi/jeffy-classify/$TARGET/json" >/dev/null; then
  echo "ERROR: $TARGET is not on PyPI yet. Publish first (see the release skill)." >&2
  exit 1
fi

before=$(ssh -o ConnectTimeout=10 "root@$DROPLET" \
  "$VENV/bin/pip show jeffy-classify 2>/dev/null | sed -n 's/^Version: //p'")
echo "droplet before: ${before:-none}"
if [ "$before" = "$TARGET" ]; then
  echo "already at $TARGET; restarting anyway to pick up unit changes"
fi

# 2. Install, then restart. Install first so the download is not part of the
#    downtime window.
ssh -o ConnectTimeout=10 "root@$DROPLET" \
  "$VENV/bin/pip install -q --upgrade 'jeffy-classify==$TARGET'"

after=$(ssh -o ConnectTimeout=10 "root@$DROPLET" \
  "$VENV/bin/pip show jeffy-classify | sed -n 's/^Version: //p'")
if [ "$after" != "$TARGET" ]; then
  echo "ERROR: droplet reports $after after install, wanted $TARGET" >&2
  exit 1
fi

ssh -o ConnectTimeout=10 "root@$DROPLET" 'systemctl restart jeffy'
echo "restarted; waiting for the encoder to load"

# 3. Verify from outside. pip show only proves what is on disk.
for i in $(seq 1 12); do
  sleep 10
  live=$(curl -fsS -m 25 "$PLAYGROUND/v1/version" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null) || live=""
  [ -n "$live" ] && break
  echo "  ...not answering yet ($i/12)"
done

if [ -z "${live:-}" ]; then
  # Pre-0.1.0a22 builds have no /v1/version. Fall back to capabilities.
  if curl -fsS -m 25 "$PLAYGROUND/v1/capabilities" >/dev/null 2>&1; then
    echo "WARNING: no /v1/version on this build; /v1/capabilities answers, so it is up"
    live="(unreported)"
  else
    echo "ERROR: playground is not answering after restart" >&2
    ssh -o ConnectTimeout=10 "root@$DROPLET" 'journalctl -u jeffy -n 30 --no-pager' >&2
    exit 1
  fi
elif [ "$live" != "$TARGET" ]; then
  echo "ERROR: playground serves $live, wanted $TARGET" >&2
  exit 1
fi

n_caps=$(curl -fsS -m 25 "$PLAYGROUND/v1/capabilities" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["count"])')

echo
echo "deployed:   $before -> $after"
echo "serving:    $live"
echo "capabilities: $n_caps"
echo
echo "Now run scripts/version_status.sh and confirm every row matches."
