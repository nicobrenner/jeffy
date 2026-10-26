#!/usr/bin/env bash
# Where is every copy of Jeffy right now?
#
# The a19/a21 drift (PyPI at a21 for a day while the playground still served
# a19, so support_ticket_v2 404'd for anyone we had told about it) was invisible
# because nothing ever compared the two. This compares them.
#
#   scripts/version_status.sh
#
# Exits 1 if any published surface lags the repo, so it works in a check too.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROPLET="${JEFFY_DROPLET:-142.93.5.94}"
PLAYGROUND="https://playground.jeffyclassify.com"
NODE_REPO="${HOME}/work/n8n-nodes-jeffy"

ok=0

say() { printf '%-22s %s\n' "$1" "$2"; }

# --- repo ----------------------------------------------------------------
local_ver=$(grep -m1 '^version' "$REPO/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
say "pyproject.toml" "$local_ver"

# --- PyPI ----------------------------------------------------------------
pypi_ver=$(curl -fsS -m 20 https://pypi.org/pypi/jeffy-classify/json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null) \
  || pypi_ver="unreachable"
say "PyPI" "$pypi_ver"
[ "$pypi_ver" = "$local_ver" ] || { say "" "  ^ differs from pyproject"; ok=1; }

# --- droplet -------------------------------------------------------------
drop_ver=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "root@$DROPLET" \
  '/opt/jeffy/venv/bin/pip show jeffy-classify 2>/dev/null | sed -n "s/^Version: //p"' 2>/dev/null) \
  || drop_ver="unreachable"
say "droplet venv" "${drop_ver:-unreachable}"
[ "$drop_ver" = "$local_ver" ] || { say "" "  ^ run scripts/deploy_droplet.sh"; ok=1; }

# --- what the live service actually serves -------------------------------
# pip show can be ahead of the running process if nobody restarted the unit.
live_ver=$(curl -fsS -m 25 "$PLAYGROUND/v1/version" 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version","?"))' 2>/dev/null)
if [ -z "${live_ver:-}" ]; then
  # No version endpoint: fall back to probing a feature we know a21 added.
  code=$(curl -fsS -m 25 -o /dev/null -w '%{http_code}' \
    "$PLAYGROUND/v1/capabilities/support_ticket_v2" 2>/dev/null) || code="???"
  live_ver="(no /v1/version; support_ticket_v2 -> HTTP $code)"
fi
say "playground serving" "$live_ver"

# --- site ----------------------------------------------------------------
site_sha=$(cd "$REPO/site" && git rev-parse --short HEAD 2>/dev/null)
site_dirty=$(cd "$REPO/site" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
say "site HEAD" "$site_sha ($site_dirty uncommitted)"

# --- n8n node ------------------------------------------------------------
if [ -f "$NODE_REPO/package.json" ]; then
  node_local=$(python3 -c "import json; print(json.load(open('$NODE_REPO/package.json'))['version'])" 2>/dev/null)
  node_npm=$(curl -fsS -m 20 https://registry.npmjs.org/n8n-nodes-jeffy 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["dist-tags"]["latest"])' 2>/dev/null) \
    || node_npm="unreachable"
  say "n8n node" "local $node_local / npm $node_npm"
  [ "$node_local" = "$node_npm" ] || ok=1
fi

exit $ok
