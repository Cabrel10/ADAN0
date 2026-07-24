#!/usr/bin/env bash
# Expose the ADAN Mission Control backend (port 8770) publicly WITHOUT sudo,
# using a Cloudflare quick tunnel (account-less). No nginx / firewall changes.
#
# Why: on this VPS only ports 80/443 are open externally (nginx vhosts owned by
# root, no write access, no sudo). cloudflared runs entirely in user space and
# punches out to Cloudflare's edge, returning a public https://*.trycloudflare.com URL.
#
# Usage:  bash web/deploy/start_tunnel.sh
# The public URL is printed in the log (grep for trycloudflare.com).
set -euo pipefail

PORT="${ADAN_PORT:-8770}"
CF="${HOME}/bin/cloudflared"
LOG="${HOME}/adan_tunnel.log"

if [ ! -x "$CF" ]; then
  echo "[tunnel] downloading cloudflared to $CF ..."
  mkdir -p "${HOME}/bin"
  curl -sL -o "$CF" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CF"
fi

echo "[tunnel] starting quick tunnel -> http://127.0.0.1:${PORT}"
nohup "$CF" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate \
  >"$LOG" 2>&1 &
echo "[tunnel] pid $! · log: $LOG"
sleep 8
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 \
  || echo "[tunnel] URL not ready yet — tail $LOG"
