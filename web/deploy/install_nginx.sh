#!/usr/bin/env bash
# One-shot nginx install for ADAN0 Terminal. Run with sudo.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)/adan-terminal.nginx.conf"
DST_AVAIL="/etc/nginx/sites-available/adan-terminal"
DST_ENABLED="/etc/nginx/sites-enabled/adan-terminal"
cp "$SRC" "$DST_AVAIL"
ln -sf "$DST_AVAIL" "$DST_ENABLED"
nginx -t
systemctl reload nginx
echo "OK: ADAN0 Terminal proxied on http://adan.novahosting.site (port 80)."
echo "For HTTPS: sudo certbot --nginx -d adan.novahosting.site"
