#!/usr/bin/env bash
# Turn on hourly backups to CSC Allas using the S3 keys that CSC's allas_conf saved
# for the login user (the [s3allas] remote in ~/.config/rclone/rclone.conf).
#
# On the VM:
#   source ~/allas-cli-utils/allas_conf -u <csc-username> -p project_2022568   # once; asks for your CSC password
#   sudo bash deploy/cpouta/enable-backups.sh [bucket-name]
#
# Bucket names are global across Allas and visible to other users. Safe to re-run.
set -euo pipefail

BUCKET="${1:-tram-mlops-2022568}"
LOGIN_USER="${SUDO_USER:-ubuntu}"
USER_CONF="$(getent passwd "$LOGIN_USER" | cut -d: -f6)/.config/rclone/rclone.conf"
CONF=/etc/tram/rclone.conf

log() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"
grep -qx '\[s3allas\]' "$USER_CONF" 2>/dev/null \
    || die "no [s3allas] remote in $USER_CONF; run allas_conf first (see the comment at the top)"

# Copy only the [s3allas] section, renamed to [allas], into a root-only file.
install -d -m 700 /etc/tram
( umask 077
  awk '/^\[/ { keep = ($0 == "[s3allas]") } keep' "$USER_CONF" | sed '1s/.*/[allas]/' > "$CONF"
  echo "ALLAS_BUCKET=$BUCKET" > /etc/tram/backup.env )
log "Wrote $CONF and /etc/tram/backup.env (root only)"

rclone --config "$CONF" lsd allas: >/dev/null || die "Allas rejected the keys in $CONF"
if rclone --config "$CONF" lsd allas: | awk '{print $NF}' | grep -qx "$BUCKET"; then
    log "Bucket $BUCKET exists"
else
    rclone --config "$CONF" mkdir "allas:$BUCKET" || die "could not create bucket $BUCKET; the name may be taken, pass another one"
    log "Created bucket $BUCKET"
fi

systemctl enable --now tram-backup.timer >/dev/null 2>&1
log "Running the first backup"
ok=true
systemctl start tram-backup.service || ok=false
journalctl -u tram-backup.service -n 5 -o cat --no-pager
$ok || die "the first backup failed; see the log above"
rclone --config "$CONF" size "allas:$BUCKET"
log "Hourly backups enabled: systemctl list-timers tram-backup.timer"
