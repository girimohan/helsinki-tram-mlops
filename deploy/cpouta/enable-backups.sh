#!/usr/bin/env bash
# Turn on hourly backups to CSC Allas using the S3 keys that CSC's allas_conf saved
# for the login user (in ~/.aws/credentials).
#
# On the VM:
#   source ~/allas-cli-utils/allas_conf -u <csc-username> -p project_2022568   # once; asks for your CSC password
#   sudo bash deploy/cpouta/enable-backups.sh [bucket-name]
#
# Bucket names are global across Allas and visible to other users. Safe to re-run.
set -euo pipefail

BUCKET="${1:-tram-mlops-2022568}"
LOGIN_USER="${SUDO_USER:-ubuntu}"
USER_HOME="$(getent passwd "$LOGIN_USER" | cut -d: -f6)"
CREDS="$USER_HOME/.aws/credentials"   # written by allas_conf
CONF=/etc/tram/rclone.conf

log() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"
[[ -f "$CREDS" ]] || die "no $CREDS; run allas_conf first (see the comment at the top)"

# allas_conf writes KEY="value" lines; also accept the usual aws_access_key_id = value.
read_key() {
    sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"?([^\"[:space:]]*)\"?[[:space:]]*$/\1/Ip" "$CREDS" | head -n1
}
access="$(read_key AWS_ACCESS_KEY_ID)"
secret="$(read_key AWS_SECRET_ACCESS_KEY)"
[[ -n "$access" && -n "$secret" ]] || die "could not read the S3 keys from $CREDS"

install -d -m 700 /etc/tram
( umask 077
  cat > "$CONF" <<EOF
[allas]
type = s3
provider = Other
env_auth = false
endpoint = a3s.fi
acl = private
access_key_id = $access
secret_access_key = $secret
EOF
  echo "ALLAS_BUCKET=$BUCKET" > /etc/tram/backup.env )
log "Wrote $CONF and /etc/tram/backup.env (root only)"

# allas_conf leaves some of its copies of the keys readable by other users; tighten them.
chmod 600 "$USER_HOME"/.s3cfg_* "$USER_HOME/.config/rclone/rclone.conf" 2>/dev/null || true

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
