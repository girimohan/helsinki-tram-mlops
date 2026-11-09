#!/usr/bin/env bash
# Copy raw data and models from the data volume to CSC Allas (S3) with rclone.
# tram-backup.timer runs it hourly. It uses `copy`, never `sync`, so a file
# deleted or damaged on the server is not removed from Allas.
#
# Needs /etc/tram/rclone.conf with an [allas] remote and /etc/tram/backup.env with
# ALLAS_BUCKET=<bucket>; see "Backups" in README.md.
set -euo pipefail

DATA_ROOT=/srv/tram
CONF=/etc/tram/rclone.conf
# shellcheck source=/dev/null
. /etc/tram/backup.env
: "${ALLAS_BUCKET:?set ALLAS_BUCKET in /etc/tram/backup.env}"

mountpoint -q "$DATA_ROOT" || { echo "$DATA_ROOT is not mounted; nothing to back up" >&2; exit 1; }

rc() { rclone --config "$CONF" --stats-one-line --stats-log-level NOTICE --stats 1h "$@"; }

# Finished days and models are copied as they are.
today="tram_$(date -u +%F).jsonl"
rc copy "$DATA_ROOT/data/raw" "allas:$ALLAS_BUCKET/data/raw" --exclude "$today"
rc copy "$DATA_ROOT/models" "allas:$ALLAS_BUCKET/models"

# The collector keeps appending to today's file, and rclone refuses to upload a file
# that changes mid-transfer, so upload a snapshot. A partial last line is harmless:
# processing skips malformed lines.
if [[ -f "$DATA_ROOT/data/raw/$today" ]]; then
    snapshot="$(mktemp -d)"
    trap 'rm -rf "$snapshot"' EXIT
    cp "$DATA_ROOT/data/raw/$today" "$snapshot/$today"
    rc copyto "$snapshot/$today" "allas:$ALLAS_BUCKET/data/raw/$today"
fi
echo "Backed up $DATA_ROOT/data/raw and $DATA_ROOT/models to allas:$ALLAS_BUCKET"
