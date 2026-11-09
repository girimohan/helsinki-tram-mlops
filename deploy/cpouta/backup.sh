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

for dir in data/raw models; do
    rclone --config "$CONF" copy "$DATA_ROOT/$dir" "allas:$ALLAS_BUCKET/$dir" \
        --stats-one-line --stats-log-level NOTICE --stats 1h   # one summary line per run
done
echo "Backed up $DATA_ROOT/{data/raw,models} to allas:$ALLAS_BUCKET"
