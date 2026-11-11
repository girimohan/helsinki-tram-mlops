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

# Finished days and models are copied as they are. Raw files are named by UTC day
# (tram_, events_, gtfsrt/trip_updates_, gtfsrt/alerts_), so today's are the ones still growing.
today="$(date -u +%F)"
rc copy "$DATA_ROOT/data/raw" "allas:$ALLAS_BUCKET/data/raw" --exclude "*${today}*"
rc copy "$DATA_ROOT/models" "allas:$ALLAS_BUCKET/models"

# rclone refuses to upload a file that changes mid-transfer, so upload snapshots of
# today's files. A partial last line (or gzip member) is harmless: readers skip it.
snapshot="$(mktemp -d)"
trap 'rm -rf "$snapshot"' EXIT
(cd "$DATA_ROOT/data/raw" && find . -type f -name "*${today}*" -exec cp --parents {} "$snapshot/" \;)
rc copy "$snapshot" "allas:$ALLAS_BUCKET/data/raw"
echo "Backed up $DATA_ROOT/data/raw and $DATA_ROOT/models to allas:$ALLAS_BUCKET"
