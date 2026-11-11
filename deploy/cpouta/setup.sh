#!/usr/bin/env bash
# Prepare an Ubuntu 24.04 cPouta VM to run the tram collector and dashboard.
#
# Run on the VM from a clone of this repository:
#   sudo bash deploy/cpouta/setup.sh /dev/vdb            # first run: set up the data volume, start the services
#   sudo bash deploy/cpouta/setup.sh --public <hostname> # also serve the dashboard over HTTPS at <hostname>
#   sudo bash deploy/cpouta/setup.sh                     # later runs: re-apply settings, rebuild and restart
#
# Safe to re-run. It formats the volume only if it is completely blank, and reuses an
# existing filesystem as is, so a volume moved from an older VM keeps its data.
# Settings are kept in .env next to docker-compose.yml.
set -euo pipefail

DATA_ROOT=/srv/tram
APP_UID=1000   # the "app" user inside the Docker image
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGIN_USER="${SUDO_USER:-ubuntu}"
DEVICE=""
PUBLIC_HOST=""

log() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --public) PUBLIC_HOST="${2:?--public needs a hostname}"; shift 2 ;;
        /dev/*) DEVICE="$1"; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ $EUID -eq 0 ]] || die "run with sudo"

install_packages() {
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        log "Docker is already installed"
    else
        log "Installing Docker from Docker's apt repository"
        apt-get update -q
        apt-get install -yq ca-certificates curl
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc
        # shellcheck source=/dev/null
        . /etc/os-release
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
            "https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
            > /etc/apt/sources.list.d/docker.list
        apt-get update -q
        apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        systemctl enable --now docker
        usermod -aG docker "$LOGIN_USER"
    fi
    if ! command -v rclone >/dev/null 2>&1 || ! command -v s3cmd >/dev/null 2>&1; then
        # rclone runs the backups; CSC's allas_conf needs the others to create the S3 keys.
        log "Installing rclone and the Allas tools"
        apt-get install -yq rclone s3cmd restic python3-openstackclient python3-swiftclient
    fi
}

mount_volume() {
    if mountpoint -q "$DATA_ROOT"; then
        log "Data volume already mounted at $DATA_ROOT"
        return
    fi
    [[ -n "$DEVICE" ]] || die "nothing is mounted at $DATA_ROOT. Pass the volume's device, e.g.
    sudo bash $0 /dev/vdb
Find it with: lsblk   (the disk whose size matches the volume you attached)"
    [[ -b "$DEVICE" ]] || die "$DEVICE is not a block device"
    if lsblk -nro MOUNTPOINT "$DEVICE" | grep -q .; then
        die "$DEVICE (or a partition on it) is already mounted; is this the system disk?"
    fi

    local fstype
    fstype="$(blkid -o value -s TYPE "$DEVICE" || true)"
    if [[ -z "$fstype" ]]; then
        # No filesystem. Only format if there is no partition table or other signature either.
        [[ -z "$(wipefs -n "$DEVICE")" ]] || die "$DEVICE has a partition table or unknown data; not formatting it"
        log "Formatting blank volume $DEVICE as ext4"
        mkfs.ext4 -q -L tramdata "$DEVICE"
        fstype=ext4
    else
        log "Reusing existing $fstype filesystem on $DEVICE"
    fi

    local uuid
    uuid="$(blkid -o value -s UUID "$DEVICE")"
    mkdir -p "$DATA_ROOT"
    if ! grep -q "UUID=$uuid" /etc/fstab; then
        # nofail: the VM still boots if the volume is ever detached.
        echo "UUID=$uuid $DATA_ROOT $fstype defaults,nofail 0 2" >> /etc/fstab
    fi
    systemctl daemon-reload
    mount "$DATA_ROOT"
    log "Mounted $DEVICE at $DATA_ROOT"
}

# set_env KEY VALUE: add or replace one line in .env
set_env() {
    local file="$REPO_DIR/.env"
    touch "$file"
    if grep -q "^$1=" "$file"; then
        sed -i "s|^$1=.*|$1=$2|" "$file"
    else
        echo "$1=$2" >> "$file"
    fi
}

configure() {
    mkdir -p "$DATA_ROOT/data/raw" "$DATA_ROOT/data/processed" "$DATA_ROOT/models"
    chown -R "$APP_UID:$APP_UID" "$DATA_ROOT/data" "$DATA_ROOT/models"

    set_env TRAM_DATA_DIR "$DATA_ROOT/data"
    set_env TRAM_MODELS_DIR "$DATA_ROOT/models"
    set_env DASHBOARD_BIND 127.0.0.1
    # On the server, page views must never start a processing run (see configs/config.toml).
    set_env TRAM_DASHBOARD_PROCESS_ON_DEMAND false
    if [[ -n "$PUBLIC_HOST" ]]; then
        set_env TRAM_PUBLIC_HOST "$PUBLIC_HOST"
        set_env COMPOSE_PROFILES public
    fi
    chown "$LOGIN_USER:" "$REPO_DIR/.env"
    log "Settings in $REPO_DIR/.env:"
    sed 's/^/      /' "$REPO_DIR/.env"
}

install_timers() {
    local unit
    for unit in "$REPO_DIR"/deploy/cpouta/systemd/*; do
        sed -e "s|@REPO_DIR@|$REPO_DIR|g" -e "s|@USER@|$LOGIN_USER|g" "$unit" \
            > "/etc/systemd/system/$(basename "$unit")"
    done
    systemctl daemon-reload
    systemctl enable --now tram-autodeploy.timer >/dev/null 2>&1
    log "Automatic deploys enabled (tram-autodeploy.timer, every 5 min)"
    systemctl enable --now tram-process.timer >/dev/null 2>&1
    log "Hourly processing enabled (tram-process.timer)"
    if [[ -f /etc/tram/rclone.conf && -f /etc/tram/backup.env ]]; then
        systemctl enable --now tram-backup.timer >/dev/null 2>&1
        log "Hourly backups to Allas enabled (tram-backup.timer)"
    else
        log "Backups not enabled yet: /etc/tram/rclone.conf and backup.env are missing (see README)"
    fi
}

start_services() {
    log "Building and starting the services"
    cd "$REPO_DIR"
    docker compose up -d --build --remove-orphans
    docker compose ps
}

install_packages
mount_volume
configure
install_timers
start_services

cat <<EOF

Done. Raw data is written to $DATA_ROOT/data/raw (one file per UTC day).
  Collector log:   docker compose logs -f ingest      (log out and back in once to use docker without sudo)
  Deploy log:      journalctl -u tram-autodeploy -n 20
  Disk usage:      df -h $DATA_ROOT
  Dashboard:       ssh tunnel to localhost:8501$( [[ -n "$PUBLIC_HOST" ]] && echo ", or https://$PUBLIC_HOST" )
EOF
