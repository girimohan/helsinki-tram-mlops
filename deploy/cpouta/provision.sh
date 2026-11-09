#!/usr/bin/env bash
# Create the cPouta resources for the tram collector with the OpenStack CLI.
# This does steps 2-5 of README.md (key pair, SSH rule, VM, volume, floating IP).
#
# Needs: python-openstackclient, and an application credential selected with
# OS_CLOUD (clouds.yaml) or a sourced app-credential openrc file. See README.md.
#
# Usage:  bash deploy/cpouta/provision.sh <your-public-ip>/32 [<another-ip>/32 ...]
#
# Safe to re-run: existing resources are reused, and only missing ones are created.
# Settings can be overridden with environment variables (defaults below).
set -euo pipefail

NAME="${NAME:-tram-collector}"
FLAVOR="${FLAVOR:-standard.small}"
IMAGE="${IMAGE:-Ubuntu-24.04}"
KEY_NAME="${KEY_NAME:-cpouta}"
PUBKEY="${PUBKEY:-$HOME/.ssh/cpouta.pub}"
SECGROUP="${SECGROUP:-tram-ssh}"
VOLUME="${VOLUME:-tram-data}"
VOLUME_GB="${VOLUME_GB:-25}"
EXTERNAL_NET="${EXTERNAL_NET:-public}"
NETWORK="${NETWORK:-}"   # the project's internal network; found automatically if there is only one

log() { echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
# The CLI prints CRLF line endings on Windows; strip the CR so captured values compare cleanly.
os() { openstack "$@" | tr -d '\r'; }
# exists <resource words...> <name>, e.g. `exists security group tram-ssh`
exists() { local n=$#; os "${@:1:n-1}" show "${!n}" >/dev/null 2>&1; }

[[ $# -ge 1 ]] || die "give the IP range(s) allowed to SSH in, e.g. 1.2.3.4/32"
command -v openstack >/dev/null || die "openstack CLI not found (pip install python-openstackclient)"
os token issue -f value -c id >/dev/null || die "cannot authenticate; check OS_CLOUD or the sourced openrc file"

# Key pair
if exists keypair "$KEY_NAME"; then
    log "Key pair $KEY_NAME exists"
else
    [[ -f "$PUBKEY" ]] || die "public key $PUBKEY not found (ssh-keygen -t ed25519 -f ~/.ssh/cpouta)"
    os keypair create --public-key "$PUBKEY" "$KEY_NAME" >/dev/null
    log "Imported key pair $KEY_NAME from $PUBKEY"
fi

# Security group: SSH from the given ranges only
if exists security group "$SECGROUP"; then
    log "Security group $SECGROUP exists"
else
    os security group create --description "SSH to the tram collector" "$SECGROUP" >/dev/null
    log "Created security group $SECGROUP"
fi
rules="$(os security group rule list "$SECGROUP" --ingress --protocol tcp -f value -c "IP Range" -c "Port Range")"
for cidr in "$@"; do
    if grep -q "^$cidr 22:22$" <<<"$rules"; then
        log "SSH already allowed from $cidr"
    else
        os security group rule create --ingress --protocol tcp --dst-port 22 --remote-ip "$cidr" "$SECGROUP" >/dev/null
        log "Allowed SSH from $cidr"
    fi
done

# VM
if exists server "$NAME"; then
    log "Server $NAME exists"
else
    if [[ -z "$NETWORK" ]]; then
        mapfile -t nets < <(os network list --internal -f value -c Name)
        [[ ${#nets[@]} -eq 1 ]] || die "found ${#nets[@]} internal networks (${nets[*]}); set NETWORK=<name>"
        NETWORK="${nets[0]}"
    fi
    log "Creating $NAME ($FLAVOR, $IMAGE) on network $NETWORK; this takes a minute"
    os server create --flavor "$FLAVOR" --image "$IMAGE" --key-name "$KEY_NAME" \
        --security-group default --security-group "$SECGROUP" --network "$NETWORK" \
        --wait "$NAME" >/dev/null
fi
status="$(os server show "$NAME" -f value -c status)"
[[ "$status" == ACTIVE ]] || die "server $NAME is $status, expected ACTIVE"

# Data volume
if exists volume "$VOLUME"; then
    log "Volume $VOLUME exists"
else
    os volume create --size "$VOLUME_GB" --description "Tram collector data" "$VOLUME" >/dev/null
    log "Created ${VOLUME_GB} GB volume $VOLUME"
    for _ in $(seq 30); do
        [[ "$(os volume show "$VOLUME" -f value -c status)" == available ]] && break
        sleep 2
    done
fi
case "$(os volume show "$VOLUME" -f value -c status)" in
    in-use) log "Volume $VOLUME is attached" ;;
    available)
        os server add volume "$NAME" "$VOLUME" >/dev/null
        log "Attached $VOLUME to $NAME" ;;
    *) die "volume $VOLUME is in an unexpected state" ;;
esac

# Floating IP: reuse one already on the VM, then an unused reserved one, else allocate.
port="$(os port list --server "$NAME" -f value -c ID | head -n1)"
ip="$(os floating ip list --port "$port" -f value -c "Floating IP Address" | head -n1)"
if [[ -n "$ip" ]]; then
    log "Floating IP $ip is already associated"
else
    ip="$(os floating ip list --status DOWN -f value -c "Floating IP Address" | head -n1)"
    if [[ -z "$ip" ]]; then
        ip="$(os floating ip create "$EXTERNAL_NET" -f value -c floating_ip_address)"
        log "Allocated floating IP $ip"
    fi
    os server add floating ip "$NAME" "$ip"
    log "Associated floating IP $ip with $NAME"
fi

cat <<EOF

Resources are ready. VM $NAME is at $ip.
Add this to ~/.ssh/config, then continue with step 6 of deploy/cpouta/README.md:

Host tram
    HostName $ip
    User ubuntu
    IdentityFile ~/.ssh/cpouta
    LocalForward 8501 localhost:8501
EOF
