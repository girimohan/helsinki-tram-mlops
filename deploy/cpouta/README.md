# Running the collector on CSC cPouta

This guide sets up one small virtual machine (VM) on [cPouta](https://docs.csc.fi/cloud/pouta/) (CSC's OpenStack cloud). The VM runs the collector around the clock and the dashboard next to it. It is roadmap item [E3](../../docs/ROADMAP.md#e3-always-on-host-on-csc-cpouta), phase 1.

```
HSL MQTT ──► VM "tram-collector" (Ubuntu 24.04, standard.small)
               ├─ ingest     (Docker, restarts automatically)  ──► /srv/tram/data/raw/tram_YYYY-MM-DD.jsonl
               └─ dashboard  (Docker, port 8501, this VM only) ──► reached through an SSH tunnel
             /srv/tram = separate 25 GB volume, so the data outlives the VM
```

It takes about 30 minutes. The only things you need on your laptop are a browser and SSH, which is built into Windows.

## Cost

cPouta charges Billing Units (BU) from the project's allocation. Prices are from [CSC's billing page](https://docs.csc.fi/cloud/pouta/vm-flavors-and-billing/):

| Resource | Rate | Per month (730 h) |
|---|---|---|
| VM `standard.small`: 2 vCPU, 1.9 GiB RAM, 80 GB disk | 0.52 BU/h | ≈ 380 BU |
| Floating (public) IP | 0.2 BU/h | ≈ 146 BU |
| 25 GB standard volume | 3.6 BU per TiB-hour | ≈ 64 BU |
| **Total** | | **≈ 590 BU/month** |

Check the project's remaining BUs at [my.csc.fi](https://my.csc.fi) (project 2022568). A VM keeps using BUs when it is shut off. Only deleting it stops the charge. Volumes and floating IPs are charged while they exist, whether or not they're attached.

The collector stores about 12 MB per hour while trams are running, which is about 7 GB per month. So 25 GB lasts about three months. Compressing finished days (roadmap item E2) will stretch that a lot, and the volume can be enlarged later.

## Two ways to create the resources

- **Web interface:** steps 2–5 below.
- **Command line:** [provision.sh](provision.sh) does steps 2–5 with the OpenStack CLI, and is safe to re-run. It needs an *application credential*: cPouta requires multi-factor login, so the CLI uses an application credential instead of your password.
  1. At <https://pouta.csc.fi>, open **Identity → Application Credentials → Create Application Credential**.
  2. Name it `tram-cli` and set an expiry date (a few months out). Leave *Secret* empty so one is generated. Leave *Unrestricted* unticked.
  3. Click **Download clouds.yaml**. The secret is shown only once.
  4. Save the file as `~/.config/openstack/clouds.yaml`. It contains a secret, so keep it outside the repository.
  5. Run the script (in Git Bash on Windows):

  ```bash
  pip install python-openstackclient                 # once, preferably in its own virtualenv
  export OS_CLOUD=openstack                          # the cloud name inside clouds.yaml
  bash deploy/cpouta/provision.sh <your-ip>/32       # prints the SSH config to add; then go to step 6
  ```

## 1. Create an SSH key (laptop, PowerShell)

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\cpouta
Get-Content $HOME\.ssh\cpouta.pub      # copy this single line
```

## 2. Log in and import the key

1. Open <https://pouta.csc.fi> and log in with your CSC account (multi-factor authentication is required). Select project **2022568**.
2. **Compute → Key Pairs → Import Public Key.** Name it `cpouta` and paste the line from step 1.
   The key is added only when a VM is created. It can't be added to an existing VM later.

## 3. Allow SSH from your own IP address only

1. Find your public IP address, for example by opening <https://ifconfig.me>.
2. **Network → Security Groups → Create Security Group**, named `tram-ssh`.
3. **Manage Rules → Add Rule.** Rule: **SSH**. CIDR: `<your-ip>/32`.

Add one rule for each network you work from, such as home and university. If SSH later hangs with "Connection timed out", your IP has probably changed: add a rule for the new one. Don't use `0.0.0.0/0`. CSC calls opening a port to everyone a bad practice. The VM doesn't need any other open port, because the dashboard is reached through SSH.

## 4. Launch the VM

**Compute → Instances → Launch Instance:**

| Step | Setting |
|---|---|
| Details | Instance name `tram-collector` |
| Source | Boot source **Image**, choose **Ubuntu-24.04**. Set *Create New Volume* to **No** if it is offered: the flavor's 80 GB disk is enough for the system. |
| Flavor | `standard.small` |
| Networks | The project's default network |
| Security Groups | `default` and `tram-ssh` |
| Key Pair | `cpouta` |
| Configuration | Leave empty. Setup runs over SSH in step 7. |

Then use the instance's **Actions → Associate Floating IP**. If no IP is listed, use **+** to allocate one. Note the IP address.

## 5. Create and attach the data volume

1. **Volumes → Create Volume.** Name `tram-data`, size **25 GiB**, default volume type.
2. Use the arrow next to *Edit Volume* and choose **Manage Attachments → Attach to Instance → `tram-collector`**.

## 6. Connect

Add this to `$HOME\.ssh\config` on your laptop. Create the file if it doesn't exist.

```
Host tram
    HostName <floating-ip>
    User ubuntu
    IdentityFile ~/.ssh/cpouta
    LocalForward 8501 localhost:8501
```

Then run `ssh tram`. The `LocalForward` line also forwards the dashboard to your laptop while you're connected.

## 7. Install and start (on the VM)

```bash
lsblk                       # find the 25G disk with no mount point; usually vdb
git clone https://github.com/girimohan/helsinki-tram-mlops.git
cd helsinki-tram-mlops
sudo bash deploy/cpouta/setup.sh /dev/vdb
```

The [script](setup.sh) does the following:
1. Installs Docker.
2. Formats the volume, but only if it is completely blank.
3. Mounts the volume at `/srv/tram`, and remounts it automatically after a reboot.
4. Writes `.env` so Compose stores data on the volume.
5. Builds the image and starts `ingest` and `dashboard`.

The script is safe to run again.

## 8. Check that it works

```bash
docker compose logs --tail 5 ingest     # "N records written", rising every 10 s
ls -lh /srv/tram/data/raw               # today's file growing
```

While `ssh tram` is connected, open <http://localhost:8501> on your laptop to see the Live view.

Log out and back in once so that `docker` works without `sudo`.

## Day-to-day operations

| Task | Command (on the VM, in `~/helsinki-tram-mlops`) |
|---|---|
| Deploy new code | `git pull && sudo bash deploy/cpouta/setup.sh` |
| Collector status | `docker compose ps` and `docker compose logs --tail 20 ingest` |
| Disk space | `df -h /srv/tram` |
| Copy data to the laptop | On the laptop: `scp "tram:/srv/tram/data/raw/tram_2026-11-*.jsonl" data/raw/` |
| Process and train on the VM | `docker compose run --rm pipeline`. Fine for a few days of data. Use a larger flavor, or wait for E2, before training on weeks of data. |
| Stop collecting | `docker compose stop ingest` |
| Apply OS updates | Ubuntu installs security updates automatically. Run `sudo reboot` now and then for kernel updates; the services start again by themselves. |

**Moving to a bigger or new VM.** The data is on the volume, not on the VM:
1. Run `docker compose stop` and then `sudo umount /srv/tram`.
2. Detach the volume in the web interface.
3. Attach it to the new VM.
4. Run `setup.sh` there with the volume's device name. The script finds the existing filesystem and reuses it without formatting.

Alternatively, **Resize Instance** changes the flavor of the existing VM in place.

**Growing the volume.**
1. Stop the services and unmount the volume, as above.
2. Detach the volume.
3. In the volume's menu, choose **Extend Volume**.
4. Re-attach it.
5. Run `setup.sh`, which mounts it.
6. Run `sudo resize2fs /dev/vdb`. ext4 can grow while it is mounted.

**Backups.** Not automated yet. The volume is the only copy of the data. Until phase 2 adds nightly copies to CSC Allas, copy finished days to your laptop now and then (see the table above).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ssh` hangs, then "Connection timed out" | Your IP isn't in `tram-ssh`. Add a rule (step 3). |
| `Permission denied (publickey)` | Wrong key or user. The user must be `ubuntu` and the key `~/.ssh/cpouta`. |
| `permission denied ... docker.sock` | Log out and back in after the first setup. |
| `setup.sh` says nothing is mounted | Pass the device: `sudo bash deploy/cpouta/setup.sh /dev/vdb` |
| Dashboard says "No tram data" | Check `docker compose logs ingest`. The VM needs outbound access to `mqtt.hsl.fi:1883`, which cPouta allows by default. |
