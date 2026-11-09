#!/usr/bin/env bash
# Pull-based deploy: if main has a new commit and every CI check on it passed,
# fast-forward this clone and rebuild the services. tram-autodeploy.timer runs it
# every 5 minutes. Nothing connects in to the server; it only reads from GitHub.
set -euo pipefail

BRANCH="${BRANCH:-main}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

git fetch -q origin "$BRANCH"
current="$(git rev-parse HEAD)"
target="$(git rev-parse "origin/$BRANCH")"
[[ "$current" != "$target" ]] || exit 0

# owner/name from the origin URL, e.g. https://github.com/girimohan/helsinki-tram-mlops.git
repo="$(git remote get-url origin | sed -E 's#(\.git)?$##; s#^.*github\.com[:/]##')"
ci="$(curl -fsS --max-time 20 -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$repo/commits/$target/check-runs" | python3 -c '
import json, sys
runs = json.load(sys.stdin)["check_runs"]
if not runs or any(r["status"] != "completed" for r in runs):
    print("pending")
elif all(r["conclusion"] in ("success", "skipped", "neutral") for r in runs):
    print("success")
else:
    print("failure")')"

case "$ci" in
    pending) echo "CI has not finished for ${target:0:7}; will check again"; exit 0 ;;
    failure) echo "CI failed for ${target:0:7}; staying on ${current:0:7}"; exit 0 ;;
esac

git merge -q --ff-only "origin/$BRANCH"
echo "Deploying ${target:0:7}: $(git log -1 --format=%s)"
docker compose up -d --build --remove-orphans
echo "Deployed ${target:0:7}"
