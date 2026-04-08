#!/usr/bin/env bash
set -uo pipefail
PING_URL="${1:-}"
REPO_DIR="${2:-.}"
if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi
PING_URL="${PING_URL%/}"
PASS=0
pass() { echo "PASSED -- $1"; PASS=$((PASS + 1)); }
fail() { echo "FAILED -- $1"; }

echo "Step 1/3: Pinging HF Space..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30)
[ "$HTTP_CODE" = "200" ] && pass "HF Space live" || { fail "HTTP $HTTP_CODE"; exit 1; }

echo "Step 2/3: Docker build..."
docker build "$REPO_DIR" && pass "Docker build succeeded" || { fail "Docker build failed"; exit 1; }

echo "Step 3/3: openenv validate..."
cd "$REPO_DIR" && openenv validate && pass "openenv validate passed" || { fail "openenv validate failed"; exit 1; }

echo "All $PASS/3 checks passed!"
