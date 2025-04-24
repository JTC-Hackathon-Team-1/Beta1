#!/usr/bin/env bash
set -e

URL="${1:-http://localhost:8000/pipeline}"
PAYLOAD='{"session_id":"test-session","source_language":"en","target_language":"es","text":"Hello, world!"}'

echo "▶️  Testing POST $URL"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")
echo "HTTP Status: $STATUS"

echo "Response body:" \
&& curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$PAYLOAD" | jq .