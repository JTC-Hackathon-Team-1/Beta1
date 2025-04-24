#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8000"

echo "1️⃣  Health check…"
curl -s "$BASE/" | jq .
echo

echo "2️⃣  OpenAPI title…"
curl -s "$BASE/openapi.json" | jq -r '.info.title'
echo

echo "3️⃣  Missing‐field validation (expect 422)…"
CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"source_language":"en","target_language":"es","text":"Hello"}')
if [[ "$CODE" != "422" ]]; then
  echo "❌ Expected 422, got $CODE" >&2
  exit 1
else
  echo "✅ Got 422 for missing session_id"
fi
echo

echo "4️⃣  Valid translation request…"
RESP=$(curl -s -X POST "$BASE/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"test-session",
    "source_language":"en",
    "target_language":"es",
    "text":"Hello, how are you today?"
  }')
echo "Response:"
echo "$RESP" | jq .
echo

TRANSLATION=$(echo "$RESP" | jq -r .translation)
if [[ "$TRANSLATION" == "null" || -z "$TRANSLATION" ]]; then
  echo "❌ No translation in response" >&2
  exit 1
else
  echo "✅ Translation: $TRANSLATION"
fi

ENT_COUNT=$(echo "$RESP" | jq '.entities | length')
if [[ "$ENT_COUNT" -gt 0 ]]; then
  echo "✅ Found $ENT_COUNT entities"
else
  echo "⚠️  No entities extracted"
fi
echo

echo "🎉 All tests passed!"