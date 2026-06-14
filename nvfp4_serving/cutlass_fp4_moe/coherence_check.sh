#!/usr/bin/env bash
# Coherence proof for the live MiniMax-M3-NVFP4 endpoint: capital of Poland
# (-> Warsaw), basic arithmetic, and a short code-gen.  Greedy (temperature 0).
set -uo pipefail
HOST="${1:-http://localhost:8000}"
MODEL="${2:-/models/MiniMax-M3-NVFP4}"

ask() {
  local prompt="$1" maxtok="${2:-64}"
  curl -s "${HOST}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$prompt" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}],\"temperature\":0,\"max_tokens\":${maxtok}}" \
    | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    msg=d["choices"][0]["message"]
    txt=msg.get("content") or ""
    rc=msg.get("reasoning_content") or ""
    out=(rc+"\n"+txt).strip() if rc else txt
    print(out.strip()[:600])
except Exception as e:
    print("PARSE-ERROR:",e, file=sys.stderr); print(sys.stdin.read() if not sys.stdin.closed else "")'
}

echo "=================================================================="
echo "Q1: capital of Poland"
echo "------------------------------------------------------------------"
ask "What is the capital of Poland? Answer in one word." 32
echo
echo "=================================================================="
echo "Q2: arithmetic"
echo "------------------------------------------------------------------"
ask "Compute 17 * 23. Give only the number." 32
echo
echo "=================================================================="
echo "Q3: short code gen"
echo "------------------------------------------------------------------"
ask "Write a Python function that returns the factorial of n. Only the code." 160
echo
echo "=================================================================="
