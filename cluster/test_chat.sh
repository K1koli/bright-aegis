#!/bin/bash
# Probe the vLLM server directly from the login node (bypasses the SSH tunnel):
# a failure here is the model's fault; success means the tunnel is the suspect.
# Payloads are separate files so they can be JSON-validated before being sent.
# Run with: ssh gaas 'bash ~/test_chat.sh'
URL=http://hpc-gaas-g02:8000/v1/chat/completions
DIR=/home/n2602358e

echo "=== 0. server health ==="
curl -s --max-time 20 http://hpc-gaas-g02:8000/v1/models | head -c 200
echo

echo "=== 1. plain chat, max_tokens=32 ==="
time curl -s --max-time 900 "$URL" \
  -H "Content-Type: application/json" \
  --data-binary "@$DIR/payload_chat.json"
echo

echo "=== 2. tool calling, max_tokens=256 ==="
time curl -s --max-time 900 "$URL" \
  -H "Content-Type: application/json" \
  --data-binary "@$DIR/payload_tools.json"
echo
