#!/bin/bash
# Probe the restarted vLLM server. Written as a file and scp'd rather than passed
# as an argument, because nested quoting through PowerShell -> ssh -> bash
# silently mangles commands.

JOB=$(qstat -u n2602358e 2>/dev/null | awk '/vllm-qwen/ && $10=="R" {print $1}' | tail -1)
echo "running job = $JOB"
qstat -u n2602358e 2>/dev/null | tail -3

NODE=$(qstat -f "$JOB" 2>/dev/null | grep exec_host | sed 's/.*= //; s|/.*||')
echo "node        = $NODE"
BASE="http://$NODE:8000"

echo
echo "=== health ==="
curl -s --max-time 15 "$BASE/v1/models" | head -c 180
echo

echo
echo "=== tool-calling test (the decisive one) ==="
curl -s --max-time 900 "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/home/n2602358e/payload_tools.json
echo

echo
echo "=== log tail ==="
tail -4 /home/n2602358e/logs/vllm-qwen35.out
