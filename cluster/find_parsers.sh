#!/bin/bash
# Verify the exact CLI flag spellings before paying a 7-minute job restart.
VL=/usr/local/software/vllm/vllm

echo "=== chat-template-kwargs flag spellings ==="
grep -rn '"--chat-template-kwargs"\|"--default-chat-template-kwargs"\|chat_template_kwargs' "$VL/engine/arg_utils.py" 2>/dev/null | head -12

echo
echo "=== reasoning-parser flag + registered reasoning parser names ==="
grep -rn '"--reasoning-parser"' "$VL/engine/arg_utils.py" 2>/dev/null | head -5
grep -n -i 'qwen3' "$VL/reasoning/__init__.py" 2>/dev/null | head -12

echo
echo "=== reasoning parser registry block ==="
grep -n '"qwen' "$VL/reasoning/__init__.py" 2>/dev/null | head -12

echo
echo "=== does the model dir advertise a reasoning parser in config? ==="
grep -o '"reasoning_parser"[^,]*' /home/n2602358e/models/Qwen3.5-9B/config.json 2>/dev/null
grep -o '"architectures"[^]]*]' /home/n2602358e/models/Qwen3.5-9B/config.json 2>/dev/null

echo
echo "=== tokenizer_config: chat template kwargs support ==="
grep -o 'enable_thinking[^,}]*' /home/n2602358e/models/Qwen3.5-9B/tokenizer_config.json 2>/dev/null | head -3
