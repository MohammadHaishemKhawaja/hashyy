#!/bin/bash
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D="$ROOT/models/flashnext"; mkdir -p "$D"; cd "$D" || exit 1
B="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main"
F=("UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
   "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf"
   "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf"
   "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
for f in "${F[@]}"; do
  out=$(basename "$f")
  echo "[$(date +%H:%M:%S)] START $out"
  curl -L -C - --retry 8 --retry-delay 5 -sS -o "$out" "$B/$f" < /dev/null
  echo "[$(date +%H:%M:%S)] done $out -> $(stat -c%s "$out" 2>/dev/null) bytes"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
ls -l --block-size=M; df -h /c | tail -1
