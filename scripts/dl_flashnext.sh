#!/bin/bash
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -u
D="$ROOT/models/flashnext"
mkdir -p "$D"; cd "$D"
B="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main"
F=(
 "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
 "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf"
 "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf"
 "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
)
for f in "${F[@]}"; do
  out="$(basename "$f")"
  echo "[$(date +%H:%M:%S)] -> $out"
  curl -L -C - --retry 8 --retry-delay 5 -sS -o "$out" "$B/$f" || echo "FAILED $out"
  ls -l --block-size=M "$out" 2>/dev/null | awk '{print "    got",$5}'
done
echo "[$(date +%H:%M:%S)] ALL DONE"; df -h /c | tail -1
