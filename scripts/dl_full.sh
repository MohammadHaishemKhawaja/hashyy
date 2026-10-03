#!/bin/bash
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D="$ROOT/models/flashnext"; mkdir -p "$D"; cd "$D" || exit 1
B="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ3_XXS"
declare -A EXP=(
 ["Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"]=10946624
 ["Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf"]=49567921344
 ["Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf"]=32382955968
)
for f in "${!EXP[@]}"; do
  for try in $(seq 1 10); do
    cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "$cur" -eq "${EXP[$f]}" ] && { echo "[$(date +%H:%M:%S)] OK $f"; break; }
    echo "[$(date +%H:%M:%S)] try $try $f: $((cur/1073741824))/$(( ${EXP[$f]}/1073741824 )) GiB"
    curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -sS -o "$f" "$B/$f" < /dev/null
  done
done
echo "[$(date +%H:%M:%S)] ALL DONE"; ls -l --block-size=M; df -h /c | tail -1
