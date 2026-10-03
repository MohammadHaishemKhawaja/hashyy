#!/bin/bash
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D="$ROOT/models/reap128"; mkdir -p "$D"; cd "$D" || exit 1
F="Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-UD-IQ4_XS.gguf"
U="https://huggingface.co/talhasarac/Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-GGUF/resolve/main/$F"
EXP=48853994112
for try in $(seq 1 10); do
  cur=$(stat -c%s "$F" 2>/dev/null || echo 0)
  if [ "$cur" -eq "$EXP" ]; then echo "[$(date +%H:%M:%S)] COMPLETE $F"; break; fi
  echo "[$(date +%H:%M:%S)] try $try: have $((cur/1073741824)) of 45 GiB"
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -sS -o "$F" "$U" < /dev/null
done
cur=$(stat -c%s "$F" 2>/dev/null || echo 0)
echo "final: $cur / $EXP  $([ "$cur" -eq "$EXP" ] && echo OK || echo INCOMPLETE)"
df -h /c | tail -1
