#!/bin/bash
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
D="$ROOT/models/reap256"; mkdir -p "$D"; cd "$D" || exit 1
B="https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF/resolve/main"
declare -A EXP=(
 ["Qwen3.8-Flash-Next-UD-Q3_K_XL-reap256-00001-of-00002.gguf"]=48170000000
 ["Qwen3.8-Flash-Next-UD-Q3_K_XL-reap256-00002-of-00002.gguf"]=13780000000
)
for f in "${!EXP[@]}"; do
  for try in 1 2 3 4 5 6 7 8; do
    cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
    # stop when within 2% of expected (API sizes are approximate)
    if [ "$cur" -gt 0 ]; then
      lo=$(( ${EXP[$f]} * 97 / 100 ))
      [ "$cur" -ge "$lo" ] && { echo "[$(date +%H:%M:%S)] OK $f ($((cur/1073741824)) GiB)"; break; }
    fi
    echo "[$(date +%H:%M:%S)] try $try $f (have $((cur/1073741824)) GiB)"
    curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -sS -o "$f" "$B/$f" < /dev/null
  done
done
echo "[$(date +%H:%M:%S)] DONE"; ls -l --block-size=M; df -h /c | tail -1
