#!/bin/sh
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wait out the pinned-L2 batch, then run the warming test. Unattended.
cd $ROOT/bench || exit 1

echo "### waiting for the pinned-L2 batch"
i=0
while [ $i -lt 120 ]; do
    if grep -q '"Q_l2pin_16"' results.jsonl 2>/dev/null; then break; fi
    sleep 30
    i=$((i+1))
done

echo "### arena status"
grep -E "L2 host cache" logs/srv_Q_l2pin_16.log 2>/dev/null || echo "(no arena line)"

echo "### pinned-L2 results"
python -c "
import json
for l in open('results.jsonl'):
    d=json.loads(l)
    if d['name'].startswith('Q_'): print('%-14s %6.2f  %s' % (d['name'], d['median'], ' '.join('%.2f'%v for v in d['vals'])))
"

echo "### matched states: control vs pinned L2"
python cmp.py logs/srv_Q_ctl.log logs/srv_Q_l2pin_16.log 2>&1 | tail -3

# make sure nothing is left holding the GPU before the next batch
taskkill /F /IM llama-server.exe >/dev/null 2>&1
sleep 5

echo ""
echo "### warming test"
python warm_test.py 2>&1 | tail -30

echo ""
echo "### quality gate on everything new"
python qcheck.py logs/out_Q_ctl.txt logs/out_Q_l2pin_16.txt logs/out_R_warm_build.txt logs/out_R_warm_use.txt 2>&1 | tail -10
