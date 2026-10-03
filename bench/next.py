import os, sys
p='matrix.py'
s=open(p,encoding='utf8').read()
i=s.index("CONFIGS = ["); j=s.index("]\n\nif len(sys.argv)")+1
new='''BEST = "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0"
CONFIGS = [
    ("L_ctl",      BEST, ""),
    ("L_l2_16",    BEST, "LLAMA_MOE_STREAM_L2_GB=16"),
    ("L_l2_19",    BEST, "LLAMA_MOE_STREAM_L2_GB=19"),
    ("L_ctl2",     BEST, ""),
]'''
s=s[:i]+new+s[j:]
open(p,'w',encoding='utf8',newline='\n').write(s)
print(new)
