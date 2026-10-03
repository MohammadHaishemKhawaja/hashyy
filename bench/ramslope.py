#!/usr/bin/env python3
"""tok/s as a function of how much RAM the OS page cache is allowed to have.

The page cache is the only L2 behind the VRAM expert cache, and it sits at ~15-16 GiB of
this box's 32 GB while ~8 GB is held by the GPU driver's WDDM backing store and the rest by
other processes. This holds N GiB of touched private memory and re-measures, so the slope
says what reclaiming that memory would actually be worth before anyone changes a BIOS or
driver setting for it.
"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep

HOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog.py")

if __name__ == "__main__":
    gibs = [float(x) for x in (sys.argv[1:] or ["0","2","4"])]
    args = "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 10 --poll 0".split()
    sweep.say("=== ram slope %s ===" % time.strftime("%H:%M:%S"))
    for g in gibs:
        h = None
        if g > 0:
            h = subprocess.Popen([sys.executable, "-u", HOG, str(g), "1200"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(40)
        try:
            sweep.run("ram_%gG" % g, args, {}, 3, 3, 128, False)
        except Exception as e:
            sweep.say("ram_%gG FAILED %r" % (g, e))
        finally:
            if h is not None:
                h.kill(); time.sleep(25)
    sweep.say("=== done ===")
