# Local inference results — measured 2026-10-01
Hardware: RTX 5070 12GB (PCIe gen.max 3) + RTX 3060 12GB (x4), 32GB DDR4-2400,
Kingston SNV2S1000G NVMe, Windows 10 LTSC. llama.cpp b11320 CUDA 13.4.

## Headline
Single 5070, Qwen3.6-35B-A3B (MoE, 256 experts, 8 active) Q4_K_M + MTP: **44.76 tok/s**
Target was 15 tok/s. Dense Qwen3.8-27B on the same single card: 7.53 tok/s.

## MoE: residency vs speed (single 5070, --spec-type draft-mtp)
| -ncmoe | tok/s | VRAM  |
|--------|-------|-------|
| 26     | 44.76 | 10.9G |
| 28     | 41.73 | 10.0G |
| 30     | 39.83 |  9.1G |
| 34     | 37.45 |  7.3G |
| 38     | 34.83 |  5.5G |
| 41     | 31.80 |  4.2G |
MTP speedup consistently 1.22-1.25x, draft acceptance 68-72%.

## Dense Qwen3.8-27B (for reference)
- Both GPUs, -sm tensor -ts 70/30: 31.92 tok/s
- Single 5070, -ngl 46: 4.88-5.21
- Single 5070 + MTP n-max=3: 7.53 (74% accept)  <- best single-card dense

## Negative results (don't repeat these)
- FFN-only offload via -ot on dense: 1.73-3.10 (worse than layer offload).
  PCIe activation round-trips cost more than the bandwidth saved.
  Colibri reached the same conclusion independently.
- ngram speculative decoding: 16% acceptance alone; stacking with MTP
  DROPS throughput 7.53 -> 6.00/6.51. Use MTP alone.
- Draft depth > 3 wastes compute: llama.cpp drafts linear chains, not trees.
  Accepted tokens plateau (~197) while drafts explode (236 -> 564).
- VRAM cliff is razor sharp. ncmoe 20 -> 22 = 8.93 -> 43.85 tok/s.
  Over the line the driver silently spills to RAM instead of erroring.

## Not needed at this model size
21.6GB model fits GPU+RAM entirely. SSD streaming only becomes necessary
above ~45GB total (RAM 32 + VRAM 12). That is the next frontier, not this one.

## UPDATE — IQ3_XXS (3-bit) results, single 5070
Smaller quant shrinks experts, so far more fit in VRAM: the cliff moved 22 -> 7.

llama-bench (-p 512 -n 128 -r 2 -ub 256):
| -ncmoe | pp512 | tg128 |
|--------|-------|-------|
| 6      | 685.5 | 80.74 |
| 7      | 780.6 | 82.54 |  <- peak
| 8      | 708.6 | 74.32 |
| 12     | 553.8 | 62.84 |
| 20     | 397.5 | 46.69 |

llama-server, real-world, no speculation:
| -ncmoe | tok/s | VRAM  |
|--------|-------|-------|
| 7      | 72.72 | 11.5G |  <- fastest, tight
| 8      | 69.04 | 11.3G |
| 10     | 62.59 | 10.8G |  <- recommended daily
| 12     | 57.41 | 10.2G |

### MTP crossover (important)
MTP helps only when MEMORY-bound. Once compute-bound it is a net loss:
| config                | no spec | +MTP  | effect |
|-----------------------|---------|-------|--------|
| Dense 27B, single GPU |  5.21   |  7.53 | 1.45x  |
| MoE Q4_K_M            | 35.72   | 44.76 | 1.25x  |
| MoE IQ3_XXS           | 66.07   | 64.19 | 0.97x  |  <- turn it OFF

### ubatch / KV quant (Q4_K_M, ncmoe 26)
-ub 512/256/128 -> tg 39.70 / 41.65 / 39.86   (256 best)
KV q8_0 vs q4_0 -> 40.69 / 39.91              (no gain, keep q8_0)
Freeing buffer VRAM did NOT move the cliff - expert weights are the binding
constraint, which is why a smaller quant works and buffer tuning doesn't.

## Hardware finding: RAM is underclocked
DIMMs are Kingston KF3600C18D4 (rated DDR4-3600) running at 2400.
XMP/DOCP is OFF. Enabling it in BIOS is ~+50% memory bandwidth for free,
which is the exact bottleneck for CPU-side experts. Board: ASRock B550M-C,
CPU Ryzen 5 5600GT (PCIe 3.0 only - explains gen.max=3). 2 of 4 DIMM slots free.

## IQ3_XXS cliff, measured (was previously inferred - corrected)
| -ncmoe | pp512 | tg128 |
|--------|-------|-------|
| 2      |  47.2 |  6.81 |
| 3      | 113.1 | 11.07 |
| 4      | 110.9 | 13.09 |
| 5      |  75.8 | 10.20 |  <- cliff
| 6      | 585.3 | 57.44 |  <- unstable: also measured 80.74 on an earlier run
| 7      | 780.6 | 82.54 |  <- stable peak
Lesson: stay 3-4 layers above the cliff, not 1. Settings adjacent to the cliff
vary run-to-run with whatever else holds VRAM.

## Fine-grained -ot tensor placement: tested, does NOT help
Hypothesis: --n-cpu-moe moves whole layers (coarse, ~300MB steps), so pulling
individual expert tensors back with -ot should fill leftover VRAM better.

GOTCHA: --n-cpu-moe expands into -ot rules internally and FIRST MATCH WINS.
An -ot placed after -ncmoe silently does nothing (identical VRAM + speed).
-ot must come BEFORE -ncmoe on the command line.

IQ3_XXS, single 5070, with correct ordering:
| config                             | tok/s | VRAM   |
|------------------------------------|-------|--------|
| -ot all exps=CPU (control)         | 26.15 | 2.78 G |
| ncmoe=26 whole                     | 35.83 | 6.63 G |
| ncmoe=26 + down_exps 22-25 -> GPU  | 36.49 | 7.06 G |
| ncmoe=24 whole                     | 37.45 | 7.17 G |  <- beats partial at = VRAM
| ncmoe=26 + gate/up 24-25 -> GPU    | 35.94 | 6.95 G |
| ncmoe=26 + ALL exps 22-25 -> GPU   | 39.28 | 7.70 G |
| ncmoe=22 whole                     | 40.10 | 7.70 G |  <- beats partial at = VRAM

CONCLUSION: whole-layer offload wins every equal-VRAM comparison. Partial-layer
placement forces extra CPU<->GPU activation round-trips that cost more than the
finer granularity gains. Same root cause as the dense FFN-only result. Just tune
ncmoe; the -ot trick adds nothing for this model.

## 8GB-card targets (IQ3_XXS, measured)
| card  | setting            | tok/s | VRAM   |
|-------|--------------------|-------|--------|
| 12 GB | ncmoe=10           | 62.59 | 10.8 G |
| 10 GB | ncmoe=18           | 46.01 |  8.7 G |
|  8 GB | ncmoe=22           | 40.10 |  7.7 G |
|  8 GB | ncmoe=26 (safe)    | 35.83 |  6.6 G |
|  6 GB | ncmoe=30           | 33.06 |  5.5 G |
|  4 GB | -ot all exps=CPU   | 26.15 |  2.8 G |

## Rejected without building: ik_llama.cpp
Documented regression on Qwen3 MoE vs mainline (~2x slower PP, ~1.5x slower TG).
Not worth the build time for this model family.

## Qwen3.8-Flash-Next (~177B MoE, UD-IQ3_XXS, 82GB) on 12GB VRAM + 32GB RAM
IT RUNS. 85GB of weights on a 12GB card via mmap from NVMe.
Config: --fit on --fit-target 1024 --lazy-mode on --load-mode mmap -fa on
        -ctk q8_0 -ctv q8_0 -ub 256 -c 8192 --jinja

| run            | tok/s | note                        |
|----------------|-------|-----------------------------|
| cold gen 1     | 1.48  | empty page cache            |
| warming gen 2-6| 2.6-3.9 | page cache filling        |
| warm baseline  | 3.75  | steady state                |
| ngram-mod n12  | 3.50  | speculation HURTS           |
| ngram-cache    | 3.68  | speculation HURTS           |
Load time 23s (mmap is lazy). VRAM 9.9GB. Output quality good.

### Why speculation fails here (unlike memory-bound cases)
We are DISK-bound with a saturated page cache (0.3 GB free RAM during runs),
not merely memory-bound. Colibri documents the mechanism: on a cold cache,
verified draft tokens route to ADDITIONAL experts (~660 -> ~1100 expert-loads
per token). Speculation therefore increases disk traffic in exactly this regime.

### MTP does not work on this build
b11320 cannot load the Flash-Next MTP head:
  shared variant -> "tensor 'token_embd.weight' not found"
      (the shared head reuses the main model's embeddings by design,
       so it cannot load standalone as a -md draft model)
  non-shared     -> "tensor 'output_hc_norm.weight' not found"
Flash-Next MTP support postdates b11320. Revisit on a newer build.

### The binding constraint, demonstrated
Community reports on this exact model+quant:
  RTX 5080 16GB + 64GB DDR5  -> 27.5-29 tok/s (76GB via mmap, also > RAM)
  5070Ti+5060Ti + 64GB RAM   -> 23.3 tok/s
  2x3080 20GB + 64GB DDR4    -> 16-17 tok/s
  3x3060 36GB + 96GB RAM     -> 14 tok/s
  THIS BOX, 12GB + 32GB@2400 -> 3.75 tok/s
Every faster setup has 64GB+ RAM. RAM quantity is the constraint, not the GPU.

## Flash-Next via REAP expert pruning - FINAL
Problem: 76.3 GiB model, only ~36 GiB usable memory. Breakdown by tensor:
  routed experts 45.29 GiB | ngram/PLE table 26.85 GiB | attn/ssm 3.0 | emb 1.0
The ngram/PLE table is 35% of the file and --lazy-mode does NOT keep it out of RAM.

REAP (arXiv 2510.13999) deletes low-saliency experts permanently.
| variant   | experts GiB | body (resident) | result                      |
|-----------|-------------|-----------------|-----------------------------|
| original  | 45.29       | 49.5 GiB        | 3.75 tok/s                  |
| REAP-256  | 25.99       | 30.8 GiB        | 7.5 cold -> 2.2, watchdog killed |
| REAP-128  | 13.86       | 18.6 GiB        | 14.41 tok/s STABLE          |

REAP128 tuning (all within noise - we are RAM-BANDWIDTH bound, not placement bound):
  fit384 ub128 14.12 | fit384 ub256 14.30 | ncmoe28 14.08
  ncmoe32 14.26 | ncmoe36 14.36 | ncmoe36 t6 14.41 <- best
  ngram-mod 14.31 | ngram-cache 14.32 (speculation does not help)
Warming: 9.06 -> 13.49 -> 14.0 -> stable 14.1-14.4 over 10 gens. RAM free 6-11 GB.

### QUALITY: code good, world knowledge damaged
Both REAP variants (independently produced) fail identically, so this is the
pruning calibration, not a bug and not pruning depth:
  "capital of Australia" -> "Australia"  (WRONG at temp 0 AND 0.7)
  syllogism -> wrong at temp 0, right at temp 0.7
  sheep trap -> correct;  2847/3 -> 949 correct;  code -> correct
Both were calibrated on code/agent corpora, so general knowledge experts were
pruned away. USE FOR CODING ONLY. Use Qwen3.6-35B-A3B for general questions.

### The remaining 0.6 tok/s
Every knob converges to ~14.4 => bound by DDR4-2400 memory bandwidth.
RAM is rated DDR4-3600 (Kingston KF3600C18D4) running at 2400; DOCP is OFF.
Enabling it is ~+50% bandwidth on exactly the bottleneck -> should give ~19-21 tok/s.

## Expert streaming (PR #25294) — BUILT AND TESTED
Built a binary that did not previously exist: current master (qwen4exp/Flash-Next
support) + the MoE expert-streaming PR merged on top. 18 merge conflicts across
1467 commits of divergence, plus 3 post-merge compile fixes
(n_expert_used became a method; ctor init-list; an unbalanced brace).
Toolchain: CUDA 13.3 assembled from redist zips (nvcc needs libnvvm's cicc,
which cuda_nvcc does not ship), portable CMake + Ninja, MSVC 2022. No installer.

Result on the FULL UNPRUNED 177B (12GB VRAM + 32GB RAM):
| config                   | tok/s | note                          |
|--------------------------|-------|-------------------------------|
| plain mmap (stock)       | 3.75  | best overall                  |
| stream, vram cache 8G    | 3.28  | direct and buffered identical |
| stream, vram cache 12G   | 2.69  |                               |
| stream, cpu cache 20G    | 2.35  | CPU expert compute costs more |
| stream, cache 24G+       | FAIL  | "unable to allocate CUDA0 buffer" |

KEY: the expert cache is allocated in VRAM, so it is capped by the 12GB card,
NOT by system RAM. That caps hit rate and is why streaming loses to mmap here.

### Why 15 tok/s is unreachable on this box, with numbers
Measured SSD at expert granularity: 1.14 GB/s random (1.9MB reads), 1.66 seq.
Per token the model must read 0.91 GB of routed experts.
  time/token = 0.91 * (1 - hit_rate) / 1.14
  15 tok/s  => time <= 0.0667 s  =>  hit_rate >= 91.7%
91.7% hit needs roughly 25 GiB of expert cache. The VRAM cache tops out near
8 GiB (~18% of the 45.29 GiB of experts), which yields ~55-65% hit => ~3.3 tok/s.
Measured 3.28. The model matches reality.

WHAT ACTUALLY CLOSES THE GAP: 64 GB RAM. The full working set is 49.5 GiB; with
64 GB it is resident and no streaming is needed. Community reports on this exact
model+quant: 16-29 tok/s, all on 64 GB+ machines.

## TWO-TIER CACHE: 3.75 -> 6.56 tok/s on the FULL UNPRUNED 177B
The PR force-disables mmap when streaming is on. That threw away ~17GB of page
cache. Patched it to opt-out (LLAMA_MOE_STREAM_KEEP_MMAP=1) so we get
  L1 = expert cache in VRAM (hard-capped by the 12GB card)
  L2 = OS page cache in RAM (free, ~17GB)
| cache | tok/s |          | config                  | tok/s |
|-------|-------|          |-------------------------|-------|
| 3G    | 5.23  |          | plain mmap (was best)   | 3.75  |
| 4G    | 5.67  |          | stream only, no L2      | 3.28  |
| 5G    | 6.11  |          | stream + L2 (6G)        | 6.56  |
| 6G    | 6.56  | <- peak  |                         |       |
| 7G    | 3.67  | VRAM spill cliff

Tried and gave nothing: tight ctx/ubatch to free VRAM (3.2-5.1), ngram-mod /
ngram-cache / ngram-map-k speculation (6.30 -> 6.34, within noise),
cache in host RAM via -ncmoe 48 (2.35, CPU expert compute costs more).

### MTP is unavailable for this model, not a config error
Both the unsloth and the official ggml-org MTP heads fail to load, on BOTH the
stock b11327 binary and the merged build:
  check_tensor_dims: tensor 'output_hc_norm.weight' not found
Our main GGUF contains no nextn/MTP block, so the separate head cannot form a
complete draft model. An MTP-bundled Flash-Next GGUF is ~80 GiB; only 17 GB free.

## THE PHYSICS, fully measured (why 15 tok/s is out of reach here)
Per token the model must touch 0.91 GB of routed experts. Measured hardware:
  PCIe  : Gen3 x16 (gen.max=3, APU limit)  -> ~12 GB/s usable
  SSD   : 1.14 GB/s random at 1.9MB (expert granularity), 1.66 sequential
  RAM   : DDR4-2400 dual channel          -> ~25 GB/s
  VRAM  : 12 GB, of which ~6 GiB is free for the expert cache
  cache : L1 6 GiB VRAM + L2 23.27 GB page cache (measured standby list)
          = ~29 GiB of 45.29 GiB experts = 64% capacity

| path                                   | ceiling        |
|----------------------------------------|----------------|
| experts in RAM, computed on GPU        | 13.2 tok/s  <- PCIe Gen3 wall |
| experts on CPU, 90% hit                |  7.0 tok/s     |
| experts on CPU, 95% hit                |  9.5 tok/s     |
| experts on CPU, 100% hit (all resident)| 14.8 tok/s     |
Measured today: 6.56 tok/s, implied hit rate ~81%.

15 tok/s needs ~100% of experts resident. 45.29 GiB of experts + 4.2 GiB of
non-expert weights does not fit in 12 GB VRAM + 32 GB RAM. That is the whole story.

### The one lever that would break the PCIe wall: speculative decoding
Verifying K tokens per weight sweep divides bytes-per-token by K (minus the
expert-union growth). At K=3 with ~1.6x union growth, effective bytes drop to
~0.49 GB/token and the PCIe ceiling rises from 13.2 to ~24.7 tok/s.
This is the correct method (SpecMoEOff reports up to 2.5x). It is BLOCKED here:
  - MTP: main GGUF has no nextn block; both unsloth and official ggml-org heads
    fail with "tensor 'output_hc_norm.weight' not found" on stock AND merged builds
  - no small draft model shares this model's 248320-token vocab
  - ngram speculation (mod/cache/map-k): measured 6.30 -> 6.34, no gain

## Heat-ranked host L2 expert cache: BUILT, TESTED, DOES NOT HELP
Added a host-RAM L2 tier to llama-moe-stream.cpp (slab arena, offset-keyed,
decaying-heat eviction), opt-in via LLAMA_MOE_STREAM_L2_GB. Idea: an L1 miss
served from RAM is a ~12 GB/s PCIe copy instead of a ~1.14 GB/s disk read.

| config                  | tok/s | RAM free |
|-------------------------|-------|----------|
| no L2 (OS page cache)   | 5.48  | 16.1 GB  |
| L2 8G  + page cache     | 4.43  |  9.4 GB  |
| L2 14G + page cache     | 4.49  |  5.0 GB  |
| L2 14G, no mmap         | 5.09  |  8.2 GB  |
| L2 18G, no mmap         | 5.07  |  4.7 GB  |

WHY IT LOSES: the OS page cache is already a 23.27 GB L2 (measured standby list).
Allocating an explicit 14-18 GB tier takes that memory away from it, so total
cache capacity goes DOWN. Secondary: eviction is an O(n_slabs) linear scan under
the lock. Beating the OS here would need more RAM, not a smarter policy at
smaller capacity - which is the same wall everything else hit.
Left in the tree, default-off. run-full177b.bat is unaffected.

## Session 2: instrumenting the 177B streaming path

Goal was 15-20 tok/s on the FULL UNPRUNED Qwen3.8-Flash-Next. Starting point 6.56.

### Method fix first

Routing on this model is bit-exact deterministic, so the `(L1 hit, miss/tok, cold)`
triple identifies the same workload point in every run. `bench/cmp.py` compares two
runs only at matched triples. This matters: the OS page cache warms across server
restarts, so raw medians drift up 3-5% per batch and a "gain" measured across
batches is usually just a warmer cache. Two zombie `llama-server` processes also
contaminated one early batch, so `bench/bench.py` now force-kills stragglers before
and after every run.

### The model, measured

48 layers, 512 experts, 10 active. Per expert the three weights are
`ffn_gate_exps` 490 KiB iq2_s + `ffn_up_exps` 490 KiB iq2_s + `ffn_down_exps`
900 KiB iq4_nl = **1.815 MiB**, so one token touches 480 experts = **871 MiB**.
Routed experts are 45.29 GiB; the `per_layer_token_embd` n-gram table is another
26.82 GiB but is already lazy-read and row-prefetched, so it costs nothing.

With `--moe-stream-cache 6` the cache is **67 slots per layer** = 13.1% of 512.

| | |
|---|---|
| L1 hit | 70% (75% at 79 slots) |
| misses | ~144/tok = ~260 MiB |
| stall == remap total | 85-135 ms/tok |
| token time | stall + ~20 ms |
| GPU utilisation | 17-20% |
| standby page cache | 15.5 GiB |

So 15 tok/s needs stall <= 47 ms and 20 tok/s needs stall <= 30 ms.

### Everything that reduces I/O work changes nothing

This is the central result. Three independent interventions each cut I/O
substantially and **none** moved throughput:

| intervention | effect on I/O | throughput |
|---|---|---|
| host-RAM L2 arena, 19 GiB, LFU/CLOCK | disk reads **cut in half**, worker time 573 -> 368 ms summed | 132.3 -> 136.2 ms stall (**3% worse**) |
| per-weight parallel loads (gate/up/down on 3 workers) | read time summed 281 -> 423 | 136.4 -> 136.7 ms stall (**null**) |
| 79 L1 slots instead of 67 | L1 hit 70 -> 75%, misses 144 -> 119/tok | 8.20 -> 8.16 tok/s (**null**) |

Plus: positional `ReadFile` replacing a global-mutex seek+read (null),
`FILE_FLAG_RANDOM_ACCESS` (null), pinned staging (null), and uploading straight
from the mmap with batched `PrefetchVirtualMemory` (**2x worse** - the page faults
then land inside the CUDA copy).

A cost that does not fall when the work falls is a **fixed cost per layer**:
~2 ms x the 40.7 layers that miss every token = ~100 ms, which is the whole stall.

### What the fixed cost is

`--moe-stream-io-threads` sweep: **6 -> 8.10, 9 -> 8.06, 12 -> 7.51, 18 -> 7.67**.
Fewer I/O threads is better, monotonically. That points at CPU oversubscription,
and the cause is in ggml: `cpuparams.poll` defaults to **50**, and `ggml_barrier`
spins on `_mm_pause()`. The remap op is a CPU custom op, so every one of the 48
layers per token enters the CPU threadpool, thread 0 blocks on expert I/O, and the
other `n_threads-1` threads **busy-spin** for the whole wait. On a 6-core CPU with
6 I/O workers also wanting cores, the handoff remap -> worker -> remap costs
milliseconds instead of microseconds.

### Cache policy and prefetch are both dead ends (trace-driven)

`LLAMA_MOE_STREAM_TRACE=<path>` dumps every routing decision; 601 tokens analysed
in `bench/policy.py`, `bench/hier.py`, `bench/predict.py`.

- At 67 slots: LRU 68.5%, ARC 68.5%, 2Q 66.9%, **BELADY 81.6%**. The shipped
  decaying-hotness policy already behaves exactly like LRU (hotness halves every
  64 tokens so most experts tie at 0 and the LRU tiebreak decides) - the LRU sim
  reproduces the measured 70.0% exactly. No online policy beats LRU here.
- Frequency-based admission: 55.9%, much worse. Routing is recency-driven, which is
  what a load-balanced router should look like.
- Cross-layer prefetch, the only way to overlap I/O with compute: best predictor
  (co-occurrence) gets 39% recall at K=10 with **5.9 wasted guesses per layer-token**.
  Its hits land mostly on experts that are already resident, so it buys nothing and
  costs several times the real miss count. Not viable.
- Zero-miss layers vs slots: 67 -> 15.2% (40.7 of 48 layers stall), 100 -> 24.5%,
  180 -> 45.5%. 2.7x the VRAM removes only a third of the stalls.
