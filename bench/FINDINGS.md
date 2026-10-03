# 177B streaming: measured decomposition and what moves the needle

Hardware: RTX 5070 12GB (PCIe gen3 x16), Ryzen 5 5600GT, 32GB DDR4-2400,
Kingston SNV2S1000G NVMe, Windows 10 LTSC. Model: Qwen3.8-Flash-Next
(`qwen4exp`), UD-IQ3_XXS, 76.32 GiB over 3 splits. FULL, UNPRUNED.

All numbers from `bench/bench.py` (one server per config, fixed 8-prompt set,
`n_predict=128`, 3 warmup gens discarded, median of 4). Routing is bit-for-bit
deterministic run to run, so the L1 hit / miss sequence repeats exactly and
within-batch comparisons are clean. **Across** batches the OS page cache carries
over and drifts results up by ~3-5%, so only compare inside one batch.

## The model, exactly

| | |
|---|---|
| layers | 48 (12 full attention, 36 SSM; `full_attention_interval=4`) |
| experts | 512 per layer, 10 active |
| per expert | `ffn_gate_exps` 490 KiB iq2_s + `ffn_up_exps` 490 KiB iq2_s + `ffn_down_exps` 900 KiB iq4_nl = **1.815 MiB** |
| routed experts total | 45.29 GiB |
| `per_layer_token_embd` | 26.82 GiB, 320,001,536 rows x 90 B, hashed n-gram table |
| other weights | ~4.21 GiB |
| per token | 480 expert reads = **871 MiB of weights touched** |

The PLE table is *not* a problem: it is `TENSOR_READ_LAZY`, excluded from the
mmap prefetch by `ranges_complement`, and `qwen4exp.cpp` already batches its row
prefetch through `llama_prefetch_rows`. Only a few rows per token are touched.

## Where the time goes (cache 6 GiB, steady state)

| metric | value |
|---|---|
| L1 slots | **67 per layer** of 512 = 13.1% capacity |
| L1 hit | **70%** |
| misses | 144/tok = **260 MiB/tok** from RAM or disk |
| stall (wall, waiting for experts) | 85-133 ms/tok |
| remap callback total | == stall (its own CPU work is negligible) |
| H2D upload | ~38 ms summed over 9 workers, ~11 ms wall |
| GPU utilisation | 17-20% |
| standby page cache | 15.5 GiB |

**token time ~= stall + 20 ms.** The 20 ms is GPU compute plus graph overhead;
everything else is waiting for expert weights.

So the targets are unambiguous:
- **15 tok/s needs stall <= 47 ms**
- **20 tok/s needs stall <= 30 ms**

## Drive, measured directly (unbuffered + overlapped, `scratchpad/qd.py`)

| read size | QD1 | QD4 | QD16 | QD32 |
|---|---|---|---|---|
| 512 KiB | 0.93 | 2.12 | 2.46 | 2.48 |
| 1024 KiB | 1.39 | 2.64 | 2.66 | 2.66 |
| 1932 KiB | 1.76 | 2.85 | 2.83 | 2.82 |

GB/s. The drive tops out at **~2.85 GB/s**. The old RESULTS.md figure of
"1.14 GB/s random at expert granularity" was a QD1 *buffered* number, and
matches 490/490/900 KiB reads at depth 1 - but see the null result below:
raising the queue depth did not help, because buffered Windows reads already
parallelise inside the cache manager.

260 MiB/tok at 2.85 GB/s is 91 ms, i.e. **~9 tok/s is the hard ceiling if every
L1 miss goes to disk.** Everything past that has to come out of RAM.

## Trace-driven residency model (`bench/hier.py` on `bench/route.trace`)

`LLAMA_MOE_STREAM_TRACE=<path>` dumps every decode routing decision; 601 tokens
captured. The model is validated: L1=67 with LRU predicts exactly the 70.0% L1
hit measured live.

| L1 slots | L2 | disk/tok | predicted |
|---|---|---|---|
| 67 | none | 265 MiB | 8.7 |
| 67 | 12 GiB | 117 MiB | 14.2 |
| 67 | 16 GiB | 84 MiB | 16.5 |
| 67 | 20 GiB | 66 MiB | 16.3 |
| 80 | 16 GiB | 84 MiB | 17.0 |
| 100 | 20 GiB | 66 MiB | 19.6 |

A 16 GiB host tier cuts disk traffic **68%**. That is the whole ballgame; L1
slots are worth about +3.6 points of hit per 13 extra slots.

### Cache policy headroom (`bench/policy.py`, 67 slots)

| policy | hit |
|---|---|
| LRU | 65.0% |
| shipped (decaying route hotness + LRU tiebreak) | 62.3% in sim, **70% live** |
| LFU + admission band | 55.9% |
| **BELADY (offline optimum)** | **77.7%** |

The shipped policy already behaves like LRU, because `route_hotness` is halved
every 64 tokens so most experts tie at 0 and the LRU tiebreak decides. Belady is
+7.7 points over LRU, worth about the same as going from 67 to 95 slots - so a
scan-resistant policy (ARC/LIRS/2Q) is the one policy idea still worth trying.
Frequency-based admission is actively harmful here: routing is recency-driven,
which is what a load-balanced MoE router should look like.

## Null and negative results (do not repeat)

| change | result |
|---|---|
| positional `ReadFile` with `OVERLAPPED` offset, global `io_mtx` removed | 7.41 -> 7.43. **No gain.** Buffered Windows reads already parallelise internally, so expert loads were never queue-depth starved. |
| `FILE_FLAG_RANDOM_ACCESS` buffered handles | 7.69 -> 7.71. **No gain.** No read-ahead amplification to remove. |
| pinned staging (`ggml_backend_dev_host_buffer_type`) | 6.62 -> 6.52. **No gain.** Upload is only ~11 ms wall of a ~130 ms token. |
| upload straight from the mmap + batched `PrefetchVirtualMemory` | 7.69 -> **3.23. Much worse.** The page faults then happen *inside* the CUDA copy, one cluster at a time and serialized across workers. Side effect worth knowing: it drove the process working set to 26.3 GiB and collapsed standby to 0.1 GiB, which proves ~26 GiB of RAM *can* be made resident. |
| explicit host L2 that copies arena -> staging, 8/12/16 GiB | 6.11 / 6.46 / 7.07 vs 7.43 control. Hit rate was good (42% at 8 GiB, 48% at 12 GiB; disk reads -35%) but the extra memcpy cost more than the disk reads it saved. Monotone in size, so the fix is to remove the copy, not to shrink the tier. |

## Fixed bugs in the inherited expert-streaming code

- Windows `llama_moe_stream_pread` serialized every read behind one global mutex
  with seek+read. Replaced with a positional `ReadFile`. (No speedup - see above -
  but it explains why `--moe-stream-io-threads` and `--moe-stream-direct` were
  both no-ops: the mutex pinned queue depth to 1, and `direct` was `GGML_UNUSED`
  on Windows.)
- The host L2 sized every slab to `max_nb_expert` (900 KiB), so each 490 KiB
  gate/up slab wasted 410 KiB - **30% of the arena**. Now one pool per weight size.
- The host L2 evicted by an O(n_slabs) linear scan under its lock on every
  insert. Now CLOCK with a rotating hand.
- The host L2 held its lock across the slab memcpy, serializing 237 MiB/tok of
  copying. Now the entry is pinned and the copy happens with the lock dropped.

## Instrumentation added

`LLAMA_MOE_STREAM_DEBUG=1` prints, every 32 tokens, to **stderr** (the library
log callback does not reach llama-server's own log file):

```
moe stream: N tok | L1 hit % | miss /tok (cold) | remap | stall | read | up ms/tok | src fast/med/slow
```

`read`/`up` are summed over the 9 I/O workers, so `read/stall` is the effective
queue depth. `src` buckets each slab read by latency (<150us / <600us / rest),
which separates page-cache hits from real disk reads.

Switches, all default-off except where noted:
`LLAMA_MOE_STREAM_L2_GB`, `_MMAP_COPY`, `_RANDOM_ACCESS` (default on),
`_PINNED` (default on), `_SERIAL_IO`, `_ADMIT=lo:hi`, `_TRACE=<path>`.

## Measurement hygiene: the page cache no longer carries over

The standby list keeps GGUF pages across server restarts, so a config measured
later in a session looked 3-5% faster than the same config measured earlier. That
drift was large enough to swamp every real effect here.

Purging the standby list properly needs `NtSetSystemInformation`
(MemoryPurgeStandbyList) or `SetSystemFileCacheSize`, both of which want
SeProfileSingleProcessPrivilege / admin - not available on this box. `bench/dropcache.py`
instead commits and touches private pages until the OS has handed the standby
pages over, then releases them:

```
standby 18.96 GiB -> 4.46 GiB   (first test)
standby 18.23 GiB -> 4.55 GiB   (two passes)
standby  ...      -> 4.56 GiB   (repeat, same session)
```

Reproducible to 0.01 GiB, which is what matters - the residual ~4.5 GiB is hot OS
file pages that re-fault immediately. One pass is enough. `bench/bench.py --cold`
runs it before starting each server, and `bench/matrix.py` passes `--cold` unless
`COLD=0`, so every config in a batch now starts from the same state.

Note this makes absolute numbers *lower* than the warm figures quoted elsewhere in
this file, because the run no longer inherits a warm cache. Compare cold to cold.

## ggml threadpool busy-wait

`cpuparams.poll` defaults to **50** and `ggml_barrier` spins on `_mm_pause()`. The
remap op is a CPU custom op, so all 48 layers per token enter the CPU threadpool;
thread 0 blocks on expert I/O while the other `n_threads-1` threads busy-spin for
the whole wait. On 6 cores with 6 I/O workers also wanting CPU, that steals cores
from exactly the threads the remap is waiting on.

| config | tok/s |
|---|---|
| default (poll 50, -t 6) | 7.66 |
| `--poll 0` | **7.92** (+3.4%) |
| `--poll 0 -t 2` | 6.71 |
| `--poll 0 -t 1` | 7.70 |

So the spinning is real but worth only ~3%, and starving the compute threads
outright costs more than it saves. Keep `--poll 0`, keep `-t 6`.

### Thread-count confound in the earlier L2 sweeps

Every L2 sweep above ran with the default **9** I/O threads, but 6 is optimal. At
6 threads the loader is nearly work-saturated - `(read + up) / 6` is 79 ms against
a 104 ms stall - whereas at 9 threads the same work divided by 9 is 41 ms against
a 127 ms stall, i.e. the extra threads added contention rather than throughput.
The L2 cuts read work ~40%, so it should translate at 6 threads even though it did
not at 9. That combination (L2 + 6 threads + `--poll 0`, cold start) is the open
experiment.

## L2 retested at the correct thread count: still loses

Cold start, 6 I/O threads, `--poll 0`, interleaved with its own control:

| config | tok/s |
|---|---|
| L_ctl (no L2) | **7.76** |
| L_l2_16 | 7.31 |

At the matched 71.9% L1 state the L2 run's stall was 137.0 ms against the
control's 79.9 ms at 72.8%, with `up` inflated to 73-100 ms summed by the
arena-fill memcpy. So the thread-count confound was not the explanation.

**The L2 is a dead end on this box, and the reason is simple: RAM is zero-sum.**
A 16-19 GiB arena leaves the machine with no free memory, so every arena write has
to make the OS reclaim a page, and that reclaim costs more than the disk read it
saved. The tier reliably halves disk reads and reliably loses wall-clock time. It
would be the right structure on a 64 GB box, where the arena would not have to
fight the page cache for the same bytes.

## Where this leaves the ceiling

Per token the model must touch 871 MiB of expert weights. Residency is
6 GiB VRAM (67 slots = 13.1% of experts) + ~15.5 GiB of page cache, so about 21.5
of 45.29 GiB. Everything else is a disk read at 2.85 GB/s, and the measured stall
runs at 90% of that limit.

token time = stall + ~20 ms, and stall is set by how many bytes have to leave the
SSD. That number is fixed by 32 GB of RAM, not by software - which is why every
software change that reduced I/O *work* (L2, more slots, parallel weight loads,
faster read primitives) left throughput alone.

## Headline A/B (cold, interleaved, 6 warmup gens, 40 matched cache states)

| config | tok/s | mean stall |
|---|---|---|
| as inherited (serial seek+read, whole-expert loads, unpinned staging, 9 io threads, poll 50) | 5.88 | 143.5 ms |
| best found (positional reads, split weights, pinned staging, 6 io threads, `--poll 0`) | **6.70** | **130.7 ms (-9.0%)** |

That is the honest figure for the software work: about **+14% throughput / -9% stall**
under identical cold-cache conditions. Every larger number quoted earlier in the
session came from a warmer page cache, not from a code change.

## What would actually reach 15-20 tok/s

Not a software change on this box. 871 MiB of expert weights must be touched per
token; residency is 6 GiB VRAM + ~15.5 GiB page cache of a 45.29 GiB expert set,
and the stall already runs at ~90% of the SSD's random-read limit.

**64 GB of RAM.** The full working set is 49.5 GiB. With 64 GB it is resident and
no streaming is needed - which is exactly what every faster community report on
this model+quant has (16-29 tok/s, all on 64 GB+). Two of four DIMM slots are free.

**The one software idea still open**, with the arithmetic:

Make the expert cache addressable by the GPU in host memory, instead of a VRAM
cache filled by explicit per-layer copies - a ggml CUDA buffer type backed by
`cudaHostAlloc(cudaHostAllocMapped)` whose device pointer the expert matmul kernels
can read directly. Then:

| tier | share | cost per token |
|---|---|---|
| 6 GiB VRAM resident | 70% | free |
| ~16 GiB pinned host, read in place over PCIe | 20% | 174 MiB at ~11 GB/s = 16 ms |
| disk | 10% | 87 MiB at 2.85 GB/s = 31 ms |

~47 ms of stall + ~20 ms of compute = **~15 tok/s**, and it removes the per-layer
fetch-and-wait chain for the host tier entirely, because the GPU just reads it
rather than waiting for a worker to copy it.

The work is in `ggml-cuda`: `ggml_backend_cuda_supports_buft` currently accepts
only the device's own buffers and split buffers, so the host buffer type would
have to become a legal operand location for the MoE matmuls. That is a real change
to ggml, not a change to the streaming layer, which is why it was not attempted here.

---

# CORRECTION: the "needs 64 GB" conclusion above is WRONG

anndev.org ran this exact model and quant at **10.8-12.4 tok/s decode on 12 GB VRAM
and 32 GB of RAM** - the same RAM this box has:
https://anndev.org/posts/2026-09-12-running-177b-moe-models-on-12gb-gpu-llama-moe-cache/

So 32 GB is enough and the RAM-capacity argument above is void. Two things explain
the gap, and only one of them is hardware:

**1. Their NVMe runs at PCIe Gen4; ours is capped at Gen3.** The post says Gen4 is
required and streams experts over it. Our drive measured **2.85 GB/s** ceiling
(`pcie.link.gen.max = 3` - the Ryzen 5 5600GT is a Cezanne APU, Gen3 only), and the
measured stall already runs at ~90% of that. A Gen4 link is roughly double, which
maps 6.7 -> ~12. This part is consistent with, not contradicted by, their result.

**2. They use a different and better-designed fork.**
`GenerelSchwerz/llama.cpp`, branch `moe-cache` - not the PR #25294 this session has
been patching. Its architecture is the one this session's own analysis concluded was
the only remaining path:

> `--moe-expert-cache-size N`: "keep N expert slabs per expert tensor on GPU with LRU
> eviction; **cold experts live in CPU pinned memory**"

A cache miss is therefore a **pinned-host to VRAM DMA at PCIe speed**, not a per-expert
disk read dispatched to a worker thread and waited on. That removes the fixed
per-layer fetch-and-wait chain which every measurement in this session identified as
the binding cost. It also has grouped decode, transfer/compute overlap, working MTP,
and `--moe-early-router` (run the next layer's router early and prepack its groups) -
an exact version of the cross-layer prefetch that failed here as a statistical predictor.

Their flags also match several things measured independently here: `-np 1`,
`--cache-ram 0`, and a *smaller* GPU cache (their 12 GB profile uses 20-24 slabs,
we used 67) - consistent with the finding that slot count is not the limiter.

**Known Windows hazard** (issue #104): Qwen3.8-Flash-Next emits corrupted output
(repeated "/") with the expert cache on Windows 11 unless the budgets are set
explicitly - `--moe-expert-cache-mib 2048 --moe-expert-cache-host-pinned-mb 2048`.
Unresolved upstream. Any result from this fork must be quality-checked, not just timed;
`bench/bench.py` already saves every generation to `logs/out_<name>.txt` for that.

Status: fork cloned to `src/moecache`, building via `build-moecache.bat` into
`src/moecache/build-mc`.

## Persisted cache warming (implemented in the PR fork)

`LLAMA_MOE_STREAM_HEAT=<path>` and `LLAMA_MOE_STREAM_WARM_GB=<n>`.

On shutdown the per-layer route hotness is written as a ranked text file. On the
next start every device slot is refilled with the hottest experts from that
ranking, and `WARM_GB` more is read with ordinary buffered reads so those slabs
land on the standby list as well.

Why it is worth having, and what it is not:

- From empty, this box needs **400-600 tokens** to reach steady state. The `cold`
  counter (first-ever touch of an expert) stays in the hundreds per 32 tokens for
  many generations, and most of that work is filling the page cache by demand
  misses one 490-900 KiB slab at a time.
- First generation measures 4.75-6.08 tok/s against 6.7-8.2 steady, so the
  warm-up is worth **25-30%** on the first few answers.
- It does **not** raise steady state, and the measurements say why: L1 hit 70% ->
  75.2% changed nothing (8.20 -> 8.16), and a 19 GiB tier that halved disk reads
  changed nothing. Getting better experts resident does not set the speed here.

So this is a cold-start feature. Report it as two numbers - cold and warm steady
state - and never warm on the prompts you then measure. `bench/bench.py` uses a
fixed 8-prompt set and discards warm-up generations, so warming on one set and
measuring on another is the correct protocol if this is ever quoted.

It should matter considerably more on the `moe-cache` fork, where cold experts
live in pinned host memory: there, what is resident in host RAM is what sets the
miss cost, so a persisted ranking controls throughput rather than just warm-up.

## The moe-cache fork, built and tested here: it cannot work on this box

Built from `GenerelSchwerz/llama.cpp` branch `moe-cache` (commit e69cf66) with the
same portable toolchain, into `src/moecache/build-mc`. It loads the model and
generates correct text, but:

| config | tok/s |
|---|---|
| `--moe-expert-cache-mib 2048 --moe-expert-cache-host-pinned-mb 2048` (issue #104 workaround) | ~1.5, disk at 148 MB/s |
| `--moe-expert-cache-mib 6144 --moe-expert-cache-host-pinned-mb 18432` | 1.58 cold -> 1.72 |

Against 6.70 on the PR fork. The startup log says why - **all 48 layers are refused
admission to the host tier**:

```
moe-cache-host: admission group=0 banks=3 bytes=1009266688
                reason=read_only_registration_unsupported cuda=none
```

### Root cause, verified directly

`moe-cache-host.cu:627` requires `cudaDevAttrHostRegisterReadOnlySupported` before
it will pin expert weights, because `--load-mode mmap` hands it a **read-only file
mapping** (`ggml_backend_cuda_moe_cached_buffer_from_host_ptr` wraps those with
`read_only = true`). A 15-line CUDA probe on this machine:

```
device: NVIDIA GeForce RTX 5070
cudaDevAttrHostRegisterSupported             = 1
cudaDevAttrHostRegisterReadOnlySupported     = 0   <-- required by the host tier
cudaDevAttrPageableMemoryAccess              = 0
cudaDevAttrCanUseHostPointerForRegisteredMem = 0
```

`cudaHostRegisterReadOnly` is not available under Windows WDDM. So the host expert
tier - the entire reason this fork is faster - is disabled, every expert falls back
to page faults off the mmap, and that is the same pathology measured earlier in this
file as 2x worse than reading into staging. Their own wiki says "Windows users must
follow the WDDM partial-pinning setup guide first" and their issue #104 is the
Windows breakage; this is that limitation, measured.

### Why the obvious workaround does not fit either

Plain (writable) host registration *is* supported here (attribute = 1), and the
fork has a writable path: with no mmap, `llama-model-loader.cpp:1976` reads straight
into a writable host buffer that carries `read_only = false` and would pin fine.
That is why their 12 GB profile uses `--load-mode none`.

It needs the expert weights in ordinary RAM: **45.29 GiB of experts against ~27 GiB
available**, and without mmap the 26.82 GiB PLE table also loses its lazy read and
would be loaded in full. It does not fit. Restricting layers with
`--moe-expert-cache-layers` would leave the unselected layers on the slow
mmap-fault path, which is what bounds the number above.

So anndev's 10.8-12.4 tok/s needs a platform where `cudaHostRegisterReadOnly` works
(i.e. not Windows WDDM) in addition to the Gen4 link.

## What would actually move this box

1. **Swap the CPU.** The Ryzen 5 5600GT is Cezanne, an APU, and is **PCIe Gen3 only** -
   that is the `pcie.link.gen.max = 3` measured at the start, and it caps the NVMe at
   the 2.85 GB/s measured here. The stall already runs at ~90% of that, so this is
   the binding constraint. A Vermeer part (Ryzen 5 5600 / 5600X, ~$80-100 used) gives
   PCIe Gen4 on the B550's CPU-attached M.2 and roughly doubles the ceiling:
   6.70 -> an expected 11-13 tok/s on the fork already tuned here, with no code change.
2. **Then 64 GB of RAM** if more is wanted; two DIMM slots are free.
3. **Or run the moe-cache fork under Linux**, where read-only host registration is
   supported and its host expert tier actually engages.

Ranked by cost per tok/s, the CPU swap is first by a wide margin, and it is the one
change this session's measurements point at directly.

## Persisted cache warming: measured, and it works where the model said it would

`LLAMA_MOE_STREAM_HEAT=<path>` (+ optional `LLAMA_MOE_STREAM_WARM_GB=<n>`).

Startup log on the second run:

```
moe stream: warmed 3216 device slots + 0.00 GiB into the page cache in 6.7 s
```

3216 = 67 slots x 48 layers, i.e. **every device slot preloaded** from the previous
run's ranking, in 6.7 s.

Both runs cold-cache (`--cold`), every generation measured (no warm-up discarded):

| | cold | preloaded | |
|---|---|---|---|
| 1st request, prefill | 2.29 | **4.46** | **+95%** |
| 1st request, decode | 5.52 | 5.86 | +6% |
| 2nd request, prefill | 4.09 | 4.42 | +8% |
| 2nd request, decode | 7.09 | 6.98 | -2% |
| 3rd request | 5.92 / 6.58 | 5.74 / 6.61 | noise |

**The win is time-to-first-token, not decode.** That follows from the measurements
above: prefill touches far more distinct experts per step than decode, so a
pre-filled L1 helps it much more. Steady-state decode is unchanged, exactly as the
residency experiments predicted (L1 hit 70 -> 75% changed nothing; halving disk
reads changed nothing).

So quote it as: first answer on a fresh server comes roughly twice as fast; decode
converges to the same number either way.

### Two bugs found before it ever ran

- **LRU stamps were inverted.** `reserve_slot_locked` bumps `slot_last_use`, so
  filling hottest-first gave the hottest expert the *oldest* stamp and it would have
  been evicted first. Fills coldest-first now.
- **`route_hotness` was not restored.** Eviction ranks by hotness first, so a cache
  warmed with all-zero hotness would have been discarded within ~30 tokens. The file
  stores heat alongside the ranking for this reason.

### And one found by testing

Saving only in the destructor is useless in practice: the harness (and any normal
`taskkill`, crash, or window close) uses `TerminateProcess`, so destructors never
run and the file was never written - which is why the first attempt showed
7.19 vs 7.22, i.e. two cold runs and pure noise. The ranking is now checkpointed
from the remap every ~256 tokens, written to `<path>.tmp` and renamed into place so
a kill mid-save cannot leave a truncated file.

### Honest-reporting caveat

In this test run 1's prompts are the same 8 prompts run 2 measures, which is the
best case for the ranking. To quote a user-facing number, warm on one prompt set
and measure on a different one. The question here was only whether the mechanism
engages, and it does.

## Pinned host memory: the actual ceiling on this box

`scratchpad/pinmax.cu` walks `cudaHostAlloc(cudaHostAllocPortable|Mapped)` upward:

```
 1 GiB OK    2 GiB OK    4 GiB OK    6 GiB OK    8 GiB OK
10 GiB OK   12 GiB OK   14 GiB OK   16 GiB FAILED (out of memory)
```

So **14 GiB is the limit** for a single pinned allocation here, against 32 GB of
RAM. That matters for two reasons:

1. It silently broke the first pinned-arena attempt. `LLAMA_MOE_STREAM_L2_GB=16`
   logged `pageable`, because `ggml_backend_buft_alloc_buffer` on the host buffer
   type failed and the code fell back to `moe_aligned_alloc`. The run was therefore
   measuring the old copying path, not the pinned one. The startup line now says
   which of the two happened - without it this would have looked like "pinning did
   not help" when pinning never happened.
2. Combined with `cudaDevAttrHostRegisterReadOnlySupported = 0`, it bounds what any
   host-expert tier can be on Windows here: at most ~14 GiB of writable pinned
   memory, and zero bytes of pinned read-only mmap. The moe-cache fork wants the
   latter, which is why it cannot engage at all.

A 12 GiB arena does pin, so that is the configuration worth measuring: the L2 at
19 GiB pageable was already break-even (7.91 vs 7.95 control), and a pinned arena
removes the arena->staging memcpy entirely by making the upload a direct DMA.

### Page-cache warming (`LLAMA_MOE_STREAM_WARM_GB`): additive, and it does not pay the arena penalty

Unlike the L2 arena, this allocates nothing - it reads the right slabs so the OS page
cache holds them, instead of letting demand misses fill it at random. So there is no
RAM taken away from anything, only startup time.

| config | startup | per-generation (cold, every gen measured) |
|---|---|---|
| device slots only | 7.0 s | 5.58  7.13  6.92  8.20 |
| + 4 GiB page cache | 10.6 s | **6.21**  7.31  6.95  8.22 |
| + 7.58 GiB page cache | 14.3 s | (see below) |

**First generation 5.58 -> 6.21, +11%**, for +3.6 s of startup, and generations 2-4
are untouched (7.31/6.95/8.22 against 7.13/6.92/8.20). That is exactly the shape a
warm-up feature should have: it buys the first answer and nothing else.

Note the 8 GiB request only warmed **7.58 GiB** - the ranking ran out. The heat file
only lists experts that were actually routed to, so a short history caps how much
there is to preload. A longer-used heat file would cover more.

Stacked with the device-slot preload, first-answer decode goes 5.52 (no warming) ->
5.86 (slots) -> 6.21 (slots + 4 GiB), i.e. **+12.5%**, on top of the +95% on first
prefill from the slots alone.

## Pinned L2 arena: it pins, it works, and it still loses

`LLAMA_MOE_STREAM_L2_GB=12` now reports what actually happened:

```
moe stream: L2 host cache 12.00 GiB, 8886 experts, 2 pools, pinned (uploads DMA from the arena)
```

So the upload is a direct DMA out of the arena with no staging copy - the thing the
`HostRegisterSupported = 1` probe said was available. Matched cache states against
its own cold control:

| at L1 71.9% | control | 12 GiB pinned |
|---|---|---|
| stall | **117.5** | 136.3 (+16% worse) |
| read, summed | 473.9 | 459.2 |
| upload, summed | 37.2 | **57.9 (+56%)** |
| src fast | 33.7 | **213.8** |
| src med | 171.7 | **11.2** |
| src slow (disk) | 199.0 | 179.4 |

At L1 62.5% it is worse again: stall 145.4 -> 176.7. Over the full run:

| | tok/s | mean stall, 32 matched states |
|---|---|---|
| control | **8.14** | **125.4 ms** |
| 12 GiB pinned arena | 6.95 | 152.1 ms (**+21.3%**) |

and the arena's own hit rate was **55%** - it served over half of every L1 miss out
of pinned host RAM and still lost 21%. There is no version of this idea that wins
here.

### Correction: it is the RAM, not the DMA path

A pageable arm at the same 12 GiB settles which of the two candidate causes is real:

| | tok/s | upload, summed | src med |
|---|---|---|---|
| control | **8.14** | 37.2 | **171.7** |
| 12 GiB pinned (DMA from arena) | 6.95 | 57.9 | 11.2 |
| 12 GiB pageable (copies to staging) | 6.92 | 66.8 | 41.1 |

Pinned and pageable are **the same** (6.95 vs 6.92). So the first explanation drafted
here - that the DMA is slower out of a big arena - was **wrong**: the pageable arm
never DMAs from the arena and its upload time is *higher* still. Both arena configs
are uniformly slower for one reason only:

**The 12 GiB it takes is 12 GiB the page cache no longer has.** `src med` drops from
171.7 to 41.1 (pageable) or 11.2 (pinned), i.e. the arena *replaced* the middle tier
instead of adding to it, and a 55%-hit arena is worth less than the page cache it
displaced. Pinning only changes how hard the displacement bites, not whether it does.

That is the same zero-sum wall as the pageable arena, now with direct evidence of
the mechanism rather than an inference. **Every L2 variant tried here loses:
pageable copying (8/12/16/19 GiB), zero-extra-copy, pinned-staging, and pinned-DMA.**
The conclusion is not "the implementation was wrong" - it is that on 32 GB there is
no spare RAM to build a tier out of.

### The honest accounting for warming

Percentages flatter this feature; wall clock does not. For one 15-token prompt and
128 generated tokens, measured startup included:

| config | startup | prefill | decode | total | vs none |
|---|---|---|---|---|---|
| no warming | 1.0 s | 6.6 s | 23.2 s | 30.7 s | - |
| device slots | 7.0 s | 3.9 s | 22.9 s | 33.8 s | **+3.1 s worse** |
| + 4 GiB | 10.6 s | 2.9 s | 20.6 s | 34.1 s | **+3.4 s worse** |
| + 7.58 GiB | 14.3 s | 2.3 s | 20.1 s | 36.7 s | **+5.9 s worse** |

and with the startup already paid, i.e. the server was up before you asked:

| config | prefill + decode | vs none |
|---|---|---|
| no warming | 29.7 s | - |
| device slots | 26.8 s | -2.9 s |
| + 4 GiB | 23.5 s | -6.2 s |
| + 7.58 GiB | **22.4 s** | **-7.4 s** |

So warming **moves** the wait from "while you are waiting for an answer" to "before
you typed", roughly 1:1. It is a clear win when the server is left running, which is
how `run-full177b.bat` is meant to be used, and a 3-6 s loss if you launch it fresh
for every single question. `WARM_GB=8` is enabled in the launcher for the first case,
with the tradeoff written in the comments and a one-line way to turn it off.

What it is *not* is a throughput gain. Steady-state decode is identical across all
four configs (generations 2-4: 7.04-7.31, 6.70-6.95, 8.20-8.22).

---

# SESSION 3: 7.2 -> 11.9 tok/s (cold, same binary, interleaved)

Everything above this line was measured against a build whose I/O path was serialized by a
Windows file-object lock. That invalidates the *interpretation* of most of the null results
above — not the measurements, but the conclusion drawn from them ("queue depth was never the
problem"). It was.

## Headline

`bench/sweep.py --cold`, four generations each, the two configs interleaved, one binary with
`LLAMA_MOE_STREAM_ONE_HANDLE=1` reproducing the old behaviour:

| config | tok/s |
|---|---|
| as inherited (one shared handle, 9 io threads, poll 50, no hot file) | 6.95 / 7.36 |
| shipping (per-worker handles, 4 io threads, `--poll 0`, 11 GiB page-locked hot file) | **11.88 / 12.00** |

Warm, not cold-started: **12.7**. A fresh conversation: 10.6 → 11.6 across its turns.
`bench/qual.py` passes 6/6 and the generated text is identical to a control run.

## 1. The shared file handle (the big one)

`open_files` opened one `HANDLE` per split with `FILE_FLAG_RANDOM_ACCESS` and no
`FILE_FLAG_OVERLAPPED`, and every worker issued `ReadFile` through it with an `OVERLAPPED`
offset. A synchronous Windows file object serializes I/O for the duration of each read, so
queue depth was pinned at 1 per file. `bench/qd.py`, 6 threads, random reads over the
46 GiB split:

| read size | shared handle | per-worker handles |
|---|---|---|
| 448 KiB unbuffered | 0.86 GB/s | 2.21 GB/s |
| 896 KiB unbuffered | 1.31 GB/s | 2.71 GB/s |
| 448 KiB buffered | 1.81 GB/s | 5.80 GB/s |
| 896 KiB buffered | 3.03 GB/s | 11.49 GB/s |

Fix: `h_rand` / `h_nobuf` are now `[worker][file]`. Stall fell 35-42% at matched
(L1 hit, miss/tok, cold) states; 6.36 -> 10.86 tok/s on the same harness.

This also flipped two earlier conclusions: `FILE_FLAG_RANDOM_ACCESS` now matters a lot
(8.15 vs 12.03 without it) and pinned staging is worth ~7% (11.13 vs 12.03 without).

## 2. Page-locking the hot experts

With the reads unblocked, the cost is DRAM, not the SSD: a page-cache hit costs three passes
over memory (cache manager memcpy in, DMA read back out). `cudaHostRegisterReadOnly` is
unsupported here, but a **read-write** file mapping registers fine and DMAs at 9.6-11.7 GB/s
with no copy (`scratchpad/pinmap.cu`). `bench/mkhot.py` + `LLAMA_MOE_STREAM_HOT` put the
hottest ~11 GiB of experts in such a mapping. Summed read time 289 -> 109 ms/tok,
stall 67 -> 55 ms/tok, and first-turn speed roughly doubled (5.5/9.6 -> 12.6/14.1).

## 3. The floor, measured rather than argued

`LLAMA_MOE_STREAM_NOWAIT=1` loads no experts at all: **47.4 tok/s**, i.e. 21 ms/token of
compute and graph. PCIe Gen3 x16 measures 12.8 GB/s (`scratchpad/h2d.cu`) and 267 MiB has to
cross it per token, which is another ~21 ms that only a higher L1 hit rate can remove — and
`--moe-stream-cache 8` falls off the driver's VRAM spill cliff to 3.85 tok/s.

Page cache is worth **0.17 tok/s per GiB** (`bench/ramslope.py`), so more RAM is a much
smaller lever than it looks. DOCP (the DIMMs are a 3600 kit at 2400) and a Gen4 CPU are the
two hardware changes that still matter.

## New tooling

| | |
|---|---|
| `bench/qd.py` | shared vs per-thread handle throughput — the measurement that found the bug |
| `bench/diskprobe.py` | MiB/token that actually left the SSD, from the OS counters |
| `bench/sweep.py` | json-driven config sweep, one server per config, tok/s + disk + memory |
| `bench/chat.py` | the single-topic conversation case, which `sweep.py` deliberately is not |
| `bench/qual.py` | 6 factual checks + diff against a control. Run before believing a number |
| `bench/mkhot.py` | builds the page-locked hot-expert file from a heat ranking |
| `bench/alloc.py` | exact LRU stack distances; per-layer slot budget via concave envelope |
| `bench/ramslope.py`, `hog.py` | tok/s as a function of available page cache |
| `scratchpad/h2d.cu`, `pinmap.cu` | PCIe ceiling, and whether a file mapping can be locked |

=== CORRECTION: THE FIRST HOT FILE WAS BUILT ON THE BENCHMARK PROMPTS ===
The numbers in the hot-file section above (12.7 warm, 11.88/12.00 cold) are INFLATED. Here
is how, because the mechanism will catch anyone who repeats this:

llama-moe-stream checkpoints the expert ranking to LLAMA_MOE_STREAM_HEAT every ~256 tokens
while it runs. bench/heatgen.py had written bench/expert2.heat at 15:05 from its own 24
training prompts; the residency sweep (cfg_res.json, 15:09) then ran four configs that set
LLAMA_MOE_STREAM_HEAT to that same file while generating on the eight MEASUREMENT prompts,
and the server rewrote it - mtime 15:20:16. bench/mkhot.py built the hot file from it at
17:52. So the ranking had seen the test set.

The two prompt sets never overlapped (zero exact matches, max word-jaccard 0.25). The leak
was entirely through the file.

HOW BIG IT WAS. Rebuilt the ranking from the training prompts only (bench/expert_train.heat)
and rebuilt the file:
    contaminated file  17832 slabs
    clean file         17829 slabs
    shared             12174   = 68.3% of the clean file
    only in the dirty one 5658 <- slabs the measured prompts put there, i.e. 32% of it

Re-measured, same protocol, cold, interleaved, 4 generations each:
                              contaminated     clean
    cold shipping             11.88 / 12.00    11.55 / 11.57
    warm 8-prompt sweep       12.7             11.43 / 11.58
    conversation median       10.59 / 10.76    10.30
    conversation first turns  12.6 / 14.1      12.1 / 13.4
=> contamination was worth about +3% cold and +9% warm. It did NOT account for the result,
which is the useful part: the hot file generalises - built on 24 unrelated prompts it still
doubles the first two turns of a conversation about B-trees, a topic in neither set.

THE CLEAN NUMBERS, which are the ones to quote:
    as inherited (one shared handle, 9 io threads, poll 50, no hot file)   5.85 - 7.07
    shipping (per-worker handles, io 4, --poll 0, 11 GiB clean hot file)   11.55 - 11.58
    warm, no hot file 10.55  ->  with hot file 11.43 / 11.58               (+9%)
    conversation, no hot file median 9.46 -> with 10.30, first turns
        6.0 / 7.9 -> 12.1 / 13.4                                          (~2x)

GUARDS NOW IN PLACE so this cannot recur:
 - bench/heatgen.py writes bench/expert_train.heat, a path no benchmark config uses.
 - bench/sweep.py raises if a config sets LLAMA_MOE_STREAM_HEAT at all. A measurement is
   never allowed to write the ranking it will later be scored against.
 - the contaminated experts.hot was deleted rather than kept around.
NOTE the same trap applies to the shipped launcher: run-full177b.bat points HEAT at
kvcache/expert.heat, which is correct for real use (the ranking should track YOUR usage),
but never build a hot file from kvcache/expert.heat and then benchmark on prompts you have
already run through that server.

VARIANCE WARNING: the inherited arm measured 6.95/7.36 in one batch and 7.07/5.85 in
another, and a warm no-hot arm once came in at 7.37 against 10.55 for the same config with a
93 s load time. This box gets noisy when the page cache is churning. Interleave the arms,
run at least two of each, and distrust any single config.
