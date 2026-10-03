# Running a 177B model on a 12 GB card

A **full, unpruned 177B mixture-of-experts model** — 76.3 GiB of weights — generating at
**11.5 tok/s** on a single RTX 5070 (12 GB) with 32 GB of system RAM, plus a 35B model at
**63 tok/s** on the same card.

Every number in this file was measured on this machine with the harness in `bench/`, and
every speed claim is gated by `bench/qual.py`, which diffs the generated text against a
control run. Nothing here is estimated or quoted from someone else's box.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [What you get](#2-what-you-get)
3. [How it works — the methods](#3-how-it-works--the-methods)
   - [Why a MoE model can be offloaded at all](#31-why-a-moe-model-can-be-offloaded-at-all)
   - [Method 1: `-ncmoe` residency (the 35B path)](#32-method-1-ncmoe-residency-the-35b-path)
   - [Method 2: expert streaming (the 177B path)](#33-method-2-expert-streaming-the-177b-path)
   - [Method 3: one file handle per I/O worker](#34-method-3-one-file-handle-per-io-worker)
   - [Method 4: the page-locked hot-expert file](#35-method-4-the-page-locked-hot-expert-file)
   - [Method 5: the persisted heat ranking](#36-method-5-the-persisted-heat-ranking)
   - [Method 6: things that are deliberately off](#37-method-6-things-that-are-deliberately-off)
4. [Where the time actually goes](#4-where-the-time-actually-goes)
5. [Using the server](#5-using-the-server)
6. [Tuning](#6-tuning)
7. [Maintenance: rebuilding the hot file](#7-maintenance-rebuilding-the-hot-file)
8. [Measuring anything yourself](#8-measuring-anything-yourself)
9. [Environment variable reference](#9-environment-variable-reference)
10. [If you cloned this](#10-if-you-cloned-this)
11. [Building from source](#11-building-from-source)
12. [Troubleshooting](#12-troubleshooting)
13. [What did not work](#13-what-did-not-work)
14. [Hardware, and what to upgrade](#14-hardware-and-what-to-upgrade)

---

## 1. Quick start

**177B (the big one):**

1. Double-click **`run-full177b.bat`**
2. Wait for `server is listening on http://127.0.0.1:8080`. First start takes ~40 s because
   it pages in and locks an 11 GiB file; later starts are the same unless you reboot.
3. Open <http://127.0.0.1:8080>

**35B (fast one):**

1. Double-click **`run.bat`**
2. Wait ~20 s, open <http://127.0.0.1:8080>

Keep the console window open — closing it stops the server. Only run one at a time; they
both bind port 8080 and both want most of your VRAM.

### Use it from other apps

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| API key | anything, e.g. `local` |
| Model name | anything |

Works with LM Studio, OpenCode, Continue, Open WebUI, Cline, the `openai` Python package —
anything that speaks the OpenAI API. **Tool / function calling works** (verified live): the
chat template carries `tools` and `tool_calls`, and `--jinja` is on, so you can POST a
`tools` array to `/v1/chat/completions` and get back a proper `tool_calls` response with
`finish_reason: tool_calls`.

---

## 2. What you get

| Profile | Launcher | Model | tok/s | VRAM | Context |
|---|---|---|---|---|---|
| **177B** | `run-full177b.bat` | Qwen3.8-Flash-Next 177B, UD-IQ3_XXS, **unpruned** | **11.5** | 11.1 GB | 32768 |
| **35B FAST** | `run.bat` (`PROFILE=FAST`) | Qwen3.6-35B-A3B, UD-IQ3_XXS | 62.6 | 10.8 GB | 8192 |
| 35B QUALITY | `run.bat` (`PROFILE=QUALITY`) | Qwen3.6-35B-A3B, Q4_K_M + MTP | 44.8 | 10.9 GB | 8192 |
| *(reference)* | — | Qwen3.8-27B **dense**, same card | 7.5 | 11.2 GB | — |

> **Only the 177B and the 35B FAST profile are runnable as shipped.** `run.bat`'s QUALITY
> profile wants `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and `run-flashnext.bat` wants a
> `models/reap128/` build; neither file is present. Download them or ignore those profiles.

### The 177B numbers in detail

Measured with `bench/sweep.py` (eight fixed unrelated prompts in rotation, `cache_prompt`
off, three warm-up generations discarded, median of four) and `bench/chat.py` (one
conversation, eight turns on one topic):

| scenario | tok/s |
|---|---|
| eight unrelated prompts in rotation — the pessimistic case | **11.43 – 11.58** |
| cold start, same thing | 11.55 – 11.57 |
| a fresh conversation, turn by turn | 9.1 → 13.4, median **10.30** |
| **as inherited** (same binary, `LLAMA_MOE_STREAM_ONE_HANDLE=1`, no hot file) | 5.85 – 7.07 |

**~7.0 → 11.6 tok/s, +65%**, cold-started and interleaved against its own control.

Why "eight unrelated prompts" is the pessimistic case: every prompt drags a different set of
experts through the cache, so the cache is never allowed to settle. A real session stays on
a topic and does better, which is what the conversation row shows.

### Quality

This is the **full, unpruned** model. `bench/qual.py` asks six factual questions and diffs
the whole output against a control run with the hot file disabled:

```
6/6 factual checks passed
IDENTICAL to control
```

The streaming layer only changes *which cache slot* an expert's weights live in. It never
changes which experts the router picked, so the output is bit-identical to running the same
model without streaming. If it ever isn't, something is broken — see
[Troubleshooting](#11-troubleshooting).

---

## 3. How it works — the methods

### 3.1 Why a MoE model can be offloaded at all

A **dense** model reads 100% of its weights for every token. Offload any of it and you pay
the full price over PCIe every single token. Measured here: a dense 27B at IQ3 runs at
**7.5 tok/s** on this card.

A **mixture-of-experts** model is different. Qwen3.8-Flash-Next has 48 layers, each with
**512 expert FFNs** of which the router picks **10** per token. So roughly 3% of the expert
weights move per token, and the other 97% can sit in system RAM or on disk doing nothing.
Same card, a *larger* MoE model: **62.6 tok/s** for the 35B.

The whole project is about exploiting that ratio as far as it goes.

```
per token, 177B:   48 layers x 10 experts x 1.886 MiB = 905.8 MiB of expert weight touched
                   but only ~30% of it misses the VRAM cache = ~267 MiB actually moved
total expert set:  48 x 512 x 1.886 MiB = 45.29 GiB
rest of the model: 26.82 GiB per-layer embedding table (lazy) + 4.21 GiB everything else
```

### 3.2 Method 1: `-ncmoe` residency (the 35B path)

For a model that *fits* in VRAM + RAM, you do not need to stream anything. You just need to
put the right things in the right place.

`--n-cpu-moe N` (`-ncmoe`) keeps attention, shared experts, embeddings and the KV cache in
VRAM, and puts **only the routed expert FFNs of N layers** in system RAM. Attention is dense
and used every token, so it must stay on the GPU; routed experts are sparse and cheap to
leave behind.

**The quant is a speed knob, not just a size knob.** Smaller experts mean more of them fit
in VRAM, which moves the point at which you must start offloading:

| `NCMOE` | IQ3_XXS tok/s | VRAM | |
|---|---|---|---|
| 7 | 72.7 | 11.5 GB | fastest, too tight if you also use the desktop |
| 8 | 69.0 | 11.3 GB | |
| **10** | **62.6** | **10.8 GB** | **shipped default** |
| 12 | 57.4 | 10.2 GB | |
| 20 | 46.7 | ~9 GB | for 10 GB cards |
| 26 | 38.9 | ~8 GB | for 8 GB cards |

Q4_K_M needs `NCMOE` 26 to reach 44.8. Going IQ3_XXS moved the knee from 22 to 7 and was
worth **47 → 82 tok/s** — more than every buffer-tuning trick combined.

> **⚠ The cliff.** Set `NCMOE` too low and speed *collapses* instead of erroring, because
> the NVIDIA driver silently spills VRAM into system RAM over PCIe:
> ```
> IQ3_XXS  NCMOE=7  -> 82.5 tok/s
> IQ3_XXS  NCMOE=6  -> 80.7 or 57.4   <- UNSTABLE, sits exactly on the edge
> IQ3_XXS  NCMOE=5  -> 10.2 tok/s     <- 8x slower, no error message
> ```
> Stay 3–4 above the cliff, not 1. Chrome and games eat VRAM and can push you over it.
> **If speed is unexpectedly bad, raise `NCMOE` by 4 before debugging anything else.**

### 3.3 Method 2: expert streaming (the 177B path)

45.29 GiB of experts do not fit in 12 GB of VRAM plus 32 GB of RAM. So for the 177B the
expert tensors are **never materialised at all**. Instead:

- Each streamed weight tensor gets a **device-side cache** of `n_slots` expert slabs.
  At `--moe-stream-cache 6` that is **67 slots per layer out of 512** — 13.1% of the
  experts, 6.07 GiB of VRAM.
- `build_moe_ffn` gets an extra CPU op, the **remap**, inserted right after the router's
  top-k. It takes the expert ids the router chose and rewrites them into *cache slot* ids.
- Any expert that is not resident is queued to a pool of I/O worker threads, which read its
  three weight slabs (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) out of the GGUF and
  upload them into free cache slots. The remap blocks until they land.
- Eviction is by decaying route hotness with an LRU tiebreak.

Crucially **the remap only changes where an expert's bytes are, never which expert runs**.
That is why streaming is lossless.

Measured hit rate: **70%** of expert lookups are already resident, so ~141 of 480 slab reads
per token actually have to fetch something.

```
   router picks 10 of 512  ->  remap  ->  7 already in VRAM (free)
                                      ->  3 missed: read + upload, everything waits
                               x 48 layers, strictly sequential
```

### 3.4 Method 3: one file handle per I/O worker

**This is the single biggest fix in the project, and it is four lines.**

The inherited code opened **one** Windows `HANDLE` per GGUF split and had every I/O worker
call `ReadFile` through it with an `OVERLAPPED` offset. That looks lock-free. It is not: a
Windows file object opened *without* `FILE_FLAG_OVERLAPPED` is **synchronous**, and the I/O
manager holds its lock for the duration of every read. Concurrent reads through one handle
queue behind each other no matter how many threads issue them — **queue depth pinned at 1**.

Measured directly with `bench/qd.py`, 6 threads, random reads over the 46 GiB split:

| read size | one shared handle | one handle per worker |
|---|---|---|
| 448 KiB unbuffered | 0.86 GB/s | **2.21 GB/s** |
| 896 KiB unbuffered | 1.31 GB/s | **2.71 GB/s** |
| 448 KiB buffered | 1.81 GB/s | **5.80 GB/s** |
| 896 KiB buffered | 3.03 GB/s | **11.49 GB/s** |

This one bug had been generating false conclusions for two sessions. `--moe-stream-io-threads`,
splitting an expert into three parallel weight loads, `FILE_FLAG_RANDOM_ACCESS`, and removing
an older global I/O mutex had **all** measured as no-ops — because none of them could raise a
queue depth the kernel was holding at 1.

`h_rand` and `h_nobuf` are now `[worker][file]`. Result: stall down 35–42% at matched cache
states, **6.4 → 10.9 tok/s**. Two previously-dead flags came back to life as well:
`FILE_FLAG_RANDOM_ACCESS` is now worth 8.15 → 12.03, and pinned staging ~7%.

### 3.5 Method 4: the page-locked hot-expert file

With reads unblocked, the bottleneck moved to **host memory bandwidth**, not the SSD.

Delivering one byte of expert weight from the OS page cache to VRAM costs **three passes
over DRAM**: the cache manager memcpies the slab into pinned staging, and then the DMA engine
reads those same bytes back out. On DDR4-2400 that is the binding resource.

The obvious fix — let the GPU DMA straight out of the page cache — was believed impossible
here. ggml's `ggml_backend_cuda_register_host_buffer` asks for `cudaHostRegisterReadOnly`,
and this machine reports:

```
cudaDevAttrHostRegisterSupported             = 1
cudaDevAttrHostRegisterReadOnlySupported     = 0   <-- not available under Windows WDDM
```

so a **read-only** GGUF mapping can never be page-locked. (That is exactly what makes the
`GenerelSchwerz/llama.cpp moe-cache` fork run at 1.7 tok/s here instead of its advertised
10–12.)

**The way around it:** open the file **read-write** and map it `PAGE_READWRITE`. Plain,
writable registration *is* supported. Nothing ever writes through the mapping. Measured with
`scratchpad/pinmap.cu`:

| mapping size | register time | DMA out of it, no CPU copy |
|---|---|---|
| 4 GiB | 0.7 s | **11.67 GB/s** |
| 8 GiB | 1.2 s | 10.89 GB/s |
| 12 GiB | 1.7 s | 9.60 GB/s |
| 14 GiB | 1.9 s | 8.32 GB/s (14 GiB is the ceiling) |

against a 12.8 GB/s link ceiling. So `bench/mkhot.py` writes the hottest ~11 GiB of experts
into a file of their own, and `LLAMA_MOE_STREAM_HOT` maps it read-write, pages it in with
the I/O threads, locks it, and uploads those slabs **in place** — one DRAM pass instead of
three, and a locked page can never fault to disk.

Measured effect: summed read time per token **289 → 109 ms**, stall 67 → 55 ms, and on a
fresh conversation the first two turns go **6.0 / 7.9 → 12.1 / 13.4 tok/s**. Steady state
gains ~9%; time-to-first-useful-answer roughly doubles.

Two things that are easy to get wrong, both of which cost a rebuild to learn:

- **The file must be disjoint from the VRAM cache.** Built from rank 0 it is mostly a second
  copy of what L1 already holds and is worth only +4%. `mkhot.py`'s 4th argument skips the
  top 67 per layer, making it the tier *behind* L1.
- **The index must be keyed on the slab's source byte offset**, never on
  (layer, weight position). llama.cpp registers a layer's streamed weights in model-builder
  order, and `qwen4exp` creates `ffn_down_exps` **before** gate/up. A positional key serves
  the wrong tensor for every slab — which runs at *full speed* (48 tok/s, because corrupted
  routing collapses onto a handful of experts) and emits `!!!!!!!!`. This is why
  `bench/qual.py` exists and why no speed number here is reported without it.

### 3.6 Method 5: the persisted heat ranking

`LLAMA_MOE_STREAM_HEAT=<path>` records how often each expert is actually routed to, ranked
per layer, checkpointed every ~256 tokens and reloaded at startup. It does two jobs:

1. **Cold start.** At startup every one of the 3216 device slots is refilled from the
   ranking, so the cache begins warm instead of taking 400–600 tokens to discover the hot
   set. Steady state is unchanged — this buys the first few answers only.
2. **It is the input to `mkhot.py`.** The hot file is just "the top N GiB of this ranking,
   minus what the VRAM cache holds anyway". So **the hot file gets better the more you use
   the model**, because the ranking tracks your actual usage.

Two implementation details that mattered:

- The ranking is sorted by an **undecayed** use count, not by the decaying `route_hotness`
  used for eviction. Hotness halves every 64 tokens, so a steadily-but-rarely used expert
  decayed to 0 and vanished from the file — which capped the ranking at ~7300 experts
  (13.5 GiB), less than the page cache can hold. With the undecayed count it reaches the
  full 24,135 experts / 44.45 GiB.
- Page-cache warming reads **coldest-first**. Reading hottest-first means the tail of the
  warm evicts the head of it, so you end up holding exactly the wrong half.

### 3.7 Method 6: things that are deliberately off

- **`--poll 0`.** ggml's threadpool busy-waits by default (`poll=50`, spinning on
  `_mm_pause`). The remap is a CPU op, so while thread 0 waits on I/O the other compute
  threads spin and starve the I/O workers of cores. Worth ~+3%.
- **`LLAMA_MOE_STREAM_KEEP_MMAP=1`.** The upstream streaming PR force-disables mmap. Keeping
  it leaves the 26.82 GiB per-layer embedding table lazily read instead of resident. This is
  not optional — without it the table has to be loaded in full and nothing fits.
- **Speculative decoding is off on the 177B** (and on the 35B FAST profile). See
  [What did not work](#12-what-did-not-work).
- **The async batched uploader is off.** It works exactly as designed — upload time drops
  53% — and buys nothing, because the uploads were already overlapping the reads. Left in
  behind `LLAMA_MOE_STREAM_ASYNC_UP=1`.

---

## 4. Where the time actually goes

Rather than reasoning about this, measure it. `LLAMA_MOE_STREAM_NOWAIT=1` maps every
requested expert onto an arbitrary resident slot and loads nothing at all. The output is
garbage by construction; the point is the clock.

```
LLAMA_MOE_STREAM_NOWAIT=1   ->   47.4 tok/s   =  21 ms/token of GPU compute + graph
```

So of an ~86 ms token: **21 ms is compute, ~65 ms is expert I/O**, and 47 tok/s is the hard
ceiling that no amount of I/O work could ever pass.

Breaking the I/O down further, at the measured 70% hit rate:

| term | size | cost |
|---|---|---|
| PCIe upload | 267 MiB/token at a measured 12.8 GB/s ceiling (`scratchpad/h2d.cu`) | ~21 ms |
| page-cache reads | ~200 MiB/token, memcpy into staging | ~15 ms |
| real disk reads | ~70 MiB/token, ~170 ops | ~20 ms |
| GPU compute + graph | — | 21 ms |

The upload term can only be reduced by a **higher L1 hit rate**, which needs VRAM this card
does not have: `--moe-stream-cache 8` falls off the driver's spill cliff to 3.85 tok/s.

And a number worth knowing before anyone buys RAM: **the OS page cache is worth about
0.17 tok/s per GiB** here (`bench/ramslope.py`, measured by holding 0/2/4/6 GiB hostage).
Recovering the entire ~8 GiB that the GPU driver's WDDM backing store holds would be worth
about 1.4 tok/s. 64 GB of RAM is *not* the big lever it is usually assumed to be, because
the extra experts it buys are cold ones.

---

## 5. Using the server

### Endpoints

Standard llama.cpp server. The useful ones:

| | |
|---|---|
| `GET /health` | `{"status":"ok"}` once the model is loaded |
| `POST /completion` | raw completion, llama.cpp-native |
| `POST /v1/chat/completions` | OpenAI-compatible, supports `tools` |
| `POST /v1/models` | model list, for clients that insist |

### Tool calling

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

r = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Weather in Tokyo? Use the tool."}],
        tools=tools)
print(r.choices[0].message.tool_calls)
```

Verified working on this build — returns a proper `tool_calls` with
`finish_reason: tool_calls`.

### Reasoning output

This is a reasoning model and will often emit a `<think>…</think>` block before answering,
sometimes a long one. To turn that off, send:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

### Context

The launcher sets **32768**. The model itself supports **262144**. Measured:

| `-c` | tok/s |
|---|---|
| 4096 | 11.36 |
| 16384 | 10.66 |
| **32768** | **11.35** |
| 65536 | 7.88 |
| 131072 | 4.97 |

32k is free; 64k and up are not, because the KV cache pushes the expert cache off its VRAM
cliff. If you need more context, you must also lower `--moe-stream-cache`, and you should
re-measure rather than assume.

### Conversations resume warm

`--slot-save-path kvcache` persists the KV cache, so reopening a conversation skips
re-prefill. Prefill is the slow path when weights are streamed, so this matters more here
than it does on a normal setup.

---

## 6. Tuning

The shipped 177B flags, and what each is worth:

| flag | value | why |
|---|---|---|
| `--moe-stream-cache` | `6` | 67 slots/layer, 6.07 GiB. **8 falls off the cliff to 3.85 tok/s**; 4 is slower (9.8); 2 fails to load. `-np 1` does *not* move the cliff. |
| `--moe-stream-io-threads` | `4` | Reads saturate at 4: 1/2/4/10 → 5.9 / 8.8 / 11.9 / 11.7. Before the handle fix, fewer was better — because they were all queued anyway. |
| `--poll` | `0` | +3%. See [3.7](#37-method-6-things-that-are-deliberately-off). |
| `-t` | *(default)* | Irrelevant. 1/2/3/4/6 all measure 10.4–10.7. With `-ngl 99` the only CPU graph op is the remap itself. |
| `-c` | `32768` | Free. See above. |
| `-np` | `1` | One slot. Does not move the VRAM cliff but costs nothing. |
| `-ub` | `256` | Prefill batch. |
| `-ctk`/`-ctv` | `q8_0` | KV quantisation. KV is small here (only 12 of 48 layers are full attention, 2 KV heads). |

For the 35B, the only knob that matters is `NCMOE` — see the table in
[3.2](#32-method-1-ncmoe-residency-the-35b-path).

---

## 7. Maintenance: rebuilding the hot file

The hot file is derived from the heat ranking, which tracks **your** usage. Rebuild it
occasionally — especially after your workload shifts (e.g. you move from chatting to coding):

```bat
python bench\mkhot.py kvcache\expert.heat models\flashnext\experts.hot 11 67
```

| argument | meaning |
|---|---|
| `kvcache\expert.heat` | the ranking the server has been accumulating |
| `models\flashnext\experts.hot` | output file (plus a `.idx` next to it) |
| `11` | size in GiB. 11 is the measured sweet spot; 14 gives +1% but takes 157 s to start and leaves the OS 2.6 GiB of page cache |
| `67` | experts per layer to **skip**, because the VRAM cache holds them anyway. Keep this equal to your slots/layer |

Takes about 3 minutes and needs 11 GiB of free disk. Stop the server first.

> ### ⚠ Do not benchmark on a hot file built from your benchmark prompts
>
> The server checkpoints the ranking **every ~256 tokens while it runs**. So if you build a
> hot file from `kvcache/expert.heat` and then benchmark on prompts you have already run
> through that server, the file has seen the test set and your number is self-confirming.
>
> This is not hypothetical — it happened here. A sweep pointed `LLAMA_MOE_STREAM_HEAT` at the
> training ranking while generating on the measurement prompts and silently rewrote it;
> **32% of the resulting hot file's contents came from the measured prompts**, inflating the
> result by +3% cold and +9% warm.
>
> Guards now in place: `bench/heatgen.py` writes `bench/expert_train.heat`, a path no
> benchmark touches, and **`bench/sweep.py` raises an error if any config sets
> `LLAMA_MOE_STREAM_HEAT` at all**. For published numbers, build the file from
> `bench/expert_train.heat`.

---

## 8. Measuring anything yourself

The tooling exists because this workload is unusually good at producing convincing wrong
numbers. Three traps, all handled:

1. **The OS page cache carries over between server restarts**, drifting every later config
   up 3–5% — bigger than most real effects. `bench/dropcache.py` evicts the standby list
   without needing admin (18.96 → 4.46 GiB, repeatable to 0.01 GiB) by committing and
   touching private pages until the OS hands them over. `sweep.py --cold` calls it.
2. **Routing is bit-exact deterministic**, so the `(L1 hit, miss/tok, cold)` triple
   identifies the same workload point in every run. `bench/cmp.py` compares two runs only at
   matched triples, removing warm-up position as a variable.
3. **A wrong-bytes bug runs faster, not slower.** Always `bench/qual.py`.

| script | what it does |
|---|---|
| `bench/qual.py` | **the quality gate.** 6 factual checks + a diff against a control run. Run this before believing any speed number |
| `bench/sweep.py` | json-driven config sweep: one server per config, tok/s + real disk traffic + memory. `--cold` to drop the cache first |
| `bench/chat.py` | the single-topic conversation case, which `sweep.py` deliberately is not |
| `bench/diskprobe.py` | MiB per token that **actually left the SSD**, from the OS performance counters. The debug line's latency buckets cannot tell a slow page-cache memcpy from a fast NVMe read; this can |
| `bench/qd.py` | shared vs per-thread file handle throughput — the measurement that found the main bug |
| `bench/mkhot.py` | builds the page-locked hot-expert file |
| `bench/heatgen.py` | builds a clean training ranking from a disjoint prompt set |
| `bench/ramslope.py`, `bench/hog.py` | tok/s as a function of available page cache |
| `bench/alloc.py` | exact LRU stack distances; optimal per-layer slot budget via concave envelope |
| `bench/policy.py`, `hier.py`, `predict.py` | offline cache-policy, two-level-residency and prefetch-predictability models over a recorded routing trace |
| `bench/gguf_map.py` | minimal GGUF reader: tensor name, type, shape, absolute file offset |
| `scratchpad/h2d.cu` | PCIe H2D ceiling, sync-per-copy vs batched |
| `scratchpad/pinmap.cu` | whether a writable file mapping can be page-locked and DMA'd from |

Example sweep:

```bash
cd bench
cat > cfg_mine.json <<'EOF'
[ {"name":"ctl", "args":"--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 4 --poll 0", "env":{}},
  {"name":"test","args":"--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 8 --poll 0", "env":{}} ]
EOF
python sweep.py --cold --cfg cfg_mine.json --warm 3 --meas 4
```

> **This box is noisy.** The inherited arm measured 6.95/7.36 in one batch and 7.07/5.85 in
> another; one warm control came in at 7.37 against 10.55 for the identical config. Always
> interleave your arms, run at least two of each, and distrust any single config's number.

Set `LLAMA_MOE_STREAM_DEBUG=1` for a line every 32 tokens on **stderr**:

```
moe stream: 32 tok | L1 hit 70.5% | miss 141.6/tok (92 cold) | remap 55.0 | stall 54.6
            | read 108.8 | up 63.3 ms/tok | src 307.6/40.2/76.9 f/m/s
```

`read` and `up` are summed over the I/O workers, so `read/stall` is the effective queue
depth. `src` buckets each slab read by latency (<150 µs / <600 µs / rest), which separates
hot-file and page-cache hits from real disk reads.

---

## 9. Environment variable reference

All default off unless stated. The ones in the launcher are marked **shipped**.

| variable | what it does |
|---|---|
| `LLAMA_MOE_STREAM_KEEP_MMAP=1` | **shipped.** Keep the loader's mmap so the 26.8 GiB PLE table stays lazy. Effectively mandatory |
| `LLAMA_MOE_STREAM_HOT=<path>` | **shipped.** The page-locked hot-expert file (needs `<path>` and `<path>.idx`) |
| `LLAMA_MOE_STREAM_HEAT=<path>` | **shipped.** Persisted expert ranking; refills device slots at startup, checkpointed every ~256 tokens |
| `LLAMA_MOE_STREAM_WARM_GB=<n>` | Additionally pull `n` GiB further down the ranking into the page cache at startup. Measured null for steady state (11.08 / 11.07 / 10.82 at 10 / 16 / 22 GiB vs 11.10 control) — a first-answer feature only |
| `LLAMA_MOE_STREAM_DEBUG=1` | Periodic stats line on stderr |
| `LLAMA_MOE_STREAM_ASYNC_UP=1` | Decoupled, batched uploader. Halves upload time, gains ~1%, has crashed once. Off |
| `LLAMA_MOE_STREAM_PINNED=0` | Disable pinned staging. Costs ~7% |
| `LLAMA_MOE_STREAM_RANDOM_ACCESS=0` | Disable `FILE_FLAG_RANDOM_ACCESS`. Costs ~30% |
| `LLAMA_MOE_STREAM_SPLIT_WEIGHTS=0` | Load an expert's 3 weights on one worker instead of three. Worse (9.6 vs 10.9) |
| `LLAMA_MOE_STREAM_ADMIT=lo:hi` | Page-cache admission band on the undecayed use count; slabs outside it are read unbuffered. Measured worse |
| `LLAMA_MOE_STREAM_TRIM=0` | Disable the post-load `EmptyWorkingSet`. Neutral either way |
| `LLAMA_MOE_STREAM_PRIO=0` | Disable raising I/O worker thread priority. Neutral either way |
| `LLAMA_MOE_STREAM_TRACE=<path>` | Dump every decode routing decision for offline analysis by `bench/policy.py` etc. |
| `LLAMA_MOE_STREAM_ONE_HANDLE=1` | **Diagnostic.** Reproduce the pre-fix shared-handle behaviour, for A/B on one binary |
| `LLAMA_MOE_STREAM_NOWAIT=1` | **Diagnostic.** Load no experts at all. Output is garbage; measures the pure compute floor |
| `LLAMA_MOE_STREAM_SERIAL_IO=1` | Restore the old seek+read under a global lock |
| `LLAMA_MOE_STREAM_MMAP_COPY=1` | Upload straight from the mmap. Measured 2x worse |

---

## 10. If you cloned this

This repo is the **work**, not the weights. It is a few MB. What it does not contain, and
where to get it:

| not in git | size | how to get it |
|---|---|---|
| `models/` | ~100 GB | `scripts/dl_full.sh` (177B) — Unsloth's `Qwen3.8-Flash-Next-GGUF`, UD-IQ3_XXS, 3 splits. 35B is `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` |
| `models/flashnext/experts.hot` | 11 GiB | **derived — you build it**, see [section 7](#7-maintenance-rebuilding-the-hot-file). Do not download someone else's; it is ranked for *their* usage |
| `src/lcpp/` | 1.3 GB | a llama.cpp checkout + `patches/`, below |
| `build-tools/` | 6.4 GB | portable CMake + Ninja + CUDA redist; see [section 11](#11-building-from-source) |
| `llama/` | 0.7 GB | optional prebuilt upstream binaries, only used by `run.bat` |
| `kvcache/expert.heat` | 50 KB | generated by the server as you use it |

Everything in the tree resolves paths from `bench/_root.py` (`$MOE_ROOT`, or the parent of
`bench/`), so you do not have to edit anything. Override individually with `$MOE_MODEL` and
`$MOE_SERVER` if your layout differs. Check what it resolved to:

```bash
python bench/_root.py
```

### Getting the modified llama.cpp

The C++ changes are carried as a patch rather than by vendoring a 1.3 GB clone of someone
else's project:

```bash
git clone https://github.com/ggml-org/llama.cpp.git src/lcpp
cd src/lcpp
git checkout 177375096650b78a2ee4b220ec9abf2a57b73ec9     # the base this was developed on
git apply ../../patches/0001-moe-stream-local-fixes.patch
```

That base commit is upstream master with the MoE expert-streaming PR
[#25294](https://github.com/ggml-org/llama.cpp/pull/25294) merged in. The patch is ~1,700
lines across six files:

| file | what changed |
|---|---|
| `src/llama-moe-stream.cpp` / `.h` | per-worker file handles, the page-locked hot file, the undecayed heat ranking, parallel coldest-first warming, the batched uploader, the diagnostics |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | four entry points exposed through the backend registry: `h2d_async`, `h2d_sync`, `host_lock`, `host_unlock` |
| `src/llama-model.cpp`, `llama-context.cpp`, `llama.cpp` | wiring |

Then build with `build-moestream.bat` (full) or `build-ms2.bat` (incremental).

### Smallest useful thing to try first

If you only want to see whether the main fix matters on your hardware, you do not need the
hot file or even a benchmark run — `bench/qd.py` answers it in 30 seconds with nothing but
a large file:

```bash
python bench/qd.py --path /path/to/any/big.gguf
```

If the "shared-handle" column is far below the "per-thread-handles" column, your platform
has the same problem described in [section 3.4](#34-method-3-one-file-handle-per-io-worker).

## 11. Building from source

Everything is portable — no installer, no admin rights.

```
build-tools/
  cmake-4.4.3-windows-x86_64/     portable CMake
  ninja.exe
  cuda/_root/                     CUDA toolkit unpacked from redist zips
src/lcpp/                         llama.cpp, branch `tryms`
                                  (master + MoE streaming PR #25294 + local fixes)
```

Full configure + build:

```bat
build-moestream.bat
```

Incremental rebuild of just the server (what you want while iterating):

```bat
build-ms2.bat
```

Requirements: Visual Studio 2022 Community (for `vcvars64.bat` and MSVC), and CUDA **12.8+**
— Blackwell `sm_120` is not supported by older toolkits. The build here uses CUDA 13.4 with
`-DCMAKE_CUDA_ARCHITECTURES=120`.

Local changes live in:

- `src/lcpp/src/llama-moe-stream.{h,cpp}` — the streaming layer: per-worker handles, the hot
  file, the heat ranking, the uploader, the diagnostics.
- `src/lcpp/ggml/src/ggml-cuda/ggml-cuda.cu` — four small additions exposed through the
  backend registry (`ggml-cuda` is a separately loaded module, so they cannot just be
  linked): `ggml_backend_cuda_h2d_async`, `_h2d_sync`, `_host_lock`, `_host_unlock`.

---

## 12. Troubleshooting

**Output is `!!!!!!!!` or other garbage.**
A wrong-bytes bug in the expert path. This runs *fast*, not slow. Delete / disable the hot
file (`LLAMA_MOE_STREAM_HOT`) and re-run `bench/qual.py`; if that fixes it, rebuild the hot
file with `bench/mkhot.py` — most likely the index and the model's weight registration order
have diverged.

**Speed suddenly collapsed to 3–5 tok/s.**
VRAM cliff. Something else took VRAM (Chrome, a game, a second server). Close it, or lower
`--moe-stream-cache`. `nvidia-smi` should show ~11.1 GB used by llama-server and ~1 GB free.

**Startup takes 2+ minutes.**
Normal if the hot file is cold — it has to read 11 GiB off the SSD and lock it. ~40 s when
the page cache is warm. If it is consistently slow, your hot file may be too big; 11 GiB is
the measured sweet spot.

**`could not page-lock N GiB, hot file disabled` in the log.**
You asked for more locked memory than the system will give. 14 GiB is the measured ceiling
on this box; drop to 11.

**`server exited rc=3221226505`.**
`STATUS_STACK_BUFFER_OVERRUN` — usually an out-of-VRAM during allocation. Lower
`--moe-stream-cache` or `-c`.

**Server won't die / port 8080 busy.**
`taskkill /F /IM llama-server.exe`. With an 11 GiB locked mapping, teardown can take 30+
seconds; `bench/sweep.py` waits up to 180 s for this reason.

**Everything is mysteriously 10–20% slower than yesterday.**
Page cache. Either warm it (just use it for a few minutes) or measure cold with
`sweep.py --cold`.

---

## 13. What did not work

Short version. The full list, with numbers, is in **`alreadyTried`** (679 lines) and
**`bench/FINDINGS.md`** (741 lines). Please read those before trying something — almost
every obvious idea has already been built and measured here.

| idea | result |
|---|---|
| **Speculative decoding (MTP) on the 177B** | Both the unsloth and official heads fail to load — the main GGUF has no `nextn` block |
| **MTP on the 35B FAST profile** | 66.1 → **64.2**. Speculation only helps while you are memory-bound; once compute-bound, drafting costs more than it saves. Dense 27B 1.45x, MoE Q4 1.25x, MoE IQ3 **0.97x** |
| **n-gram speculation** | 16% acceptance alone; stacked with MTP it makes things worse |
| **Lossless compression of expert slabs** | zlib-1 ratio **1.052** (bigger), zlib-6 0.990, lzma-1 0.995. IQ quants are already at entropy. A compressed host tier cannot exist |
| **Better L1 cache policy** | LRU 69.97% (matches the 70% measured live). ARC 68.5%, 2Q 66.9%, static-by-frequency 55.8%, hybrid pinning 64–69%. Belady 82.6% but unreachable. **LRU is optimal here** — routing is recency-driven, which is what a load-balanced router should look like |
| **Non-uniform per-layer slot allocation** | Exact stack distances + concave-envelope budget split: 68.45% → 69.28% hit, −2.6% misses. Not worth it |
| **Cross-layer expert prefetch** | 39% recall at K=10 with 5.9 wasted fetches per layer-token. The hits land mostly on experts that are already resident. Not viable |
| **An explicit host-RAM L2 arena** | Five variants, 8–21 GiB, pageable / zero-copy / pinned-staging / pinned-DMA. All lose. On 32 GB there is no spare RAM to build a tier out of — it is zero-sum with the page cache |
| **Page-cache warming for throughput** | Null at 10, 16 and 22 GiB. The cache converges to the same ~16 GiB of content whatever you seed it with |
| **More RAM** | Measured at **0.17 tok/s per GiB**. 64 GB is not the lever people assume |
| **The `moe-cache` fork** | 1.72 vs 6.70 here. It requires `cudaHostRegisterReadOnly`, which WDDM does not provide. See [3.5](#35-method-4-the-page-locked-hot-expert-file) for the workaround this project found instead |
| **REAP-pruned variants** | Fast, but world knowledge is damaged — a pruned build could not name Canberra. Rejected on quality |
| **`ik_llama.cpp`** | Documented Qwen3-MoE regression |
| **`-sm row` across two GPUs** | OOM at every split |
| **FFN-only offload via `-ot` on dense** | 1.7–3.1 vs 4.9 baseline. PCIe activation round-trips cost more than the bandwidth saved |

---

## 14. Hardware, and what to upgrade

```
RTX 5070 12 GB (compute 12.0, PCIe gen.max 3, x16 — confirmed gen3 x16 under load)
Ryzen 5 5600GT  (Cezanne APU)
32 GB DDR4-2400 (Kingston KF3600C18D4 — a 3600 kit, XMP/DOCP off)
ASRock B550M-C
Kingston SNV2S1000G NVMe  (measured 2.85 GB/s at QD4+, 0.93 GB/s at QD1/512 KiB)
Windows 10 IoT Enterprise LTSC 19044, driver 591.86
llama.cpp branch `tryms`, CUDA 13.4
```

Ranked by measured value per pound:

1. **Enable DOCP in BIOS.** The DIMMs are a DDR4-3600 kit running at 2400, and the remaining
   stall is host-DRAM bound. Free, and the largest single change left.
   (Reboot → Del → OC Tweaker → DRAM Profile → DOCP Profile 1; try 3200 if 3600 won't post.)
2. **A non-APU CPU.** The 5600GT is Cezanne, which is **PCIe Gen3 only** — that is the
   measured `pcie.link.gen.max = 3`, and it caps both the NVMe and the upload link that now
   bounds the expert stream. A Ryzen 5 5600 or 5600X (~$80–100 used) gives Gen4 on the
   CPU-attached M.2 and roughly doubles both ceilings.
3. **A card with more VRAM.** The L1 hit rate is 70% at 67 of 512 slots, and raising it cuts
   PCIe traffic, DRAM traffic and disk reads simultaneously. This is the only thing that
   lifts the 21 ms/token PCIe floor.
4. **More RAM** — last, and smaller than you think. 0.17 tok/s per GiB.

---

## Related work

- [Strata](https://github.com/Niko1221/Strata) — same idea at 125B; needs 64 GB DDR5
- [Colibrì](https://github.com/uv-genai/colibri) — 744B via SSD expert streaming, 0.05–1.23 tok/s
- [GenerelSchwerz/llama.cpp `moe-cache`](https://github.com/GenerelSchwerz/llama.cpp) — cold experts in pinned host memory; needs a non-WDDM platform
- [SpecExec](https://arxiv.org/abs/2406.02532) · [TEAL](https://arxiv.org/abs/2408.14690) · [Local Routing Consistency](https://arxiv.org/abs/2505.16056)

Raw data: [`alreadyTried`](alreadyTried) · [`bench/FINDINGS.md`](bench/FINDINGS.md) ·
[`RESULTS.md`](RESULTS.md) · [`bench/results.jsonl`](bench/results.jsonl)
