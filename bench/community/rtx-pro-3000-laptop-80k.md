# Community report: RTX PRO 3000 Blackwell Laptop GPU, 80K context

Contributed by [@ddonofrio](https://github.com/ddonofrio), November 4, 2026.

The full, unpruned Qwen3.8-Flash-Next 177B model is usable for interactive work on
this 12 GB laptop GPU with an 80,000-token context allocation. The owner reports
that follow-up requests feel responsive once the first long prompt has been
processed. This report preserves the working configuration and the observations
behind that experience.

These are observations from everyday use, not a controlled benchmark or a new
recommended default. In particular, **80K is the configured capacity**; the measured
follow-up request contained **31,767 input tokens**, mostly reused from the KV cache.
The results should not be ranked against the main README's controlled RTX 5070 tests.

## Hardware and software

| Component | Configuration |
|---|---|
| CPU | Intel Core Ultra 7 265HX; 20 physical cores, 20 logical processors |
| RAM | 32 GiB DDR5, 2 x 16 GiB; configured at 6400 MT/s; module part `M435R2GA3PB1-CCPRC` |
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU; 12,227 MiB reported VRAM (12 GB class) |
| Model storage | `NVMe PC811 SED SK hynix 1024GB`, 1 TB NVMe SSD; NTFS system drive |
| OS | Windows 11 Pro, version 10.0.26300, build 26300 |
| NVIDIA driver | 596.53 |
| Toolchain | CUDA 13.2.86, MSVC 19.51.36252.0, CMake 4.4.3, Ninja 1.13.2 |
| Build | Release, CUDA `sm_120`, `GGML_NATIVE=ON`; `GGML_CUDA_FA_ALL_QUANTS=OFF` |

A snapshot during use showed 11,371 MiB GPU memory in use and 1.64 GiB available
system RAM. These are point-in-time readings, not peak measurements or guaranteed
memory requirements. GPU power limits, power draw and clocks were not recorded.

## Model and source

- Model: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  `UD-IQ3_XXS`, full 177B model, three shards totaling 81,961,823,936 bytes
  (about 76.3 GiB). No pruning or draft model.
- Source recipe: llama.cpp base
  [`6fdd0ac`](https://github.com/ggml-org/llama.cpp/commit/6fdd0ac8907fd973a42b876357823ad2124cd8ed)
  plus this [pinned combined compatibility patch](https://github.com/ddonofrio/hashyy/blob/81853942b99b0ef1bc04add78e02fea62900cfb4/patches/0001-moe-stream-local-fixes.patch).
  It includes MoE streaming from PR #25294, Qwen3.8 support from PR #27742, and the
  local Windows adaptations. The base and patch are a pair; this is different from
  the older source recipe in the main README.
- The [pinned Windows build script](https://github.com/ddonofrio/hashyy/blob/81853942b99b0ef1bc04add78e02fea62900cfb4/build-moestream.bat)
  builds `llama-server` and `llama-bench`. Model download checksums, patch checksum,
  build flags and the identities of the binaries used are recorded in the
  [configuration and observation snapshot](rtx-pro-3000-laptop-80k.json).

## Exact working launch profile

Save this as a `.bat` file at the repository root, with the model and built server
in the paths shown. It is the executable content of the local launcher; relative
paths make it independent of the Windows account name.

```bat
@echo off
set "CUDA_PATH=%~dp0build-tools\cuda\_root"
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set CUDA_VISIBLE_DEVICES=0
set LLAMA_MOE_STREAM_KEEP_MMAP=1
set LLAMA_MOE_STREAM_HOT=%~dp0models\flashnext\experts.hot
set LLAMA_MOE_STREAM_HEAT=%~dp0kvcache\expert.heat
mkdir "%~dp0kvcache" 2>nul
"%~dp0src\lcpp\build-ms\bin\llama-server.exe" ^
  -m "%~dp0models\flashnext\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf" ^
  --moe-stream --moe-stream-cache 5 --moe-stream-io-threads 4 ^
  --poll 0 ^
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 256 -c 80000 -np 1 ^
  --slot-save-path "%~dp0kvcache" --jinja ^
  --host 127.0.0.1 --port 8080
```

| Setting | Value and purpose |
|---|---|
| Context and concurrency | `-c 80000 -np 1`; server reports a rounded slot allocation of 80,128 tokens. Input and generated output share this budget. |
| Expert VRAM budget | `--moe-stream-cache 5`; resolves to 56 experts per layer across 48 layers, 2,688 slots and about 4.95 GiB of expert tensors. Other tensors, KV and buffers also use VRAM. |
| Attention and KV | Flash attention on; K and V both `q8_0`; physical microbatch `-ub 256`. |
| Expert I/O | Four streaming I/O workers; `--poll 0`. |
| Host memory | `LLAMA_MOE_STREAM_KEEP_MMAP=1` and an approximately 11 GiB page-locked hot-expert file. |
| Persistent ranking | `LLAMA_MOE_STREAM_HEAT` points to the owner's accumulated usage. It is not a disjoint benchmark training set. |
| Chat behavior | Jinja enabled; thinking enabled by the template default; `--reasoning-preserve` not explicitly enabled. |

The 5 GiB budget was selected to leave more GPU memory for the larger context.
There is no controlled 5-versus-6 GiB comparison here, so this report does not assign
a speedup to that change alone.

### Hot file and cache state

The hot file was built from the owner's existing heat ranking with:

```bat
python bench\mkhot.py kvcache\expert.heat models\flashnext\experts.hot 11 56
```

Here `11` is the host-file budget in GiB and `56` skips each layer's top-ranked
experts already assigned to the device cache. The resulting file contains 5,941
experts / 17,823 tensor slabs and occupies 11,810,902,016 bytes. Its index uses the
`moe_hot 2` format. Preserve the existing `.hot` and `.hot.idx` together; rebuilding
from an evolving heat file changes the experiment. Stop the server before rebuilding.

The private heat ranking, hot files and prompts are not distributed. The recipe
reproduces the settings, but a different usage history will produce a different
expert selection and may produce different performance.

Repeated requests can reuse a matching prompt prefix while it remains in the
server's slot. `--slot-save-path` enables explicit slot save/restore operations;
setting the directory alone does not automatically persist conversations across
server restarts.

## Observed performance

| Phase | Observation | Measurement scope |
|---|---|---|
| First long prompt after starting the server: model described as "cold" by the operator | Approximately **32 tokens/s prompt processing** | Operator-reported average while prefill was running; no final request timing was retained. This is not a generation rate. |
| Storage activity during that initial prefill | **2.83-3.36 GB/s physical reads**, roughly **8.6-10.0k reads/s** | Three Windows physical-disk counter samples on the model's system drive. GB/s is decimal; this includes other activity on that drive. |
| Later follow-up, with the server kept running | **31,767 input tokens: 30,618 reused, 1,149 newly processed** | Live `/slots` counters; **96.4%** of the input reused a matching prefix. No complete prefill duration was retained. |
| Generation during that follow-up | Approximately **14.4 tokens/s** | Decoded count rose from 1,208 to 1,352 over approximately 10 seconds. This is a short live window, not a full-request mean. |

The initial request put substantial pressure on storage while loading expert
weights for a model much larger than RAM or VRAM. Disk-backed expert loading was
the apparent bottleneck during that prefill, consistent with the high physical
read rates. The observations do not isolate storage from PCIe transfers, CPU work
or GPU computation, and they do not show that all disk reads disappear after the
first request.

"Cold" here means the first long request after server startup, **not a verified
cold OS cache**: the standby cache was not flushed, the learned heat ranking and
hot file were enabled, and one prefill snapshot already showed 6,144 cached prefix
tokens. A repeat with empty caches could behave differently.

The follow-up completed and the slot returned to idle. Final generation duration,
time to first token and complete API usage were not retained, so no full-request
decode average or exact generated-token total is claimed. The earlier local
32K-context / 6 GiB-cache logs are a different profile and are excluded.

### How to interpret these observations

The [JSON snapshot](rtx-pro-3000-laptop-80k.json) preserves the configuration and
counter values transcribed from the setup session. It is not a raw server log.
Times are local to Europe/Madrid; polling timestamps have whole-second precision
and the API and Windows counter reads were sequential.

The workload was private interactive use with a learned cache, no fixed seed and
no repeated trial protocol. The observed request used temperature 1.0, top-k 20,
top-p approximately 0.95, min-p approximately 0.05, and a 32,000-token output limit.
No quality gate was run for this report. It demonstrates a working configuration
on another GPU, not output-quality equivalence or a performance advantage over
other systems. A comparable benchmark would need the repository's public workload,
disjoint cache training, controlled initial cache state and complete timing logs.
