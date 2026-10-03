@echo off
REM FULL UNPRUNED Qwen3.8-Flash-Next (177B) - nothing removed, nothing dumbed down.
REM Single RTX 5070 12GB + 32GB DDR4-2400, 76.3 GiB of weights.
REM
REM Uses a locally built llama.cpp: current master (Flash-Next support)
REM + MoE expert-streaming PR #25294 merged in + local fixes.
REM
REM MEASURED ON THIS BOX (bench/sweep.py and bench/chat.py, median of 3, quality-gated
REM by bench/qual.py which diffs the output against a no-streaming control):
REM   11.4-11.6    eight unrelated prompts in rotation - the pessimistic case, every
REM                prompt drags a different expert set through the cache
REM   9.1-13.4     a fresh conversation, turn by turn (median 10.3); the first two turns
REM                are the ones the hot file helps most, 6.0/7.9 -> 12.1/13.4
REM   5.9-7.1      as inherited, same binary, same cold start
REM   The hot file is built from prompts DISJOINT from the ones these were measured on.
REM
REM Context is 32768. The model supports 262144; 32k measured exactly as fast as 4k
REM (11.35 vs 11.36), 65536 drops to 7.9 because the KV pushes the expert cache off its
REM VRAM cliff. Raise it only if you also drop --moe-stream-cache, and re-measure.
REM
REM Why these flags (all measured - see bench/FINDINGS.md and alreadyTried):
REM   --moe-stream-cache 6    67 expert slots per layer = 6.07 GiB of VRAM. 8 falls off the
REM                           cliff (3.85 tok/s - the driver starts spilling VRAM over PCIe),
REM                           4 is slower (9.8). -np 1 does NOT move the cliff.
REM   --moe-stream-io-threads 4
REM                           reads saturate at 4 workers: 1/2/4/10 -> 5.9/8.8/11.9/11.7.
REM                           Before each worker got its own file handle, fewer was better
REM                           because they were all queued behind one file object.
REM   --poll 0                ggml's threadpool busy-waits by default and the expert remap is
REM                           a CPU op, so the other compute threads spin while thread 0 waits
REM                           on I/O. Worth about +3%. -t does not matter at all here.
REM   LLAMA_MOE_STREAM_KEEP_MMAP=1
REM                           the PR force-disables mmap when streaming; keeping it leaves the
REM                           26.8 GiB per-layer embedding table lazy instead of resident.
REM   LLAMA_MOE_STREAM_HOT    the page-locked hot-expert file. This is the big one after the
REM                           handle fix: those slabs are DMA'd to the GPU in place, so they
REM                           cost one pass over DRAM instead of three, and being locked they
REM                           can never fault to disk. Worth +0.9 tok/s steady and roughly 2x
REM                           on the first two turns of a conversation.
REM                           REBUILD IT when your own usage has shifted - it is ranked from
REM                           the heat file, so it gets better the more you use the model:
REM                             python bench\mkhot.py kvcache\expert.heat ^
REM                                    models\flashnext\experts.hot 11 67
REM                           (11 = GiB, 67 = slots the VRAM cache already holds, so the file
REM                            is the tier BEHIND it rather than a second copy of it)
REM   LLAMA_MOE_STREAM_HEAT   remembers which experts you actually use and refills all 3216
REM                           device slots from that ranking at startup. Cold-start only.
REM
REM Tool calling works: the chat template carries tools/tool_calls and --jinja is on, so
REM POST /v1/chat/completions with a "tools" array behaves like the OpenAI API.
set CUDA_VISIBLE_DEVICES=0
set LLAMA_MOE_STREAM_KEEP_MMAP=1
set LLAMA_MOE_STREAM_HOT=%~dp0models\flashnext\experts.hot
set LLAMA_MOE_STREAM_HEAT=%~dp0kvcache\expert.heat
mkdir "%~dp0kvcache" 2>nul
"%~dp0src\lcpp\build-ms\bin\llama-server.exe" ^
  -m "%~dp0models\flashnext\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf" ^
  --moe-stream --moe-stream-cache 6 --moe-stream-io-threads 4 ^
  --poll 0 ^
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 256 -c 32768 -np 1 ^
  --slot-save-path "%~dp0kvcache" --jinja ^
  --host 127.0.0.1 --port 8080
