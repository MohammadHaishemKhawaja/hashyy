# Raw run logs (sessions 1-2)

The llama-server output behind the tables in `../RESULTS.md` and the early sections of
`../alreadyTried`. Kept because every claim in this project is supposed to be checkable,
not because they are pleasant to read.

Naming is roughly `log_<what-was-being-tested>.txt`. `log_ms_*` are the MoE-streaming
sweeps, `log_t_*` and `log_q*` are the 35B `-ncmoe` / quantisation sweeps, `log_reap*` are
the pruned variants that were rejected on quality.

Session-3 logs are not here: `bench/sweep.py` writes one per config into `bench/logs/`,
which is gitignored because it is 2.3 MB of mostly-superseded noise. The session-3 results
that matter are in `bench/results.jsonl` and `bench/FINDINGS.md`.
