Performance evidence for LLVM PR #240760

Original PR commit: ed3ec9f6e94d6b21487b890824fc5f9bb168baf7
Optimized commit: ffb57dc24a0868fc4bb53a179e1e506f5b37a3c3
Optimized tree: 7a514a41d171f551fd739adebeacee11f41aa6cb

Build both commits with the same Release, assertions-enabled configuration and
NVPTX enabled. Keep optimize.py, run.py and bench.cpp in the same directory.
On NVIDIA Thor sm_110 with CUDA 13.2-compatible headers and ptxas:

python3 optimize.py --original /path/to/original/llc \
  --candidate /path/to/optimized/llc --out /path/to/results

See evidence/optimized/summary.json for the method, final results and limits.
The first-* and repeat-* JSON files retain raw randomized paired timings.
The original/candidate PTX and SASS and generated compare.cpp are retained.
Correctness summaries and the compile matrix are also included. Absolute
paths in raw evidence refer to the author's measurement environment.

These are repeated out-of-line monotonic i32 CAS measurements, without
contention; timed comparisons succeed. Each thread owns its word. Native
metadata controls only run on global/shared memory. Clocks were not locked.
Do not interpret these results as real-application slowdowns or other-GPU
measurements. Failed comparisons are covered by separate correctness checks.

This evidence branch is separate from the upstream source patch.
Assisted-by: OpenAI Codex.
