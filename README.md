# LLM-Inference-Benchmarking

Benchmarking throughput and latency of LLM serving under batching, memory, and quantization constraints on a single NVIDIA T4 (16 GB, compute capability 7.5).

Stack: Python · PyTorch · vLLM · Qwen2.5-3B-Instruct · pandas · matplotlib

# Motivation

Serving an LLM is a resource-allocation problem, not a modeling one. A single GPU must hold the model weights, the KV cache, and activation scratch space simultaneously — and every concurrent request competes for the same memory. This project measures how batching, sequence length, and memory configuration trade off against each other, and where the practical limits sit on consumer-grade hardware.

# Setup
Component	Value
GPU	NVIDIA Tesla T4 (16 GB, SM 7.5)
Model	Qwen2.5-3B-Instruct, fp16
Serving engine	vLLM (V1 engine)
Max model length	2048 tokens
gpu_memory_utilization	0.85
Attention backend	Triton (FlashAttention 2 requires SM ≥ 8.0)

# Measured memory breakdown at startup:

Model weights: 5.79 GiB
KV cache: 5.08 GiB (148,080 tokens → 72× max concurrency at full 2048-token context)
CUDA graphs: 0.54 GiB

# Method

All requests use an identical prompt with temperature=0.0 and ignore_eos=True, forcing every request to generate exactly max_tokens tokens deterministically. This isolates batch size as the only independent variable — without it, variable-length outputs would confound throughput comparisons across runs.

Each configuration is timed end to end; throughput is total output tokens divided by wall-clock time. Failed configurations (OOM) are recorded as rows rather than aborting the sweep, so capacity limits appear in the data.

# Results: throughput vs. batch size

fp16, gpu_memory_utilization=0.85, 128 output tokens per request.

Batch size	Throughput (tok/s)	Wall time (s)	Requests/s	Per-request rate (tok/s)
1	28.0	4.57	0.22	28.0
2	69.8	3.67	0.55	34.9
4	136.9	3.74	1.07	34.2
8	264.3	3.87	2.06	33.0
16	490.5	4.17	3.83	30.7
32	793.0	5.17	6.20	24.8
64	1366.8	5.99	10.68	21.4

 Show Image

# Findings

Batching is close to free performance in this range. Increasing batch size 64× yielded a 48.8× throughput gain (28.0 → 1366.8 tok/s) at a 1.31× cost in wall-clock time (4.57s → 5.99s). For any workload that is throughput-bound, this is a decisive win.

Scaling efficiency degrades gradually rather than cliff-edging. Doubling batch size roughly doubles throughput through batch 8 (2.49×, 1.96×, 1.93×), then softens (1.86×, 1.62×, 1.72×) as the GPU approaches compute saturation. No hard ceiling was reached at batch 64.

Individual users pay a modest cost. Per-request generation falls from 28.0 tok/s at batch 1 to 21.4 tok/s at batch 64 — roughly 24% slower per user, while serving 64× more of them concurrently.

Anomaly — batch 1 warmup. Batch 1 shows higher wall time (4.57s) than batch 2 (3.67s), attributable to first-run warmup: the log confirms a Triton kernel JIT compilation during the first inference pass. Subsequent runs should discard a warmup call before measurement.

# Hardware constraints encountered
FlashAttention 2 unavailable — requires compute capability ≥ 8.0; the T4 is 7.5. vLLM fell back to the Triton attention backend, which is expected to cost throughput relative to Ampere-class hardware.
FlashInfer top-p/top-k sampling unavailable for the same reason; fell back to the default sampler.
max_autotune_gemm disabled — insufficient SMs on the T4.

These are worth stating explicitly: absolute numbers here are not comparable to A100/H100 benchmarks, but the shape of the tradeoff curves is.

# Known limitations
Synthetic uniform load. All requests use an identical prompt of identical length. This flatters vLLM's prefix caching and produces perfectly uniform batches — real traffic has varied prompt lengths and staggered arrival times.
Wall-clock time, not per-request latency. The reported latency column measures time until all requests in the batch complete, not the wait experienced by an individual user. These coincide only when requests start and finish together.
VRAM not captured. vLLM runs the model in a separate EngineCore process, so torch.cuda.max_memory_allocated() called from the parent process reports zero. Measuring actual usage requires querying nvidia-smi directly.

# Planned work
 Extend the sweep to batch sizes 128 / 256 / 512 / 1024 to locate the throughput knee and the OOM boundary
 Sweep max_tokens (128 / 512 / 1024) to quantify KV cache pressure from sequence length
 Raise gpu_memory_utilization to 0.95 and measure the change in max concurrency
 Benchmark an AWQ int4-quantized variant: throughput gain vs. output quality cost
 Add a mixed-length prompt workload to compare synthetic vs. realistic traffic patterns
 Fix VRAM measurement via nvidia-smi sampling

# Reproducing
bash
pip install vllm

Open notebooks/inference_benchmark.ipynb on any CUDA GPU with ≥ 16 GB VRAM and run all cells. Raw results are written to results/ as CSV; figures to figures/.

# Repository layout
.
├── README.md
├── notebooks/
│   └── inference_benchmark.ipynb
├── results/
│   └── results_batch_sweep_fp16.csv
└── figures/
    └── throughput_latency.png
