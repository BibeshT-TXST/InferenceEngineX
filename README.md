# InferenceEngineX

InferenceEngineX is planned to be a custom inference engine, built for learning and testing inference capabilities locally. The GPU being used is NVIDIA RTX 5070 laptop GPU and the model architecture is Qwen.

### GPU Specifications
- Name: NVIDIA RTX 5070 laptop GPU
- Architecture: Blackwell, GB206 die
- Memory: 8GB GDDR7, across a 128-bit bus
- Memory Bandwidth: 384 GB/s

### Model Specifications
- Qwen2.5-1.5B (1.5 Billion Parameters)
- ~3GB of memory space in GPU
- Target Draft Model: Qwen2.5-0.5B (~1GB VRAM in BF16, used for Speculative Decoding)
- Precision Formats: BF16 (Baseline), FP8 E4M3 (Blackwell 5th-Gen Tensor Core native), INT4 / AWQ (Weight-Only Quantization)


## Engine Architecture
- Tokenizer & Detokenizer
- Zero-Copy Weight Loader
- CUDA Runtime & Kernel Layer(NVIDIA CUDA Rust)
- The Model Execution Graph (Qwen Architecture)
- KV Cache & Memory Manager
- Logits Sampler
- High-Concurrency Serving & Router Layer (Async gRPC/HTTP SSE server, Token-Bucket Admission Control, and Prefix-Aware Load Balancing)
- Global Prefix & Distributed KV Cache (RadixAttention tree for cross-request KV reuse and Host-RAM/Disk KV offloading)
- Speculative Decoding & Test-Time Compute Engine (Draft-Target verification loop and parallel tree/beam rollout scheduler for RL/reasoning workloads)
- Full-Stack Flight Recorder & Profiler (Deterministic request trace logger, replay harness, and NVTX/CUPTI hardware timeline instrumentation)


## Metrics to Document
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- Peak VRAM Footprint & KV Cache Capacity
- Tail Latency Under Load (p50, p95, p99 TTFT and ITL across concurrent client streams)
- Goodput & SLA Attainment (Requests/sec meeting strict TTFT < 200ms and TPOT < 30ms constraints)
- Hardware Utilization Efficiency (Model FLOPs Utilization [MFU] during Prefill and Memory Bandwidth Utilization [MBU % of 384 GB/s] during Decode)
- Speculative Acceptance Rate & Speedup (Average accepted draft tokens per step vs. target model verification overhead)
- Numerical Fidelity & Drift (KL-Divergence and Perplexity delta across BF16, FP8, and INT4 kernels)


## Project Plan
- Phase 1: The Naive Baseline
- Phase 2: Kernel Fusion & Memory Bandwidth (The TensorRT-LLM Mindset)
- Phase 3: Continuous Batching & PagedAttention (The vLLM Mindset)
- Phase 3.5: Algorithmic Acceleration & Test-Time Compute (Speculative Decoding with Qwen2.5-0.5B, Prefix Caching / RadixAttention, and Best-of-N / Tree Rollouts for RL and reasoning scaling)
- Phase 4: Cloud Migration (Scaling Up on L40S / H100 / B200)
- Phase 5: Full-Stack Observability, Deterministic Replay & Production CI/CD (Custom NVTX/CUPTI trace-and-replay debugger, automated regression benchmarking harnesses, and containerized GPU CI/CD pipelines)


## Repository Rules
- Every Phase must be labeled as a milestone
- Every Milestones Must have n Issues to do a specific task in 1 branch
- Every Issue and PR must have a standard template
- Atomic Git commits with appropriate In text documentation in every branch
- Each mile stone completion must require tests, Metrics form those texts, overall development cycle in that phase is a separate readme.
- Each mile stone completion must also have a video explaining what, why , how and moving forward related decisions 
- Every use of Research paper needs to be documented in a separate file Citations.md and must reference were, how and what was used clearly.
- No outside contributions, only feedback
- Every PR modifying GPU kernels or precision numerics must pass an automated logit-parity and KL-divergence gate against the reference baseline before merging.
- Every Milestone must archive a hardware trace artifact (Nsight Systems `.nsys-rep` or Perfetto trace JSON) alongside a reproducible traffic replay log.
- Every Milestone must publish a versioned, reproducible Docker container image with an automated benchmark regression report via CI/CD.


---
**Built with care to serve people**
