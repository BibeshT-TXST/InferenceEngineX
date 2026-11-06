# InferenceEngineX

InferenceEngineX is planned to be a custom inference engine, built for learning and testing inference capabilities locally. The GPU being used is NVIDIA RTX 5070 laptop GPU and the model architecture is Qwen.

### GPU Specifications
- Name: NVIDIA RTX 5070 laptop GPU
- Architecture: Blackwell, GB206 die
- Memory: 8GB GDDR7, across a 128-bit bus
- Memory Bandwidth: 384 GB/s

### Model Specifications
- Quen2.5-1.5B(1.5 Billion Parameters)
- ~3GB of memory space in GPU

## Engine Architecture
- Tokenizer & Detokenizer
- Zero-Copy Weight Loader
- CUDA Runtime & Kernel Layer(NVIDIA CUDA Rust)
- The Model Execution Graph (Qwen Architecture)
- KV Cache & Memory Manager
- Logits Sampler

## Metrics to Document
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- Peak VRAM Footprint & KV Cache Capacity

## Project Plan
- Phase 1: The Naive Baseline
- Phase 2: Kernel Fusion & Memory Bandwidth (The TensorRT-LLM Mindset)
- Phase 3: Continuous Batching & PagedAttention (The vLLM Mindset)
- Phase 4: Cloud Migration (Scaling Up on L40S / H100 / B200)

## Repository Rules
- Every Phase must be labeled as a milestone
- Every Milestones Must have n Issues to do a specific task in 1 branch
- Every Issue and PR must have a standard template
- Atomic Git commits with appropriate In text documentation in every branch
- Each mile stone completion must require tests, Metrics form those texts, overall development cycle in that phase is a separate readme.
- Each mile stone completion must also have a video explaining what, why , how and moving forward related decisions 
- Every use of Research paper needs to be documented in a separate file Citations.md and must reference were, how and what was used clearly.
- No outside contributions, only feedback

---
**Built with care to serve people**
