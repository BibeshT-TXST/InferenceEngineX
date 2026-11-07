//! Process-wide GPU handle: one device, one stream, one cuBLAS handle.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};

use crate::cuda::vram::VramTracker;
use crate::error::Result;

/// Phase 1 runs every copy, kernel and GEMM on ONE stream. That single-stream
/// invariant is what makes disabling cudarc's event tracking sound; adding a
/// second stream (Phase 2+) means re-enabling it or syncing streams by hand.
pub struct GpuContext {
    // No separate `Arc<CudaContext>` field: the stream already keeps the context alive
    // (`stream.context()`), and a second handle would just be a second source of truth.
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    vram: VramTracker,
}

impl GpuContext {
    /// `cap_bytes` sets the VRAM soft guard: the engine passes `PHASE1_VRAM_CAP_BYTES`,
    /// tests pass a tiny cap to trip the guard without allocating gigabytes.
    pub fn new(cap_bytes: usize) -> Result<Self> {
        let ctx = CudaContext::new(0)?;

        // SAFETY: Must run before `new_stream()` and before any allocation, so no
        // `CudaSlice` ever records events. Sound because every operation in this engine
        // is enqueued on the single stream created below: stream order alone serializes
        // alloc -> write -> read -> free, so the cross-stream events are redundant.
        unsafe { ctx.disable_event_tracking() };

        // Non-blocking: this stream does not implicitly sync with the legacy NULL stream,
        // so driver work on stream 0 can never stall the decode loop.
        let stream = ctx.new_stream()?;
        // cuBLAS bound to the same stream keeps GEMMs ordered with our own kernels
        // without extra events, which preserves the single-stream invariant.
        let blas = CudaBlas::new(stream.clone())?;
        let vram = VramTracker::new(ctx, cap_bytes);

        Ok(Self { stream, blas, vram })
    }

    /// Blocks the host until all queued work finishes. It is also the point where
    /// `cuMemFreeAsync`'d buffers return to the driver, so VRAM readings after a
    /// `drop` are only meaningful after this call.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    pub fn vram(&self) -> &VramTracker {
        &self.vram
    }
}
