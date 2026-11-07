//! Device-wide VRAM accounting and the Phase 1 soft OOM guard.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::error::{EngineError, Result};

/// CONTEXT.md §4 hard cap (context + weights + KV cache + scratchpad), in MiB to match `nvidia-smi`.
pub const PHASE1_VRAM_CAP_BYTES: usize = 4_500 * 1024 * 1024;

pub struct VramTracker {
    ctx: Arc<CudaContext>,
    cap_bytes: usize,
}

impl VramTracker {
    pub fn new(ctx: Arc<CudaContext>, cap_bytes: usize) -> Self {
        Self { ctx, cap_bytes }
    }

    /// Returns `(allocated, free, total)` in bytes.
    /// `allocated` is device-wide (`total - free`): it includes this process's CUDA
    /// context and, on WDDM, any other process on the dGPU. Tests must compare deltas,
    /// never absolutes, and must `synchronize()` after a drop because cudarc frees via
    /// `cuMemFreeAsync`, which only returns memory to the driver at a sync point.
    pub fn query(&self) -> Result<(usize, usize, usize)> {
        let (free, total) = self.ctx.mem_get_info()?;
        Ok((total - free, free, total))
    }

    /// Checked before every engine allocation. Counting device-wide usage means the
    /// guard can only trip early (other processes, unreleased pool memory), never late.
    pub fn guard(&self, requested: usize) -> Result<()> {
        let (used, _, _) = self.query()?;
        if used.saturating_add(requested) > self.cap_bytes {
            return Err(EngineError::VramGuard { requested, used, cap: self.cap_bytes });
        }
        Ok(())
    }
}
