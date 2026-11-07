//! Engine-wide error type. Library code returns typed errors; only tests and bins use `anyhow`.

use cudarc::cublas::result::CublasError;
use cudarc::driver::DriverError;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Any CUDA Driver API failure (context creation, alloc, memcpy, sync).
    /// Kept as the raw `CUresult` so the exact driver code reaches the log.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("cuBLAS error: {0}")]
    Blas(#[from] CublasError),

    /// Soft OOM guard tripped *before* calling the driver, so blowing the Phase 1
    /// budget is a typed error instead of a driver OOM halfway through weight loading.
    #[error("VRAM guard: requested {requested} B with {used} B in use exceeds cap {cap} B")]
    VramGuard {
        requested: usize,
        used: usize,
        cap: usize,
    },

    /// Host/device element-count mismatch on copy. Checked on the host because
    /// the driver would happily copy a short buffer and leave stale VRAM behind.
    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    /// Zero-sized or `usize`-overflowing shape, rejected before the driver sees it:
    /// `cuMemAlloc(0)` fails with an opaque INVALID_VALUE, and a wrapped size would under-allocate.
    #[error("invalid tensor shape {shape:?}")]
    InvalidShape { shape: [usize; 4] },
}

pub type Result<T> = std::result::Result<T, EngineError>;
