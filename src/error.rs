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

    /// Soft OOM guard tripped *before* calling the driver. Turning an 8 GB card's
    /// hard OOM into a recoverable error keeps the WDDM display path from being starved.
    #[error("VRAM guard: requested {requested} B, free {free} B, reserve {reserve} B")]
    VramGuard { requested: usize, free: usize, reserve: usize },

    /// Host/device element-count mismatch on copy. Checked on the host because
    /// the driver would happily copy a short buffer and leave stale VRAM behind.
    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, EngineError>;