//! Owned, shape-tagged device memory.

use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::cuda::context::GpuContext;
use crate::error::{EngineError, Result};

/// Raw owned VRAM. `CudaSlice` already frees on drop (`cuMemFreeAsync` on its own
/// stream), so a wrapper struct would add nothing; shape logic lives in `DeviceTensor`.
pub type DeviceBuffer<T> = CudaSlice<T>;

/// Row-major tensor of up to 4 dims (unused leading dims = 1).
/// Byte length and device pointer are derived, never stored: a cached `CUdeviceptr`
/// could outlive the `SyncOnDrop` guard cudarc ties to it (INVARIANTS §1.2).
pub struct DeviceTensor<T> {
    buf: DeviceBuffer<T>,
    shape: [usize; 4],
    strides: [usize; 4],
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceTensor<T> {
    /// The engine's only allocation path, so the VRAM guard sees every byte and
    /// Branch 7 can prove `forward()` never reaches it.
    pub fn zeros(gpu: &GpuContext, shape: [usize; 4]) -> Result<Self> {
        let numel = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .filter(|&n| n > 0)
            .ok_or(EngineError::InvalidShape { shape })?;
        let bytes = numel
            .checked_mul(size_of::<T>())
            .ok_or(EngineError::InvalidShape { shape })?;

        gpu.vram().guard(bytes)?;
        let buf = gpu.stream().alloc_zeros::<T>(numel)?;
        // Suffix products cannot overflow: each is <= numel, which was checked above.
        let strides = [
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1,
        ];
        Ok(Self {
            buf,
            shape,
            strides,
        })
    }

    /// Enqueues HtoD on the tensor's stream and returns.
    // shortcut: from pageable `&[T]` the driver stages through its own pinned buffer, so this
    // never overlaps compute; upgrade to `PinnedHostSlice` if HtoD overlap ever matters.
    pub fn copy_from_host_async(&mut self, src: &[T]) -> Result<()> {
        self.check_len(src.len())?;
        // Clone the Arc first: `stream()` borrows `buf`, which `memcpy_htod` needs mutably.
        let stream = self.buf.stream().clone();
        Ok(stream.memcpy_htod(src, &mut self.buf)?)
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        self.copy_from_host_async(src)?;
        Ok(self.buf.stream().synchronize()?)
    }

    /// Enqueues DtoH. Into pageable memory the driver completes the copy before
    /// returning, which is why a plain `&mut [T]` borrow is sound here.
    pub fn copy_to_host_async(&self, dst: &mut [T]) -> Result<()> {
        self.check_len(dst.len())?;
        Ok(self.buf.stream().memcpy_dtoh(&self.buf, dst)?)
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        self.copy_to_host_async(dst)?;
        Ok(self.buf.stream().synchronize()?)
    }

    /// Exact match, not cudarc's `>=` assert: a short host slice would otherwise leave
    /// stale VRAM in the tail, and a mismatch should be an error, not a panic.
    fn check_len(&self, got: usize) -> Result<()> {
        let expected = self.buf.len();
        if got != expected {
            return Err(EngineError::ShapeMismatch { expected, got });
        }
        Ok(())
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn strides(&self) -> [usize; 4] {
        self.strides
    }
    pub fn numel(&self) -> usize {
        self.buf.len()
    }
    pub fn num_bytes(&self) -> usize {
        self.buf.num_bytes()
    }

    /// Kernel launches and cuBLAS (Branches 4, 6) take `&CudaSlice` directly.
    pub fn as_slice(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    /// Also Branch 2's raw-byte path: `unsafe { as_slice_mut().transmute_mut::<u8>(n) }`.
    pub fn as_slice_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}
