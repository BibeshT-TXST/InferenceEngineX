//! Branch 1 exit gate: bit-exact 64 MiB bf16 round trip + VRAM accounting on alloc/drop.
//! GPU-only: run on the laptop (`cargo test --test test_cuda_context -- --nocapture`).

use std::sync::Mutex;
use std::time::Instant;

use half::bf16;
use inference_engine_x::cuda::buffer::DeviceTensor;
use inference_engine_x::cuda::context::GpuContext;
use inference_engine_x::cuda::vram::PHASE1_VRAM_CAP_BYTES;
use inference_engine_x::error::EngineError;

/// cargo runs tests on parallel threads and they all share the process's primary CUDA
/// context, so an allocation in one test would corrupt another test's free-VRAM deltas.
static GPU: Mutex<()> = Mutex::new(());

const MIB: usize = 1024 * 1024;
/// 64 MiB of bf16 = [1, 1, 4096, 8192].
const NUMEL: usize = 64 * MIB / 2;
/// Headroom for memory-pool / page rounding; tighten once the laptop prints the real delta.
const SLACK: usize = 4 * MIB;

/// xorshift64, because `rand` is not a Branch 1 dependency. Raw bit patterns on purpose:
/// NaN/Inf/subnormals are included, and only a `to_bits()` comparison can verify those.
fn random_bf16(n: usize) -> Vec<bf16> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bf16::from_bits(s as u16)
        })
        .collect()
}

/// Syncs first: cudarc frees via `cuMemFreeAsync`, which only returns memory to the
/// driver at a sync point, so an unsynced reading after `drop` would look like a leak.
fn free_bytes(gpu: &GpuContext) -> anyhow::Result<usize> {
    gpu.synchronize()?;
    Ok(gpu.vram().query()?.1)
}

#[test]
fn roundtrip_64mib_bf16_and_vram_delta() -> anyhow::Result<()> {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = GpuContext::new(PHASE1_VRAM_CAP_BYTES)?;
    let (ctx_used, _, total) = gpu.vram().query()?;
    println!(
        "base context footprint (device-wide): {} MiB / {} MiB",
        ctx_used / MIB,
        total / MIB
    );

    let host = random_bf16(NUMEL);
    let free_before = free_bytes(&gpu)?;
    let mut t = DeviceTensor::<bf16>::zeros(&gpu, [1, 1, 4096, 8192])?;
    // saturating: if another process frees VRAM meanwhile, fail the assert, don't panic on underflow.
    let delta = free_before.saturating_sub(free_bytes(&gpu)?);
    println!(
        "alloc delta: {delta} B ({:.2} MiB)",
        delta as f64 / MIB as f64
    );
    assert!(
        (64 * MIB..=64 * MIB + SLACK).contains(&delta),
        "alloc delta {delta} B"
    );

    // Warm-up: the first copy pays the driver's one-time staging-buffer setup.
    t.copy_from_host(&host)?;
    const REPS: usize = 5;
    let start = Instant::now();
    for _ in 0..REPS {
        t.copy_from_host(&host)?;
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "HtoD (pageable): {:.2} GB/s",
        (REPS * t.num_bytes()) as f64 / secs / 1e9
    );

    let mut back = vec![bf16::ZERO; NUMEL];
    t.copy_to_host(&mut back)?;
    // position() rather than assert_eq! on the Vecs: a failure must not print 33M elements.
    let first_mismatch = host
        .iter()
        .zip(&back)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(first_mismatch, None, "first bit mismatch at index");

    drop(t);
    let leaked = free_before.saturating_sub(free_bytes(&gpu)?);
    println!("after drop: {leaked} B not yet returned");
    assert!(leaked <= SLACK, "{leaked} B not returned after drop");
    Ok(())
}

#[test]
fn guard_and_shape_errors() -> anyhow::Result<()> {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());

    // The context alone already uses more than 1 MiB device-wide, so the guard must trip.
    let tiny = GpuContext::new(MIB)?;
    let err = DeviceTensor::<bf16>::zeros(&tiny, [1, 1, 4096, 8192]).err();
    assert!(
        matches!(err, Some(EngineError::VramGuard { .. })),
        "{err:?}"
    );

    let gpu = GpuContext::new(PHASE1_VRAM_CAP_BYTES)?;
    let err = DeviceTensor::<f32>::zeros(&gpu, [1, 1, 0, 8]).err();
    assert!(
        matches!(err, Some(EngineError::InvalidShape { .. })),
        "{err:?}"
    );

    let mut t = DeviceTensor::<u32>::zeros(&gpu, [1, 1, 1, 8])?;
    let err = t.copy_from_host(&[0u32; 7]).err();
    assert!(
        matches!(
            err,
            Some(EngineError::ShapeMismatch {
                expected: 8,
                got: 7
            })
        ),
        "{err:?}"
    );
    Ok(())
}
