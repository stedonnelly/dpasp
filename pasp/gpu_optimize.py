"""GPU-accelerated credal polynomial optimization using PyTorch + custom CUDA kernels.

When a CUDA GPU is available, a custom fused CUDA kernel evaluates all 2^m corners
in a single kernel launch with no intermediate tensor materialization. Each CUDA thread
handles one corner, accumulating the polynomial sum in registers using direct factor
multiplication (no expf() — uses all CUDA cores instead of SFU-limited transcendentals).

For very large polynomials (n > TILE_N_THRESHOLD), a tiled kernel splits the n-loop
across thread blocks using shared memory, reducing global memory traffic.

Set DPASP_FP16=1 to store factor arrays in float16 (halves memory bandwidth, accumulates
in float32). Safe for typical probability values (0.01–0.99).

Falls back to PyTorch matmul (for MPS or when nvcc is unavailable), then to the C
implementation if no GPU is found at all.
"""

import ctypes
import traceback
import numpy as np
import os

_torch = None
_device = None
_available = None
_cuda_module = None
_cuda_module_tried = False

# Diagnostic log — collects reasons for GPU fallback so users can debug.
# Each entry is a (stage, message) tuple.  Printed by diagnose().
_fallback_log = []


def _log_fallback(stage, msg):
    """Record a GPU fallback reason for later diagnosis.

    'init' and 'kernel' stages are silent by default (no GPU is normal in CI).
    Runtime failures ('smp', 'minmax', 'batch') always print — those mean
    the GPU was expected to work but didn't.

    All entries are always recorded in _fallback_log for diagnose().
    """
    _fallback_log.append((stage, msg))
    # Always print runtime failures; init/kernel only with DPASP_GPU_DEBUG
    if stage in ('smp', 'minmax', 'batch'):
        print(f"[dpasp:gpu:{stage}] {msg}", flush=True)
    elif os.environ.get('DPASP_GPU_DEBUG', '0') == '1':
        print(f"[dpasp:gpu:{stage}] {msg}", flush=True)

# Threshold for switching to tiled kernel (n terms).  When n exceeds this, global
# memory traffic dominates — the tiled kernel uses shared memory to reduce it.
_TILE_N_THRESHOLD = 32768

# Maximum m for GPU path.  m=30 → 2^30 corners = 4GB result tensor (float32).
_MAX_GPU_M = 30


def _init():
    """Lazy init: check for GPU availability once.

    Set DPASP_NO_GPU=1 to force CPU mode (useful for debugging/benchmarking).
    """
    global _torch, _device, _available
    if _available is not None:
        return _available
    if os.environ.get('DPASP_NO_GPU', '0') == '1':
        _available = False
        _log_fallback('init', 'DPASP_NO_GPU=1 is set — GPU disabled by user')
        return False
    try:
        import torch
        _torch = torch
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            _available = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device('mps')
            _available = True
        else:
            _available = False
            reasons = []
            if not torch.cuda.is_available():
                reasons.append(f'torch.cuda.is_available()=False '
                               f'(CUDA runtime: {getattr(torch.version, "cuda", "None")})')
            if hasattr(torch.backends, 'mps'):
                reasons.append(f'torch.backends.mps.is_available()='
                               f'{torch.backends.mps.is_available()}')
            _log_fallback('init', f'No GPU backend found: {"; ".join(reasons)}')
    except ImportError as e:
        _available = False
        _log_fallback('init', f'PyTorch not installed: {e}')
    return _available


def is_gpu_available():
    """Check if GPU acceleration is available for credal optimization."""
    return _init()


def get_device_name():
    """Return a human-readable name of the GPU device, or None."""
    if not _init():
        return None
    if _device.type == 'cuda':
        return _torch.cuda.get_device_name(0)
    return str(_device)


def diagnose():
    """Print a full GPU diagnostic report to help debug fallback issues.

    Call this after a run that fell back to CPU to see exactly what failed.
    Also callable standalone:  python -c "from pasp.gpu_optimize import diagnose; diagnose()"

    Set DPASP_GPU_DEBUG=1 to get real-time diagnostics printed during execution.
    """
    print("=" * 70)
    print("dpasp GPU Diagnostic Report")
    print("=" * 70)

    # Environment
    print(f"\n--- Environment ---")
    print(f"  DPASP_NO_GPU  = {os.environ.get('DPASP_NO_GPU', '<not set>')}")
    print(f"  DPASP_FP16    = {os.environ.get('DPASP_FP16', '<not set>')}")
    print(f"  DPASP_GPU_DEBUG = {os.environ.get('DPASP_GPU_DEBUG', '<not set>')}")

    # PyTorch
    print(f"\n--- PyTorch ---")
    try:
        import torch
        print(f"  torch version : {torch.__version__}")
        print(f"  CUDA runtime  : {getattr(torch.version, 'cuda', 'None')}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device    : {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            print(f"  CUDA arch list: {torch.cuda.get_arch_list()}")
        if hasattr(torch.backends, 'mps'):
            print(f"  MPS available : {torch.backends.mps.is_available()}")
    except ImportError as e:
        print(f"  NOT INSTALLED: {e}")

    # CUDA kernel compilation
    print(f"\n--- CUDA Kernel ---")
    print(f"  Module tried  : {_cuda_module_tried}")
    print(f"  Module loaded : {_cuda_module is not None}")
    if _cuda_module is not None:
        for fn in ['credal_poly_eval_direct', 'credal_poly_eval_fp16',
                    'credal_poly_eval_tiled']:
            print(f"  {fn}: {hasattr(_cuda_module, fn)}")

    # nvcc
    import shutil
    nvcc = shutil.which('nvcc')
    print(f"  nvcc path     : {nvcc or 'NOT FOUND'}")
    if nvcc:
        import subprocess
        try:
            ver = subprocess.check_output([nvcc, '--version'], stderr=subprocess.STDOUT,
                                          timeout=5).decode().strip().split('\n')[-1]
            print(f"  nvcc version  : {ver}")
        except Exception as e:
            print(f"  nvcc version  : error ({e})")

    # ninja
    ninja = shutil.which('ninja')
    print(f"  ninja path    : {ninja or 'NOT FOUND (needed for fast JIT compilation)'}")

    # Fallback log
    print(f"\n--- Fallback Log ({len(_fallback_log)} entries) ---")
    if not _fallback_log:
        print("  (empty — no fallbacks recorded yet)")
    for i, (stage, msg) in enumerate(_fallback_log):
        print(f"  [{i+1}] [{stage}] {msg}")

    # Module state
    print(f"\n--- Module State ---")
    print(f"  _available    : {_available}")
    print(f"  _device       : {_device}")
    print(f"  _torch loaded : {_torch is not None}")
    print("=" * 70)


################################################################################
# Custom CUDA kernels — direct-product polynomial evaluation (no expf)
################################################################################

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Kernel 1: DIRECT-PRODUCT evaluation — one thread per corner.
//
// Each thread evaluates:
//   sum_i C[i] * prod_j factor(i,j)
// where factor(i,j) = fL[i*m+j] if bit j of corner b is set, else fU[i*m+j].
//
// This avoids expf() entirely — all work is FMA on CUDA cores, not SFU-limited.
// ---------------------------------------------------------------------------
__global__ void credal_poly_eval_direct_kernel(
    const float* __restrict__ C,     // (n,) coefficients
    const float* __restrict__ fL,    // (n, m) factor values when selecting L
    const float* __restrict__ fU,    // (n, m) factor values when selecting U
    float* __restrict__ result,      // (k,) output per corner
    const int n, const int m, const long long k
) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k) return;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float prod = 1.0f;
        const float* fL_row = fL + (long long)i * m;
        const float* fU_row = fU + (long long)i * m;
        #pragma unroll 4
        for (int j = 0; j < m; j++) {
            prod *= ((b >> j) & 1) ? fL_row[j] : fU_row[j];
        }
        sum += C[i] * prod;
    }
    result[b] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 1b: DIRECT-PRODUCT with FP16 factor storage.
//
// fL and fU are stored as __half to halve memory bandwidth.
// Accumulation remains float32 for precision.
// ---------------------------------------------------------------------------
__global__ void credal_poly_eval_fp16_kernel(
    const float* __restrict__ C,           // (n,) coefficients — float32
    const __half* __restrict__ fL_half,    // (n, m) factor values — float16
    const __half* __restrict__ fU_half,    // (n, m) factor values — float16
    float* __restrict__ result,            // (k,) output per corner
    const int n, const int m, const long long k
) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k) return;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float prod = 1.0f;
        const __half* fL_row = fL_half + (long long)i * m;
        const __half* fU_row = fU_half + (long long)i * m;
        #pragma unroll 4
        for (int j = 0; j < m; j++) {
            prod *= __half2float(((b >> j) & 1) ? fL_row[j] : fU_row[j]);
        }
        sum += C[i] * prod;
    }
    result[b] = sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: TILED evaluation for large n.
//
// Same 1-thread-per-corner model, but each thread processes n in chunks via
// shared memory.  A block cooperatively loads a tile of C/fL/fU into smem,
// then all threads in the block evaluate that tile.  This dramatically reduces
// global memory traffic when n >> L2 cache size.
//
// No intermediate (n_tiles, k) tensor needed — each thread accumulates its
// full sum across all tiles in registers.
// ---------------------------------------------------------------------------
#define TILE_SIZE 64   // terms per tile — fits C + fL + fU for m<=32 in 48KB smem

__global__ void credal_poly_eval_tiled_kernel(
    const float* __restrict__ C,
    const float* __restrict__ fL,
    const float* __restrict__ fU,
    float* __restrict__ result,    // (k,) output per corner
    const int n, const int m, const long long k
) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory layout:
    //   sC[TILE_SIZE] | sfL[TILE_SIZE * m] | sfU[TILE_SIZE * m]
    extern __shared__ float smem[];
    float* sC  = smem;
    float* sfL = smem + TILE_SIZE;
    float* sfU = smem + TILE_SIZE + TILE_SIZE * m;

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles of the n-dimension
    for (int i_start = 0; i_start < n; i_start += TILE_SIZE) {
        int tile_n = TILE_SIZE;
        if (i_start + tile_n > n) tile_n = n - i_start;

        // Cooperatively load C tile
        for (int idx = tid; idx < tile_n; idx += blockDim.x) {
            sC[idx] = C[i_start + idx];
        }
        // Cooperatively load fL and fU tiles
        int total_factors = tile_n * m;
        for (int idx = tid; idx < total_factors; idx += blockDim.x) {
            int row = idx / m;
            int col = idx % m;
            sfL[row * m + col] = fL[(long long)(i_start + row) * m + col];
            sfU[row * m + col] = fU[(long long)(i_start + row) * m + col];
        }
        __syncthreads();

        // Evaluate this tile's terms (only if thread is in-bounds)
        if (b < k) {
            for (int i = 0; i < tile_n; i++) {
                float prod = 1.0f;
                const float* sfL_row = sfL + i * m;
                const float* sfU_row = sfU + i * m;
                #pragma unroll 4
                for (int j = 0; j < m; j++) {
                    prod *= ((b >> j) & 1) ? sfL_row[j] : sfU_row[j];
                }
                sum += sC[i] * prod;
            }
        }
        __syncthreads();  // Ensure all threads done before loading next tile
    }

    if (b < k) {
        result[b] = sum;
    }
}

// ---------------------------------------------------------------------------
// C++ wrappers callable from Python
// ---------------------------------------------------------------------------

torch::Tensor credal_poly_eval_direct(torch::Tensor C, torch::Tensor fL,
                                       torch::Tensor fU, int m) {
    int n = C.size(0);
    long long k = 1LL << m;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(C.device());
    auto result = torch::empty({k}, opts);

    int threads = 256;
    int blocks = (int)((k + threads - 1) / threads);
    credal_poly_eval_direct_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(), fL.data_ptr<float>(), fU.data_ptr<float>(),
        result.data_ptr<float>(), n, m, k);

    return result;
}

torch::Tensor credal_poly_eval_fp16(torch::Tensor C, torch::Tensor fL_half,
                                     torch::Tensor fU_half, int m) {
    int n = C.size(0);
    long long k = 1LL << m;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(C.device());
    auto result = torch::empty({k}, opts);

    int threads = 256;
    int blocks = (int)((k + threads - 1) / threads);
    credal_poly_eval_fp16_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(),
        reinterpret_cast<const __half*>(fL_half.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(fU_half.data_ptr<at::Half>()),
        result.data_ptr<float>(), n, m, k);

    return result;
}

torch::Tensor credal_poly_eval_tiled(torch::Tensor C, torch::Tensor fL,
                                      torch::Tensor fU, int m) {
    int n = C.size(0);
    long long k = 1LL << m;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(C.device());
    auto result = torch::empty({k}, opts);

    int threads = 256;
    int blocks = (int)((k + threads - 1) / threads);
    // Shared memory: TILE_SIZE floats for C + 2 * TILE_SIZE * m floats for fL/fU
    int smem = (TILE_SIZE + 2 * TILE_SIZE * m) * (int)sizeof(float);

    credal_poly_eval_tiled_kernel<<<blocks, threads, smem>>>(
        C.data_ptr<float>(), fL.data_ptr<float>(), fU.data_ptr<float>(),
        result.data_ptr<float>(), n, m, k);

    return result;
}
"""

_CUDA_CPP = """
#include <torch/extension.h>
torch::Tensor credal_poly_eval_direct(torch::Tensor C, torch::Tensor fL,
                                       torch::Tensor fU, int m);
torch::Tensor credal_poly_eval_fp16(torch::Tensor C, torch::Tensor fL_half,
                                     torch::Tensor fU_half, int m);
torch::Tensor credal_poly_eval_tiled(torch::Tensor C, torch::Tensor fL,
                                      torch::Tensor fU, int m);
"""


def _get_cuda_module():
    """JIT-compile the custom CUDA kernels. Cached after first call.

    The compiled .so is cached in ~/.cache/torch_extensions/ — subsequent calls
    load the cached binary in ~1s. Only recompiles if the CUDA source changes.
    """
    global _cuda_module, _cuda_module_tried
    if _cuda_module_tried:
        return _cuda_module
    _cuda_module_tried = True
    try:
        if _device is None or _device.type != 'cuda':
            _log_fallback('kernel', f'Skipping CUDA kernel: device={_device}')
            return None
        from torch.utils.cpp_extension import load_inline
        _cuda_module = load_inline(
            name='credal_cuda',
            cpp_sources=_CUDA_CPP,
            cuda_sources=_CUDA_SOURCE,
            functions=['credal_poly_eval_direct',
                       'credal_poly_eval_fp16', 'credal_poly_eval_tiled'],
            verbose=False,
        )
    except Exception as e:
        tb = traceback.format_exc()
        _log_fallback('kernel', f'CUDA kernel compilation failed: {e}\n{tb}')
        print(f"[dpasp] CUDA kernel compilation failed ({e}), using PyTorch fallback",
              flush=True)
        _cuda_module = None
    return _cuda_module


def warmup():
    """Pre-compile the CUDA kernels so the first query doesn't pay the JIT cost.

    Call this once at startup (e.g. after import pasp) to trigger compilation.
    Subsequent calls are no-ops.  Compilation is cached on disk by PyTorch.
    """
    if _init() and _device is not None and _device.type == 'cuda':
        _get_cuda_module()


def _use_fp16():
    """Check if FP16 factor storage is requested via environment variable."""
    return os.environ.get('DPASP_FP16', '0') == '1'


def _eval_cuda(C, fL, fU, m):
    """Evaluate polynomial at ALL corners using the best available CUDA kernel.

    Selects kernel based on problem size and FP16 setting:
    - n > TILE_N_THRESHOLD: tiled kernel (shared memory, better cache behavior)
    - DPASP_FP16=1: FP16 storage kernel (halved memory bandwidth)
    - default: direct-product kernel (no expf, all CUDA cores)

    Returns (min_val, max_val) as Python floats, or None if kernel unavailable.
    """
    mod = _get_cuda_module()
    if mod is None:
        return None
    n = C.shape[0]
    if _use_fp16():
        fL_h = fL.half()
        fU_h = fU.half()
        result = mod.credal_poly_eval_fp16(C.contiguous(), fL_h.contiguous(),
                                            fU_h.contiguous(), m)
    elif n > _TILE_N_THRESHOLD:
        result = mod.credal_poly_eval_tiled(C.contiguous(), fL.contiguous(),
                                             fU.contiguous(), m)
    else:
        result = mod.credal_poly_eval_direct(C.contiguous(), fL.contiguous(),
                                              fU.contiguous(), m)
    # PyTorch min/max are fast single-kernel reductions on GPU
    return result.min().item(), result.max().item()


################################################################################
# PyTorch fallback path (for MPS or when CUDA kernel unavailable)
################################################################################

def _get_free_vram():
    """Query actual free VRAM in bytes. Falls back to a conservative estimate."""
    torch = _torch
    if _device is not None and _device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(_device)
            return free
        except Exception:
            pass
    return 2 * 1024 * 1024 * 1024  # conservative 2GB fallback


def _pick_batch_size(n, m):
    """Choose batch size based on polynomial size and actual free VRAM.

    _eval_polynomial_batched creates up to 3 simultaneous (batch, n) float32
    tensors (log_prod, exp result, weighted product), plus a (batch, m) bits
    tensor. We budget for the peak: 3 * batch * n * 4 bytes.
    """
    if n == 0:
        return min(1 << m, 65536)

    free_bytes = _get_free_vram()
    # Use at most 60% of free VRAM to leave headroom for fragmentation
    usable = int(free_bytes * 0.6)

    # Peak memory: 3 * (batch, n) float32 + (batch, m) float32
    bytes_per_corner = (3 * n + m) * 4
    batch = usable // bytes_per_corner
    batch = max(1, min(batch, 1 << m, 65536))
    # Round down to power of 2 for efficiency
    if batch > 1:
        batch = 1 << (batch.bit_length() - 1)
    return batch


def _eval_polynomial_batched(m, corners_batch, C, fL, fU):
    """Evaluate polynomial at a batch of corners on GPU (PyTorch fallback).

    Uses log-space matmul:
        log_prod[b, i] = base[i] + bits[b, :] @ delta[i, :]^T

    Memory-efficient: deletes intermediates as soon as they're consumed,
    keeping peak usage to ~2 * (batch, n) instead of 3.

    Args:
        m: number of variables
        corners_batch: (batch,) int64 tensor
        C: (n,) float32 tensor - coefficients
        fL: (n, m) float32 tensor - factor values when selecting L
        fU: (n, m) float32 tensor - factor values when selecting U

    Returns:
        (batch,) float32 tensor - polynomial values at each corner
    """
    torch = _torch
    shifts = torch.arange(m, device=_device)

    bits = ((corners_batch.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()  # (batch, m)

    # Precompute log-space quantities from (n, m) factor arrays — small tensors
    log_fL = torch.log(fL.clamp(min=1e-38))
    log_fU = torch.log(fU.clamp(min=1e-38))
    base = log_fU.sum(dim=1)        # (n,)
    delta = log_fL - log_fU          # (n, m)
    del log_fL, log_fU

    # Single big tensor: (batch, n)
    log_prod = base.unsqueeze(0) + bits @ delta.t()
    del bits, base, delta

    # exp in-place to avoid a second (batch, n) allocation
    log_prod.exp_()                   # now contains prod values

    # Multiply by C and sum — use matmul to avoid broadcasting a (batch, n) temp
    # log_prod is (batch, n), C is (n,) → matrix-vector product gives (batch,)
    return log_prod @ C               # (batch,)


def _precompute_factors(S, C, L, U):
    """Precompute factor arrays for direct-product evaluation.

    Returns (C_f32, fL, fU) all as float32 tensors on GPU.
    - fL[i,j] = L[j] if S[i,j] else 1-L[j]  — factor value when selecting Lower bound
    - fU[i,j] = U[j] if S[i,j] else 1-U[j]  — factor value when selecting Upper bound
    """
    torch = _torch
    fL = torch.where(S, L.unsqueeze(0), 1 - L.unsqueeze(0))  # (n, m) float32
    fU = torch.where(S, U.unsqueeze(0), 1 - U.unsqueeze(0))  # (n, m) float32
    return C.float(), fL, fU


def _to_gpu(S_flat, C, n, m):
    """Move polynomial data to GPU. S as bool, C as float32."""
    torch = _torch
    if n == 0:
        S = torch.zeros((0, m), dtype=torch.bool, device=_device)
        t_C = torch.zeros(0, dtype=torch.float32, device=_device)
        return S, t_C
    S = torch.tensor(
        np.asarray(S_flat, dtype=np.bool_).reshape(n, m),
        device=_device
    )
    t_C = torch.tensor(
        np.asarray(C, dtype=np.float64),
        dtype=torch.float32, device=_device
    )
    return S, t_C


def _batched_minmax(m, k, n, precomp, want_min):
    """Evaluate polynomial over all k corners in batches, returning min or max.

    Automatically halves batch size on OOM and retries, so this adapts to
    whatever VRAM is actually free at the moment.

    Args:
        m: number of variables
        k: total corners (2^m)
        n: number of polynomial terms (used for batch sizing)
        precomp: tuple (C, fL, fU) from _precompute_factors
        want_min: if True return global min, else global max

    Returns:
        float — the min or max polynomial value across all corners
    """
    torch = _torch
    best = float('inf') if want_min else float('-inf')
    batch = _pick_batch_size(n, m)

    start = 0
    while start < k:
        end = min(start + batch, k)
        try:
            corners = torch.arange(start, end, dtype=torch.int64, device=_device)
            values = _eval_polynomial_batched(m, corners, *precomp)
            val = values.min().item() if want_min else values.max().item()
            if want_min:
                best = min(best, val)
            else:
                best = max(best, val)
            del corners, values
            start = end
        except _torch.cuda.OutOfMemoryError:
            # OOM — free everything and retry with half the batch
            torch.cuda.empty_cache()
            old_batch = batch
            batch = max(1, batch // 2)
            _log_fallback('batch', f'OOM at batch={old_batch}, retrying with batch={batch}')
            if batch < 1:
                raise
    return best


def gpu_optimize_credal_smp(S_a_buf, C_a_buf, S_b_buf, C_b_buf, L_buf, U_buf,
                             n_a, n_b, m):
    """GPU-accelerated credal optimization for the smp (no evidence) case.

    Computes: low = min over corners of f_a(X), up = max over corners of f_b(X)

    Returns:
        (low, up) tuple of floats, or None if GPU not available
    """
    if not _init():
        return None
    if m > _MAX_GPU_M:
        print(f"[dpasp] m={m} exceeds GPU limit ({_MAX_GPU_M}), falling back to CPU",
              flush=True)
        return None

    torch = _torch
    k = 1 << m

    t_L = torch.tensor(np.asarray(L_buf, dtype=np.float64),
                       dtype=torch.float32, device=_device)
    t_U = torch.tensor(np.asarray(U_buf, dtype=np.float64),
                       dtype=torch.float32, device=_device)

    S_a, t_C_a = _to_gpu(S_a_buf, C_a_buf, n_a, m)
    S_b, t_C_b = _to_gpu(S_b_buf, C_b_buf, n_b, m)

    global_low = float('inf')
    global_up = float('-inf')

    with torch.no_grad():
        # Compute min of f_a
        if n_a > 0:
            precomp_a = _precompute_factors(S_a, t_C_a, t_L, t_U)
            del S_a, t_C_a
            cuda_result = _eval_cuda(*precomp_a, m)
            if cuda_result is not None:
                global_low = cuda_result[0]
            else:
                global_low = _batched_minmax(m, k, n_a, precomp_a, want_min=True)
            del precomp_a
        else:
            global_low = 0.0

        # Free memory before second polynomial
        torch.cuda.empty_cache()

        # Compute max of f_b
        if n_b > 0:
            precomp_b = _precompute_factors(S_b, t_C_b, t_L, t_U)
            del S_b, t_C_b
            cuda_result = _eval_cuda(*precomp_b, m)
            if cuda_result is not None:
                global_up = cuda_result[1]
            else:
                global_up = _batched_minmax(m, k, n_b, precomp_b, want_min=False)
            del precomp_b
        else:
            global_up = 0.0

    return (global_low, global_up)


def gpu_optimize_credal_minmax(S_a_buf, C_a_buf, S_b_buf, C_b_buf,
                                S_c_buf, C_c_buf, S_d_buf, C_d_buf,
                                L_buf, U_buf, n_a, n_b, n_c, n_d, m):
    """GPU-accelerated credal optimization for the evidence (minmax) case.

    Computes: low = min over corners of a/(a+d), up = max over corners of b/(b+c)

    Returns:
        (low, up) tuple of floats, or None if GPU not available
    """
    if not _init():
        return None
    if m > _MAX_GPU_M:
        print(f"[dpasp] m={m} exceeds GPU limit ({_MAX_GPU_M}), falling back to CPU",
              flush=True)
        return None

    torch = _torch
    k = 1 << m

    t_L = torch.tensor(np.asarray(L_buf, dtype=np.float64),
                       dtype=torch.float32, device=_device)
    t_U = torch.tensor(np.asarray(U_buf, dtype=np.float64),
                       dtype=torch.float32, device=_device)

    S_a, t_C_a = _to_gpu(S_a_buf, C_a_buf, n_a, m)
    S_b, t_C_b = _to_gpu(S_b_buf, C_b_buf, n_b, m)
    S_c, t_C_c = _to_gpu(S_c_buf, C_c_buf, n_c, m)
    S_d, t_C_d = _to_gpu(S_d_buf, C_d_buf, n_d, m)

    # Precompute factors for each polynomial
    polys = {}
    for label, S, C, n in [('a', S_a, t_C_a, n_a), ('b', S_b, t_C_b, n_b),
                             ('c', S_c, t_C_c, n_c), ('d', S_d, t_C_d, n_d)]:
        if n > 0:
            polys[label] = _precompute_factors(S, C, t_L, t_U)
        else:
            polys[label] = None
    del S_a, S_b, S_c, S_d, t_C_a, t_C_b, t_C_c, t_C_d

    global_low = 1.0
    global_up = 0.0

    with torch.no_grad():
        # Try CUDA kernel path
        mod = _get_cuda_module()
        if mod is not None:
            zeros_k = torch.zeros(k, dtype=torch.float32, device=_device)

            # Evaluate all four polynomials
            arr_a = _eval_cuda_full(*polys['a'], m, mod) if polys['a'] else zeros_k
            arr_b = _eval_cuda_full(*polys['b'], m, mod) if polys['b'] else zeros_k
            arr_c = _eval_cuda_full(*polys['c'], m, mod) if polys['c'] else zeros_k
            arr_d = _eval_cuda_full(*polys['d'], m, mod) if polys['d'] else zeros_k

            # y = a/(a+d), z = b/(b+c)
            denom_y = arr_a + arr_d
            y = torch.where(denom_y != 0, arr_a / denom_y, zeros_k)
            denom_z = arr_b + arr_c
            z = torch.where(denom_z != 0, arr_b / denom_z, zeros_k)

            global_low = y.min().item()
            global_up = z.max().item()
        else:
            # PyTorch matmul fallback — evaluate 4 polynomials per batch.
            # Budget VRAM for the largest polynomial (others are smaller or zero).
            max_n = max(n_a, n_b, n_c, n_d, 1)
            batch = _pick_batch_size(max_n, m)

            start = 0
            while start < k:
                end = min(start + batch, k)
                try:
                    corners = torch.arange(start, end, dtype=torch.int64, device=_device)
                    bsz = end - start

                    # Evaluate each polynomial sequentially to limit peak VRAM.
                    # Each call allocates ~1 (batch, n) tensor; del frees before next.
                    val_a = _eval_polynomial_batched(m, corners, *polys['a']) if polys['a'] else \
                        torch.zeros(bsz, dtype=torch.float32, device=_device)
                    val_d = _eval_polynomial_batched(m, corners, *polys['d']) if polys['d'] else \
                        torch.zeros(bsz, dtype=torch.float32, device=_device)

                    denom_y = val_a + val_d
                    y_batch = torch.where(denom_y != 0, val_a / denom_y,
                                          torch.zeros_like(val_a))
                    batch_low = y_batch.min().item()
                    del val_a, val_d, denom_y, y_batch

                    val_b = _eval_polynomial_batched(m, corners, *polys['b']) if polys['b'] else \
                        torch.zeros(bsz, dtype=torch.float32, device=_device)
                    val_c = _eval_polynomial_batched(m, corners, *polys['c']) if polys['c'] else \
                        torch.zeros(bsz, dtype=torch.float32, device=_device)

                    denom_z = val_b + val_c
                    z_batch = torch.where(denom_z != 0, val_b / denom_z,
                                          torch.zeros_like(val_b))
                    batch_up = z_batch.max().item()
                    del val_b, val_c, denom_z, z_batch, corners

                    if batch_low < global_low:
                        global_low = batch_low
                    if batch_up > global_up:
                        global_up = batch_up
                    start = end
                except _torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    old_batch = batch
                    batch = max(1, batch // 2)
                    _log_fallback('batch', f'OOM at batch={old_batch} (minmax), '
                                  f'retrying with batch={batch}')
                    if batch < 1:
                        raise

    return (global_low, global_up)


def _eval_cuda_full(C, fL, fU, m, mod):
    """Evaluate polynomial at ALL corners, returning the full (k,) result tensor."""
    n = C.shape[0]
    if _use_fp16():
        return mod.credal_poly_eval_fp16(C.contiguous(), fL.half().contiguous(),
                                          fU.half().contiguous(), m)
    elif n > _TILE_N_THRESHOLD:
        return mod.credal_poly_eval_tiled(C.contiguous(), fL.contiguous(),
                                           fU.contiguous(), m)
    else:
        return mod.credal_poly_eval_direct(C.contiguous(), fL.contiguous(),
                                            fU.contiguous(), m)


def _array_from_ptr(ptr, count, dtype):
    """Create a numpy array from a raw C pointer (as integer) without copying."""
    if ptr == 0 or count == 0:
        return np.empty(0, dtype=dtype)
    if dtype == np.bool_:
        ct = ctypes.c_bool * count
    elif dtype == np.float64:
        ct = ctypes.c_double * count
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    arr = ct.from_address(ptr)
    return np.ctypeslib.as_array(arr)


def _gpu_optimize_smp_from_c(s_a_ptr, c_a_ptr, s_b_ptr, c_b_ptr, l_ptr, u_ptr,
                               n_a, n_b, m):
    """C-callable wrapper for gpu_optimize_credal_smp.

    Receives raw C pointers as integers and constructs numpy arrays from them.
    Catches all exceptions so the C caller gets None (→ CPU fallback) instead of
    an unhandled Python exception that C would silently clear with PyErr_Clear().
    """
    try:
        if not _init():
            return None

        S_a_buf = _array_from_ptr(s_a_ptr, n_a * m, np.bool_)
        C_a_buf = _array_from_ptr(c_a_ptr, n_a, np.float64)
        S_b_buf = _array_from_ptr(s_b_ptr, n_b * m, np.bool_)
        C_b_buf = _array_from_ptr(c_b_ptr, n_b, np.float64)
        L_buf = _array_from_ptr(l_ptr, m, np.float64)
        U_buf = _array_from_ptr(u_ptr, m, np.float64)

        return gpu_optimize_credal_smp(S_a_buf, C_a_buf, S_b_buf, C_b_buf,
                                        L_buf, U_buf, n_a, n_b, m)
    except Exception as e:
        tb = traceback.format_exc()
        _log_fallback('smp', f'GPU optimization failed (n_a={n_a}, n_b={n_b}, m={m}): '
                      f'{type(e).__name__}: {e}\n{tb}')
        return None


def _gpu_optimize_minmax_from_c(s_a_ptr, c_a_ptr, s_b_ptr, c_b_ptr,
                                  s_c_ptr, c_c_ptr, s_d_ptr, c_d_ptr,
                                  l_ptr, u_ptr, n_a, n_b, n_c, n_d, m):
    """C-callable wrapper for gpu_optimize_credal_minmax.

    Receives raw C pointers as integers and constructs numpy arrays from them.
    Catches all exceptions so the C caller gets None (→ CPU fallback) with logging.
    """
    try:
        if not _init():
            return None

        S_a_buf = _array_from_ptr(s_a_ptr, n_a * m, np.bool_)
        C_a_buf = _array_from_ptr(c_a_ptr, n_a, np.float64)
        S_b_buf = _array_from_ptr(s_b_ptr, n_b * m, np.bool_)
        C_b_buf = _array_from_ptr(c_b_ptr, n_b, np.float64)
        S_c_buf = _array_from_ptr(s_c_ptr, n_c * m, np.bool_)
        C_c_buf = _array_from_ptr(c_c_ptr, n_c, np.float64)
        S_d_buf = _array_from_ptr(s_d_ptr, n_d * m, np.bool_)
        C_d_buf = _array_from_ptr(c_d_ptr, n_d, np.float64)
        L_buf = _array_from_ptr(l_ptr, m, np.float64)
        U_buf = _array_from_ptr(u_ptr, m, np.float64)

        return gpu_optimize_credal_minmax(S_a_buf, C_a_buf, S_b_buf, C_b_buf,
                                           S_c_buf, C_c_buf, S_d_buf, C_d_buf,
                                           L_buf, U_buf, n_a, n_b, n_c, n_d, m)
    except Exception as e:
        tb = traceback.format_exc()
        _log_fallback('minmax', f'GPU optimization failed '
                      f'(n_a={n_a}, n_b={n_b}, n_c={n_c}, n_d={n_d}, m={m}): '
                      f'{type(e).__name__}: {e}\n{tb}')
        return None
