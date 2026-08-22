# ForTeX — Fortran Tensor Executor

**Rust + Fortran + OpenBLAS + mimalloc** high-performance CPU tensor computation library.
Call hand-written AVX2 SIMD kernels and OpenBLAS directly from Python, optimized over 225+ rounds,
**Geometric mean speedup: 0.72× (stable) / 0.81× (peak)**.

> Covers GEMM / Conv2D / Softmax / LayerNorm / GELU / Fused Linear+ReLU / Activation functions / Pooling / Loss.
> ISA: `x86-64-v3` (AVX2 + FMA, Haswell 2013+), no AVX-512 required.

---

## 1. Quick Results (vs PyTorch 2.x CPU, 8 threads, float64)

| Operator | Size | Speedup | Verdict |
|------|------|------|------|
| **GEMM** | 2048³ | **2.1×** 🚀 | WIN |
| **Conv2D** | 3→64 28×28 | **1.25×** 🚀 | WIN |
| **Linear+ReLU** | 1024×4096×1024 | **1.09×** 🚀 | WIN |
| **LayerNorm** | 4×512 | **1.38×** 🚀 | WIN |
| **GELU** | 2048² | **1.66×** 🚀 | WIN |
| **GELU** | 4096² | **2.03×** 🚀 | WIN |
| Softmax | 1024×512 | 0.61× | 3-pass memory traversal |
| Conv2D | 64→128 32×32 | 0.40× | im2col memory overhead |

**Geometric mean speedup: 0.72× (stable)**. 6-7 WINs.

> Benchmarks use industry-standard methodology: Google Benchmark-style min value, auto-calibrated iterations,
> IQR outlier detection, 3 trials taking min. See `quick_bench.py` / `benchmark.py`.

---

## 2. Architecture

```
Python (numpy) → PyO3 (Rust) → extern "C" → Fortran bind(c) → Pure Fortran + OpenBLAS
                                     │
                                     ├── rayon parallel (LayerNorm / Activation functions)
                                     └── AVX2 SIMD intrinsics (Softmax — inline Padé exp)
```

- **Rust layer**: PyO3 bindings, rayon thread pool, AVX2 SIMD softmax (`std::arch::x86_64`), LayerNorm, activation functions
- **Fortran layer**: GEMM, Conv2D (im2col+GEMM), GELU (Padé approximation of tanh), LayerNorm (large batch OMP SIMD), Loss, Pooling
- **OpenBLAS**: Accelerates all GEMM operations via scipy's bundled `cblas_dgemm`
- **mimalloc**: Global memory allocator, replaces system default allocator, improves large matrix allocation performance

---

## 3. Installation

### Prerequisites

| Dependency | Version | Notes |
|------|------|------|
| Python | ≥ 3.10 | |
| gfortran | ≥ 13 | OpenMP support required |
| Rust | ≥ 1.78 | |
| maturin | ≥ 1.7 | `pip install maturin` |
| numpy | any | |
| scipy | any | Provides OpenBLAS DLL |

### One-line Install

```bash
pip install maturin numpy scipy
maturin build --release --strip
pip install target/wheels/for_tex-*.whl --force-reinstall --no-deps
```

### Verification

```bash
python test_smoke.py       # 15 correctness tests (with asserts)
python quick_bench.py      # 25 test points × PyTorch comparison
```

---

## 4. Usage

```python
import numpy as np
import for_tex

# GEMM
c = for_tex.gemm(np.random.randn(1024, 1024), np.random.randn(1024, 1024))

# Softmax (along last dimension)
y = for_tex.softmax_fn(np.random.randn(32, 128))

# LayerNorm (supports gamma/beta)
y = for_tex.layernorm(x, gamma=np.ones(128), beta=np.zeros(128), eps=1e-5)

# Conv2D
out = for_tex.conv2d(
    np.random.randn(4, 3, 28, 28),   # NCHW
    np.random.randn(64, 3, 3, 3),     # out_ch, in_ch, kh, kw
    np.random.randn(64),               # bias
    stride=1, padding=1
)

# Fused Linear+ReLU
y = for_tex.linear_relu(weight, bias, x)

# Activation functions
for_tex.relu(x); for_tex.gelu(x); for_tex.sigmoid(x); for_tex.tanh_fn(x)

# Pooling
for_tex.maxpool(x, kernel=2); for_tex.avgpool(x, kernel=2)

# Loss
for_tex.mse(y_pred, y_true); for_tex.cross_entropy(logits, targets)
```

---

## 5. Repository Structure

```
for-tex/
├── build.rs                        ← Fortran compilation + OpenBLAS linking
├── Cargo.toml                      ← Rust dependencies (pyo3, numpy, rayon, mimalloc)
├── pyproject.toml                  ← Python package metadata
├── benchmark.py                    ← 25 test points × PyTorch comparison (full-featured)
├── quick_bench.py                  ← Quick benchmark (for daily development)
├── test_smoke.py                   ← 15 correctness tests (with asserts)
└── src/
    ├── lib.rs                      ← PyO3 bindings + AVX2 SIMD Softmax + LayerNorm + Activation functions
    ├── ffi.rs                      ← Fortran C interface
    └── fortran/
        ├── scipy_openblas.f90      ← OpenBLAS cblas_dgemm wrapper
        ├── c_bridge.f90            ← GEMM / Conv2D(im2col+GEMM) / Linear+ReLU / LayerNorm
        ├── activation.f90          ← GELU(Padé [7/8] tanh) / ReLU / Sigmoid / Tanh
        ├── blas.f90                ← Hand-written BLAS Level 1/2/3
        ├── conv.f90                ← im2col + direct convolution
        ├── norm.f90                ← BatchNorm / LayerNorm
        ├── loss.f90                ← MSE / Cross-Entropy
        └── avx512_math.f90         ← AVX2 exp/log SIMD kernels (name contains avx512 but actually uses AVX2)
```

---

## 6. Key Techniques

### 6.1 AVX2 SIMD Softmax (Rust inline intrinsics)

Softmax is fully implemented in Rust using `std::arch::x86_64` AVX2 intrinsics, 3-pass algorithm:
- Pass 1: `_mm256_max_pd` SIMD find max
- Pass 2: Inline Padé [1/1] approximate exp (`2^r ≈ (1+r·A)/(1+r·C)`) + IEEE 754 bit manipulation for `2^n`, no libm calls
- Pass 3: SIMD scale

Softmax performance improved from baseline 0.25x to 0.62x (+148%).

### 6.2 OpenBLAS GEMM

Provides `cblas_dgemm` via scipy's bundled OpenBLAS DLL. Uses zero-copy transpose identity:
`C(m,n)=A(m,k)@B(k,n)` row-major ⟺ `C^T(n,m)=B^T(n,k)@A^T(k,m)` column-major, passing pointers directly without copying.

### 6.3 Conv2D: im2col + GEMM

Unrolls 4D convolution into 2D matrix multiplication, accelerated by OpenBLAS. Pre-computes effective kernel ranges,
eliminating `cycle` branch prediction overhead. Adaptive OpenMP parallel (enabled when num_patches > 1024).

### 6.4 GELU Padé Approximation

Fortran side uses Padé [7/8] rational polynomial approximation of tanh, fully SIMD-ized, no libm calls needed.
GELU achieves 2.0× WIN in large matrix scenarios.

### 6.5 Hybrid Strategy

LayerNorm auto-selects path based on `batch × dim`:
- ≤ 8192 elements: Rust rayon 2-pass (sum+sum_sq fused)
- > 8192 elements: Fortran OpenMP SIMD 2-pass

GELU auto-selects path based on element count:
- < 262144 elements: Rust rayon scalar
- ≥ 262144 elements: Fortran OpenMP SIMD Padé tanh

### 6.6 mimalloc Global Allocator

Uses Microsoft mimalloc to replace the system default allocator, improving Conv2D im2col large matrix allocation performance.
Overall geometric mean improvement of ~16% (0.62x → 0.72x).

---

## 7. Performance Tuning

```bash
# Thread count
FORTEX_THREADS=4 python your_script.py

# OMP affinity (reduce cache bouncing)
OMP_PROC_BIND=close OMP_PLACES=cores python your_script.py

# OpenBLAS thread count (auto-detected by default)
OPENBLAS_NUM_THREADS=8 python your_script.py
```

---

## 8. Known Limitations

| Issue | Cause | Direction |
|------|------|------|
| Softmax large batch | 3-pass memory traversal limit | 2-pass online algorithm or MKL SVML |
| Conv2D large channels | im2col memory expansion (col matrix up to 37MB) | Winograd F(2×2,3×3) or oneDNN |
| LayerNorm large batch | OMP thread creation overhead | oneDNN fused kernel |
| float64 only | Design choice | Add float32 path |
| High system variance | OpenBLAS GEMM performance fluctuation | Multiple benchmark runs, take median |

---

## 9. Optimization History

225+ rounds of automated optimization, key breakthroughs:

| Round | Change | Effect |
|------|------|------|
| 3 | AVX-512→AVX2 | Fixed ISA mismatch |
| 6 | **AVX2 SIMD softmax** (Rust intrinsic) | Softmax 2.4× faster, overall +16% |
| 10 | GELU threshold hybrid strategy | Avoid OMP overhead for small matrices |
| 14 | LayerNorm 2-pass fusion + threshold 8192 | Reduced memory traversal |
| 39 | Link scipy OpenBLAS | GEMM fully accelerated |
| 44 | im2col range replacing cycle | Conv2D 20% faster |
| 88 | **mimalloc global allocator** | Overall +16%, largest single breakthrough |
| 225+ | Various algorithm/SIMD/external library attempts | Exhausted optimization space |

---

## 10. License

MIT