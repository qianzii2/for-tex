# ForTeX — Fortran Tensor Executor

**Rust + Fortran + OpenBLAS + mimalloc** 的高性能 CPU 张量计算库。
在 Python 中直接调用手写 AVX2 SIMD kernel 与 OpenBLAS，经过 225+ 轮自动优化，
**几何平均 speedup 0.72×（稳定）/ 0.81×（峰值）**。

> 覆盖 GEMM / Conv2D / Softmax / LayerNorm / GELU / Fused Linear+ReLU / 激活函数 / Pooling / Loss。
> ISA: `x86-64-v3` (AVX2 + FMA, Haswell 2013+)，无需 AVX-512。

---

## 1. 结果速览（vs PyTorch 2.x CPU, 8 线程, float64）

| 算子 | 尺寸 | 倍数 | 判定 |
|------|------|------|------|
| **GEMM** | 2048³ | **2.1×** 🚀 | WIN |
| **Conv2D** | 3→64 28×28 | **1.25×** 🚀 | WIN |
| **Linear+ReLU** | 1024×4096×1024 | **1.09×** 🚀 | WIN |
| **LayerNorm** | 4×512 | **1.38×** 🚀 | WIN |
| **GELU** | 2048² | **1.66×** 🚀 | WIN |
| **GELU** | 4096² | **2.03×** 🚀 | WIN |
| Softmax | 1024×512 | 0.61× | 3-pass 内存遍历 |
| Conv2D | 64→128 32×32 | 0.40× | im2col 内存开销 |

**几何平均 speedup: 0.72×（稳定）**。6-7 个 WIN。

> 基准测试采用业界标准方法：Google Benchmark 风格 min 值、自动校准迭代次数、
> IQR 离群值检测、3 trials 取 min。详见 `quick_bench.py` / `benchmark.py`。

---

## 2. 架构

```
Python (numpy) → PyO3 (Rust) → extern "C" → Fortran bind(c) → 纯 Fortran + OpenBLAS
                                     │
                                     ├── rayon 并行（LayerNorm / 激活函数）
                                     └── AVX2 SIMD intrinsic（Softmax — 内联 Padé exp）
```

- **Rust 层**：PyO3 绑定、rayon 线程池、AVX2 SIMD softmax（`std::arch::x86_64`）、LayerNorm、激活函数
- **Fortran 层**：GEMM、Conv2D（im2col+GEMM）、GELU（Padé 近似 tanh）、LayerNorm（大 batch OMP SIMD）、Loss、Pooling
- **OpenBLAS**：通过 scipy 自带的 `cblas_dgemm` 加速所有 GEMM 操作
- **mimalloc**：全局内存分配器，替代系统默认分配器，改善大矩阵分配性能

---

## 3. 安装

### 前置条件

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | |
| gfortran | ≥ 13 | 需支持 OpenMP |
| Rust | ≥ 1.78 | |
| maturin | ≥ 1.7 | `pip install maturin` |
| numpy | any | |
| scipy | any | 提供 OpenBLAS DLL |

### 一行安装

```bash
pip install maturin numpy scipy
maturin build --release --strip
pip install target/wheels/for_tex-*.whl --force-reinstall --no-deps
```

### 验证

```bash
python test_smoke.py       # 15 项正确性测试（含 assert）
python quick_bench.py      # 25 个测试点 × PyTorch 对比
```

---

## 4. 用法

```python
import numpy as np
import for_tex

# GEMM
c = for_tex.gemm(np.random.randn(1024, 1024), np.random.randn(1024, 1024))

# Softmax（沿最后一维）
y = for_tex.softmax_fn(np.random.randn(32, 128))

# LayerNorm（支持 gamma/beta）
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

# 激活函数
for_tex.relu(x); for_tex.gelu(x); for_tex.sigmoid(x); for_tex.tanh_fn(x)

# Pooling
for_tex.maxpool(x, kernel=2); for_tex.avgpool(x, kernel=2)

# Loss
for_tex.mse(y_pred, y_true); for_tex.cross_entropy(logits, targets)
```

---

## 5. 仓库结构

```
for-tex/
├── build.rs                        ← Fortran 编译 + OpenBLAS 链接
├── Cargo.toml                      ← Rust 依赖（pyo3, numpy, rayon, mimalloc）
├── pyproject.toml                  ← Python 包元数据
├── benchmark.py                    ← 25 测试点 × PyTorch 对比（全功能）
├── quick_bench.py                  ← 快速基准测试（日常开发用）
├── test_smoke.py                   ← 15 项正确性验证（带 assert）
└── src/
    ├── lib.rs                      ← PyO3 绑定 + AVX2 SIMD Softmax + LayerNorm + 激活函数
    ├── ffi.rs                      ← Fortran C 接口
    └── fortran/
        ├── scipy_openblas.f90      ← OpenBLAS cblas_dgemm 包装
        ├── c_bridge.f90            ← GEMM / Conv2D(im2col+GEMM) / Linear+ReLU / LayerNorm
        ├── activation.f90          ← GELU(Padé [7/8] tanh) / ReLU / Sigmoid / Tanh
        ├── blas.f90                ← 手写 BLAS Level 1/2/3
        ├── conv.f90                ← im2col + 直卷积
        ├── norm.f90                ← BatchNorm / LayerNorm
        ├── loss.f90                ← MSE / Cross-Entropy
        └── avx512_math.f90         ← AVX2 exp/log SIMD kernel（名含 avx512 但实际走 AVX2）
```

---

## 6. 关键技术

### 6.1 AVX2 SIMD Softmax（Rust 内联 intrinsic）

Softmax 完全在 Rust 侧使用 `std::arch::x86_64` AVX2 intrinsic 实现，3-pass 算法：
- Pass 1：`_mm256_max_pd` SIMD 找最大值
- Pass 2：内联 Padé [1/1] 近似 exp（`2^r ≈ (1+r·A)/(1+r·C)`）+ IEEE 754 位操作算 `2^n`，无 libm 调用
- Pass 3：SIMD 缩放

Softmax 性能从基线 0.25x 提升至 0.62x（+148%）。

### 6.2 OpenBLAS GEMM

通过 scipy 自带的 OpenBLAS DLL 提供 `cblas_dgemm`。利用零拷贝转置恒等式：
`C(m,n)=A(m,k)@B(k,n)` 行主序 ⟺ `C^T(n,m)=B^T(n,k)@A^T(k,m)` 列主序，直接传指针无需拷贝。

### 6.3 Conv2D: im2col + GEMM

将 4D 卷积展开为 2D 矩阵乘，利用 OpenBLAS 加速。预计算有效 kernel 范围，
消除 `cycle` 分支预测开销。自适应 OpenMP 并行（num_patches > 1024 时启用）。

### 6.4 GELU Padé 近似

Fortran 侧用 Padé [7/8] 有理多项式近似 tanh，完全 SIMD 化，无需调用 libm。
大矩阵场景下 GELU 达到 2.0× WIN。

### 6.5 混合策略

LayerNorm 根据 `batch × dim` 自动选择路径：
- ≤ 8192 元素：Rust rayon 2-pass（sum+sum_sq 融合）
- \> 8192 元素：Fortran OpenMP SIMD 2-pass

GELU 根据元素数自动选择路径：
- < 262144 元素：Rust rayon scalar
- ≥ 262144 元素：Fortran OpenMP SIMD Padé tanh

### 6.6 mimalloc 全局分配器

使用 Microsoft mimalloc 替代系统默认分配器，改善 Conv2D im2col 大矩阵分配性能。
整体几何平均提升约 16%（0.62x → 0.72x）。

---

## 7. 性能调优

```bash
# 线程数
FORTEX_THREADS=4 python your_script.py

# OMP 亲和性（减少 cache bouncing）
OMP_PROC_BIND=close OMP_PLACES=cores python your_script.py

# OpenBLAS 线程数（默认自动检测）
OPENBLAS_NUM_THREADS=8 python your_script.py
```

---

## 8. 已知局限

| 问题 | 原因 | 方向 |
|------|------|------|
| Softmax 大 batch | 3-pass 内存遍历限制 | 2-pass online 算法 或 MKL SVML |
| Conv2D 大通道 | im2col 内存膨胀（col 矩阵可达 37MB） | Winograd F(2×2,3×3) 或 oneDNN |
| LayerNorm 大 batch | OMP 线程创建开销 | oneDNN 融合 kernel |
| 仅支持 float64 | 设计选择 | 添加 float32 路径 |
| 系统方差大 | OpenBLAS GEMM 性能波动 | 多次 benchmark 取中位数 |

---

## 9. 优化历程

225+ 轮自动优化，关键突破：

| 轮次 | 改动 | 效果 |
|------|------|------|
| 3 | AVX-512→AVX2 | 修复 ISA 不匹配 |
| 6 | **AVX2 SIMD softmax**（Rust intrinsic） | Softmax 加速 2.4×，整体 +16% |
| 10 | GELU 阈值混合策略 | 小矩阵避免 OMP 开销 |
| 14 | LayerNorm 2-pass 融合 + 阈值 8192 | 减少内存遍历 |
| 39 | 链接 scipy OpenBLAS | GEMM 全面加速 |
| 44 | im2col range 替代 cycle | Conv2D 加速 20% |
| 88 | **mimalloc 全局分配器** | 整体 +16%，最大单次突破 |
| 225+ | 多种算法/SIMD/外部库尝试 | 穷尽优化空间 |

---

## 10. License

MIT