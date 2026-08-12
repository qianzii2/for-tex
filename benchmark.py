"""
ForTeX vs PyTorch vs NumPy Benchmark
=====================================
比较 GEMM、Conv2D、Softmax、LayerNorm 的纯计算性能
"""
import os
import time
import numpy as np
import torch
import for_tex

# ★ Round-15: OMP 线程亲和性 — 必须在这里设置，不能在 .venv 内设置
#   OMP_PROC_BIND=close: 让每个线程绑定到相邻核心，减少 cache bouncing
#   OMP_PLACES=cores: 每个核心一个 place
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

# 线程数 & 绑核：自动检测本机逻辑核心数；让 OMP / PyTorch / NumPy 全部用同一配置
NTHREADS = int(os.environ.get("FORTEX_THREADS", os.cpu_count() or 8))
torch.set_num_threads(NTHREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")
os.environ.setdefault("MKL_NUM_THREADS", str(NTHREADS))
print(f"[env] NTHREADS={NTHREADS} (override via FORTEX_THREADS env)")

def bench(name, fn, warmup=20, iters=200, trials=3):
    """运行 benchmark，返回中位耗时 (ms) — 多次 trial 取最稳定"""
    for _ in range(warmup):
        fn()
    trial_means = []
    for _ in range(trials):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        times = np.asarray(times)
        # 去掉最高的 20% 离群点（OS 抖动 / 上下文切换）
        keep = times < np.percentile(times, 80)
        trial_means.append(float(np.mean(times[keep])))
    trial_means = np.asarray(trial_means)
    # 取最快的 trial 中位数（最稳定的"能力上限"）
    best_trial = float(np.min(trial_means))
    median_trial = float(np.median(trial_means))
    print(f"  {name:<20s} {median_trial:8.3f} ms (best-of-{trials} {best_trial:.3f})")
    return median_trial

def compare(title, fortex_ms, torch_ms, numpy_ms=None):
    print(f"\n  【{title}】")
    if numpy_ms is not None:
        print(f"  NumPy   : {numpy_ms:8.3f} ms")
    print(f"  ForTeX  : {fortex_ms:8.3f} ms")
    print(f"  PyTorch : {torch_ms:8.3f} ms")
    speedup = torch_ms / fortex_ms
    label = "🚀 FASTER" if speedup > 1.01 else ("⚖️  TIE" if speedup > 0.99 else "🐢 slower")
    print(f"  ForTeX vs PyTorch: {speedup:.2f}x {label}")

torch.set_num_threads(NTHREADS)

print("=" * 64)
print("  ForTeX Benchmark")
print("=" * 64)

# ========== 1. GEMM (DGEMM) ==========
print("\n--- 1. GEMM (矩阵乘法) ---")

# 小矩阵
m, n, k = 256, 256, 256
a = np.random.randn(m, k)
b = np.random.randn(k, n)
a_t = torch.tensor(a)
b_t = torch.tensor(b)

ft_ms = bench("ForTeX gemm", lambda: for_tex.gemm(a, b), iters=200)
th_ms = bench("PyTorch matmul", lambda: torch.matmul(a_t, b_t), iters=200)
np_ms = bench("NumPy matmul", lambda: a @ b, iters=200)
compare("GEMM 256x256x256", ft_ms, th_ms, np_ms)

# 中型矩阵
m, n, k = 1024, 1024, 1024
a = np.random.randn(m, k)
b = np.random.randn(k, n)
a_t = torch.tensor(a)
b_t = torch.tensor(b)

ft_ms = bench("ForTeX gemm", lambda: for_tex.gemm(a, b), iters=50)
th_ms = bench("PyTorch matmul", lambda: torch.matmul(a_t, b_t), iters=50)
np_ms = bench("NumPy matmul", lambda: a @ b, iters=50)
compare("GEMM 1024x1024x1024", ft_ms, th_ms, np_ms)

# ========== 2. Simple GEMM (matmul) ==========
print("\n--- 2. Simple GEMM (Fortran matmul) ---")

m, n, k = 512, 512, 512
a = np.random.randn(m, k)
b = np.random.randn(k, n)
a_t = torch.tensor(a)
b_t = torch.tensor(b)

ft_ms = bench("ForTeX simple_gemm", lambda: for_tex.simple_gemm(a, b), iters=100)
th_ms = bench("PyTorch matmul", lambda: torch.matmul(a_t, b_t), iters=100)
np_ms = bench("NumPy matmul", lambda: a @ b, iters=100)
compare("Simple GEMM 512x512x512", ft_ms, th_ms, np_ms)

# ========== 3. Fused Linear+ReLU ==========
print("\n--- 3. Fused Linear+ReLU (算子融合) ---")

m, n, k = 512, 1024, 512
weight = np.random.randn(m, k)
bias = np.random.randn(m)
x = np.random.randn(k, n)
w_t = torch.tensor(weight)
b_t = torch.tensor(bias)
x_t = torch.tensor(x)

ft_ms = bench("ForTeX fused", lambda: for_tex.linear_relu(weight, bias, x), iters=100)
th_ms = bench("PyTorch F.linear+relu", lambda: torch.relu(torch.nn.functional.linear(x_t.T, w_t, b_t)).T.numpy(), iters=100)
compare("Fused Linear+ReLU", ft_ms, th_ms)

# ========== 4. Conv2D ==========
print("\n--- 4. Conv2D ---")

n, ci, co, h, w = 4, 16, 32, 56, 56
kh, kw = 3, 3
img = np.random.randn(n, ci, h, w)
kernel = np.random.randn(co, ci, kh, kw)
bias = np.random.randn(co)
img_t = torch.tensor(img)
kernel_t = torch.tensor(kernel)
bias_t = torch.tensor(bias)

ft_ms = bench("ForTeX conv2d", lambda: for_tex.conv2d(img, kernel, bias, padding=1), iters=20)
th_ms = bench("PyTorch conv2d", lambda: torch.nn.functional.conv2d(img_t, kernel_t, bias_t, padding=1).numpy(), iters=20)
compare("Conv2D 16->32 56x56", ft_ms, th_ms)

# 小型卷积
n, ci, co, h, w = 2, 3, 64, 28, 28
kh, kw = 7, 7
img = np.random.randn(n, ci, h, w)
kernel = np.random.randn(co, ci, kh, kw)
bias = np.random.randn(co)
img_t = torch.tensor(img)
kernel_t = torch.tensor(kernel)
bias_t = torch.tensor(bias)

ft_ms = bench("ForTeX conv2d", lambda: for_tex.conv2d(img, kernel, bias, stride=2, padding=3), iters=20)
th_ms = bench("PyTorch conv2d", lambda: torch.nn.functional.conv2d(img_t, kernel_t, bias_t, stride=2, padding=3).numpy(), iters=20)
compare("Conv2D 3->64 28x28 s2", ft_ms, th_ms)

# ========== 5. Softmax ==========
print("\n--- 5. Softmax ---")

batch, dim = 1024, 512
x = np.random.randn(batch, dim)
x_t = torch.tensor(x)

ft_ms = bench("ForTeX softmax", lambda: for_tex.softmax_fn(x), iters=100)
th_ms = bench("PyTorch softmax", lambda: torch.softmax(x_t, dim=-1).numpy(), iters=100)
compare("Softmax 1024x512", ft_ms, th_ms)

# ========== 6. LayerNorm ==========
print("\n--- 6. LayerNorm ---")

x = np.random.randn(batch, dim)
gamma = np.ones(dim)
beta = np.zeros(dim)
x_t = torch.tensor(x)
g_t = torch.tensor(gamma)
b_t = torch.tensor(beta)

ft_ms = bench("ForTeX layernorm", lambda: for_tex.layernorm(x, gamma, beta), iters=100)
th_ms = bench("PyTorch layernorm", lambda: torch.nn.functional.layer_norm(x_t, (dim,), g_t, b_t).numpy(), iters=100)
compare("LayerNorm 1024x512", ft_ms, th_ms)

# ========== 7. GELU ==========
print("\n--- 7. GELU 激活 ---")

x = np.random.randn(2048, 2048)
x_t = torch.tensor(x)

ft_ms = bench("ForTeX gelu", lambda: for_tex.gelu(x), iters=50)
th_ms = bench("PyTorch gelu", lambda: torch.nn.functional.gelu(x_t).numpy(), iters=50)
compare("GELU 2048x2048", ft_ms, th_ms)

print("\n" + "=" * 64)
print("  Benchmark Complete")
print("=" * 64)
