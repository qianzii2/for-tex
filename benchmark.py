"""
ForTeX vs PyTorch Benchmark — 业界标准微基准测试
==================================================

设计参考：
  - Google Benchmark (C++): 自动校准, min 作为性能指标, 多 repetition
  - torch.utils.benchmark.Timer: blocked_autorange, min/median 报告
  - pytest-benchmark: IQR 离群值检测, 统计报告
  - asv (airspeed-velocity): 峰值性能检测, 多次 warmup

核心原则：
  1. min 作为"无干扰峰值性能"（排除 OS 调度/中断噪声）
  2. 自动校准迭代次数（确保每次 trial 足够长，减少 timer 精度影响）
  3. IQR 离群值检测（仅剔除真正的离群值，不无条件截断）
  4. PyTorch 对比零额外开销（无 .numpy() 转换，数据预转置）
  5. 正确性验证前置（smoke test 必须先通过）
"""
import os
import sys
import time
import platform
import numpy as np

# ── 环境变量必须在 import for_tex 之前设置 ──────────────────────────
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

NTHREADS = int(os.environ.get("FORTEX_THREADS", os.cpu_count() or 8))
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NTHREADS))

import for_tex
import torch

torch.set_num_threads(NTHREADS)
torch.set_grad_enabled(False)  # 纯推理模式，零 autograd 开销

# ── 正确性验证前置 ──────────────────────────────────────────────────
from test_smoke import run_all_tests

print("=" * 72)
print("  ForTeX Benchmark — Fair Comparison with PyTorch")
print("=" * 72)
print()
print("--- Correctness Check (smoke test) ---")
failures = run_all_tests()
if failures > 0:
    print(f"\n[ABORT] {failures} correctness test(s) failed. Benchmark refused.")
    sys.exit(1)
print("\n[OK] All correctness tests passed. Proceeding to benchmark.\n")


# ============================================================================
# Benchmark 核心函数
# ============================================================================

def auto_calibrate(fn, target_time=0.5, min_iters=10, max_iters=500):
    """
    自动校准迭代次数，使每次 trial 总耗时 >= target_time 秒。
    参考 torch.utils.benchmark.Timer.blocked_autorange()。
    """
    n_probe = min(min_iters, 10)
    times = []
    for _ in range(n_probe):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    if avg * min_iters >= target_time:
        return min_iters
    iters = int(target_time / avg)
    return max(min(iters, max_iters), min_iters)


def bench(fn, warmup=10, trials=7, target_time=0.5):
    """
    严谨的微基准测试。

    流程：
      1. 预热 (warmup) — JIT 编译、cache 预热
      2. 自动校准迭代次数
      3. trials 次独立 trial，每次取 min（Google Benchmark 标准）
      4. IQR 离群值检测 — 仅剔除受 OS 严重干扰的 trial
      5. 报告 min/median/mean±std

    返回: dict {'min','median','mean','std','n_trials','n_iters','unit'}
    """
    for _ in range(warmup):
        fn()

    iters = auto_calibrate(fn, target_time)

    trial_mins = []
    for _ in range(trials):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        trial_mins.append(min(times))  # ★ 每次 trial 取 min

    trial_mins = np.array(trial_mins) * 1000.0

    # IQR 离群值检测（仅当有足够方差时）
    if len(trial_mins) >= 5 and np.std(trial_mins) > 0:
        q1, q3 = np.percentile(trial_mins, 25), np.percentile(trial_mins, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clean = trial_mins[(trial_mins >= lower) & (trial_mins <= upper)]
        if len(clean) >= 3:
            trial_mins = clean

    return {
        'min': float(np.min(trial_mins)),
        'median': float(np.median(trial_mins)),
        'mean': float(np.mean(trial_mins)),
        'std': float(np.std(trial_mins)),
        'n_trials': len(trial_mins),
        'n_iters': iters,
        'unit': 'ms',
    }


def speedup(ft_result, pt_result):
    """Speedup = PyTorch_min / ForTeX_min。> 1.0 表示 ForTeX 更快。"""
    return pt_result['min'] / ft_result['min']


def verdict(sp, tight=False):
    """将 speedup 转为人类可读标签。"""
    t = 1.02 if tight else 1.05
    if sp > t:
        return "WIN"
    elif sp < 1.0 / t:
        return "LOSS"
    else:
        return "TIE"


# ============================================================================
# 格式化输出
# ============================================================================

def print_header():
    cpu_name = platform.processor() or "Unknown"
    print(f"System:      {platform.system()} {platform.release()} | {os.cpu_count()} logical cores")
    print(f"CPU:         {cpu_name}")
    print(f"Python:      {platform.python_version()}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"ForTeX:      {for_tex.__version__}")
    print(f"Threads:     {NTHREADS} (OMP_PROC_BIND=close, OMP_PLACES=cores)")
    print(f"Precision:   float64")


def fmt_time(ms):
    if ms < 0.001:
        return f"{ms*1e6:.1f}ns"
    elif ms < 1.0:
        return f"{ms*1e3:.1f}us"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def print_table(title, rows):
    """打印 benchmark 对比表格。"""
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")
    print(f"  {'':<22s} {'ForTeX':>34s}  {'PyTorch':>34s}  {'Speedup':>8s}  {'Verdict':>8s}")
    print(f"  {'':<22s} {'min':>10s}  {'median':>10s}  ±{'std':>8s}  {'min':>10s}  {'median':>10s}  ±{'std':>8s}")
    print(f"  {'─'*22}  {'─'*34}  {'─'*34}  {'─'*8}  {'─'*8}")
    for row in rows:
        ft, pt = row['fortex'], row['pytorch']
        sp = speedup(ft, pt)
        v = verdict(sp)
        print(f"  {row['label']:<22s} {fmt_time(ft['min']):>10s}  {fmt_time(ft['median']):>10s}  ±{fmt_time(ft['std']):>8s}  "
              f"{fmt_time(pt['min']):>10s}  {fmt_time(pt['median']):>10s}  ±{fmt_time(pt['std']):>8s}  "
              f"{sp:>7.2f}x  {v:>8s}")


# ============================================================================
# 各算子 Benchmark
# ============================================================================

def bench_gemm():
    """GEMM (DGEMM)"""
    print("\n" + "=" * 72)
    print("  1. GEMM (DGEMM) — 矩阵乘法")
    print("=" * 72)
    print("  ForTeX: K-blocked GEMM (Fortran)")
    print("  PyTorch: torch.matmul → MKL")

    configs = [(256,256,256,"256³"), (512,512,512,"512³"),
               (1024,1024,1024,"1024³"), (2048,2048,2048,"2048³")]
    rows = []
    for m, n, k, label in configs:
        a = np.random.randn(m, k)
        b = np.random.randn(k, n)
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        ft = bench(lambda a=a, b=b: for_tex.gemm(a, b))
        pt = bench(lambda a_t=a_t, b_t=b_t: torch.matmul(a_t, b_t))
        sp = speedup(ft, pt)
        print(f"  {label:<8s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("GEMM Summary", rows)
    return rows


def bench_simple_gemm():
    """Simple GEMM"""
    print("\n" + "=" * 72)
    print("  2. Simple GEMM (Fortran matmul intrinsic)")
    print("=" * 72)
    print("  ForTeX: matmul() → gfortran AVX-512 FMA")
    print("  PyTorch: torch.matmul → MKL")

    m, n, k = 512, 512, 512
    a = np.random.randn(m, k)
    b = np.random.randn(k, n)
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    ft = bench(lambda a=a, b=b: for_tex.simple_gemm(a, b))
    pt = bench(lambda a_t=a_t, b_t=b_t: torch.matmul(a_t, b_t))
    sp = speedup(ft, pt)
    print(f"  512³     ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
    rows = [{'label': '512³', 'fortex': ft, 'pytorch': pt}]
    print_table("Simple GEMM", rows)
    return rows


def bench_linear_relu():
    """Fused Linear+ReLU"""
    print("\n" + "=" * 72)
    print("  3. Fused Linear+ReLU")
    print("=" * 72)
    print("  ForTeX: matmul + bias + ReLU 一次内存遍历")
    print("  PyTorch: F.linear + relu 两个独立 kernel")

    configs = [(512,1024,512,"512×1024×512"), (1024,4096,1024,"1024×4096×1024")]
    rows = []
    for m, n, k, label in configs:
        weight = np.random.randn(m, k)
        bias = np.random.randn(m)
        x = np.random.randn(k, n)
        x_pt = torch.from_numpy(x.T.copy())  # 预转置，避免 lambda 内 .T 开销
        w_pt = torch.from_numpy(weight)
        b_pt = torch.from_numpy(bias)
        ft = bench(lambda w=weight, b=bias, x=x: for_tex.linear_relu(w, b, x))
        pt = bench(lambda xp=x_pt, wp=w_pt, bp=b_pt: torch.relu(torch.nn.functional.linear(xp, wp, bp)))
        sp = speedup(ft, pt)
        print(f"  {label:<16s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("Fused Linear+ReLU", rows)
    return rows


def bench_conv2d():
    """Conv2D"""
    print("\n" + "=" * 72)
    print("  4. Conv2D")
    print("=" * 72)
    print("  ForTeX: 直卷积（4 重循环，零拷贝，编译器 SIMD）")
    print("  PyTorch: F.conv2d → oneDNN")

    configs = [
        (4, 16, 32, 56, 56, 3, 3, 1, 1, "16→32 56×56 k3p1"),
        (2, 3, 64, 28, 28, 7, 7, 2, 3, "3→64 28×28 k7s2p3"),
        (8, 64, 128, 32, 32, 3, 3, 1, 1, "64→128 32×32 k3p1"),
    ]
    rows = []
    for n, ci, co, h, w, kh, kw, stride, pad, label in configs:
        img = np.random.randn(n, ci, h, w)
        kernel = np.random.randn(co, ci, kh, kw)
        bias = np.random.randn(co)
        img_t = torch.from_numpy(img)
        ker_t = torch.from_numpy(kernel)
        b_t = torch.from_numpy(bias)
        ft = bench(lambda img=img, k=kernel, b=bias, s=stride, p=pad:
                    for_tex.conv2d(img, k, b, stride=s, padding=p))
        pt = bench(lambda it=img_t, kt=ker_t, bt=b_t, s=stride, p=pad:
                    torch.nn.functional.conv2d(it, kt, bt, stride=s, padding=p))
        sp = speedup(ft, pt)
        print(f"  {label:<22s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("Conv2D", rows)
    return rows


def bench_softmax():
    """Softmax"""
    print("\n" + "=" * 72)
    print("  5. Softmax")
    print("=" * 72)
    print("  ForTeX: 2-pass + AVX-512 exp kernel")
    print("  PyTorch: torch.softmax → MKL SVML exp")

    configs = [
        (4, 512, "4×512 (tiny)"), (64, 512, "64×512 (small)"),
        (1024, 512, "1024×512 (med)"), (4096, 512, "4096×512 (large)"),
        (1024, 64, "1024×64 (s-dim)"), (1024, 4096, "1024×4096 (l-dim)"),
    ]
    rows = []
    for batch, dim, label in configs:
        x = np.random.randn(batch, dim)
        x_t = torch.from_numpy(x)
        ft = bench(lambda x=x: for_tex.softmax_fn(x))
        pt = bench(lambda x_t=x_t: torch.softmax(x_t, dim=-1))
        sp = speedup(ft, pt)
        print(f"  {label:<22s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("Softmax", rows)
    return rows


def bench_layernorm():
    """LayerNorm"""
    print("\n" + "=" * 72)
    print("  6. LayerNorm")
    print("=" * 72)
    print("  ForTeX: 2-pass (mean/var + normalize/affine)")
    print("  PyTorch: F.layer_norm → oneDNN fused kernel")

    configs = [
        (4, 512, "4×512 (tiny)"), (64, 512, "64×512 (small)"),
        (1024, 512, "1024×512 (med)"), (4096, 512, "4096×512 (large)"),
        (1024, 64, "1024×64 (s-dim)"), (1024, 4096, "1024×4096 (l-dim)"),
    ]
    rows = []
    for batch, dim, label in configs:
        x = np.random.randn(batch, dim)
        gamma = np.ones(dim)
        beta = np.zeros(dim)
        x_t = torch.from_numpy(x)
        g_t = torch.from_numpy(gamma)
        b_t = torch.from_numpy(beta)
        ft = bench(lambda x=x, g=gamma, b=beta: for_tex.layernorm(x, g, b, eps=1e-5))
        pt = bench(lambda xt=x_t, gt=g_t, bt=b_t: torch.nn.functional.layer_norm(xt, (dim,), gt, bt, eps=1e-5))
        sp = speedup(ft, pt)
        print(f"  {label:<22s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("LayerNorm", rows)
    return rows


def bench_gelu():
    """GELU"""
    print("\n" + "=" * 72)
    print("  7. GELU 激活")
    print("=" * 72)
    print("  ForTeX: Padé [7/8] 近似 tanh, 完全 SIMD 化")
    print("  PyTorch: F.gelu → MKL (libm tanh)")

    configs = [((512,512),"512×512"), ((2048,2048),"2048×2048"), ((4096,4096),"4096×4096")]
    rows = []
    for shape, label in configs:
        x = np.random.randn(*shape)
        x_t = torch.from_numpy(x)
        ft = bench(lambda x=x: for_tex.gelu(x))
        pt = bench(lambda x_t=x_t: torch.nn.functional.gelu(x_t))
        sp = speedup(ft, pt)
        print(f"  {label:<16s} ForTeX: {fmt_time(ft['min']):>10s}  |  PyTorch: {fmt_time(pt['min']):>10s}  |  {sp:.2f}x {verdict(sp)}")
        rows.append({'label': label, 'fortex': ft, 'pytorch': pt})
    print_table("GELU", rows)
    return rows


# ============================================================================
# 汇总
# ============================================================================

def print_summary(all_results):
    print("\n\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    wins = sum(1 for r in all_results if speedup(r['fortex'], r['pytorch']) > 1.05)
    ties = sum(1 for r in all_results if 0.95 <= speedup(r['fortex'], r['pytorch']) <= 1.05)
    losses = sum(1 for r in all_results if speedup(r['fortex'], r['pytorch']) < 0.95)

    print(f"  Total tests:  {len(all_results)}")
    print(f"  WIN:          {wins}")
    print(f"  TIE:          {ties}")
    print(f"  LOSS:         {losses}")
    print()

    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r['group']].append(r)

    print(f"  {'Operator':<20s}  {'Tests':>5s}  {'Min Speedup':>12s}  {'Max Speedup':>12s}  {'Verdict':>10s}")
    print(f"  {'─'*20}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}")
    for group, items in groups.items():
        sps = [speedup(it['fortex'], it['pytorch']) for it in items]
        mn, mx = min(sps), max(sps)
        if mn > 1.05:
            v = "WIN"
        elif mx < 0.95:
            v = "LOSS"
        else:
            v = "MIXED"
        print(f"  {group:<20s}  {len(items):>5d}  {mn:>11.2f}x  {mx:>11.2f}x  {v:>10s}")

    print()
    print("  Speedup = PyTorch_min / ForTeX_min")
    print("  > 1.0 → ForTeX faster;  < 1.0 → PyTorch faster")
    print("=" * 72)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print_header()

    all_results = []

    # 为每个结果附加 group 标签
    def tag(group, rows):
        for r in rows:
            r['group'] = group
        all_results.extend(rows)

    tag("GEMM",              bench_gemm())
    tag("Simple GEMM",       bench_simple_gemm())
    tag("Linear+ReLU",       bench_linear_relu())
    tag("Conv2D",            bench_conv2d())
    tag("Softmax",           bench_softmax())
    tag("LayerNorm",         bench_layernorm())
    tag("GELU",              bench_gelu())

    print_summary(all_results)