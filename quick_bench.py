"""
Quick benchmark — fast performance data for all 25 test points.
3 trials per item, target_time=0.2s, warmup=3.
"""
import os, sys, time, platform
import numpy as np

os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
NTHREADS = int(os.environ.get("FORTEX_THREADS", os.cpu_count() or 8))
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NTHREADS))

import for_tex
import torch
torch.set_num_threads(NTHREADS)
torch.set_grad_enabled(False)

def auto_calibrate(fn, target=0.2, min_i=5, max_i=200):
    ts = []
    for _ in range(min_i):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    avg = sum(ts)/len(ts)
    return max(min(int(target/avg), max_i), min_i)

def bench(fn, warmup=3, trials=3, target=0.2):
    for _ in range(warmup): fn()
    iters = auto_calibrate(fn, target)
    trial_mins = []
    for _ in range(trials):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter()-t0)
        trial_mins.append(min(times))
    return np.min(trial_mins) * 1000.0

def sp(ft_ms, pt_ms):
    return pt_ms / ft_ms

def verdict(sp_val):
    if sp_val > 1.05: return "WIN"
    elif sp_val < 0.95: return "LOSS"
    return "TIE"

all_results = []

def measure(label, group, ft_fn, pt_fn):
    global all_results
    ft = bench(ft_fn)
    pt = bench(pt_fn)
    s = sp(ft, pt)
    v = verdict(s)
    print(f"  {label:<28s} ft={ft:>8.3f}ms  pt={pt:>8.3f}ms  {s:.2f}x {v}")
    all_results.append({'label':label,'group':group,'ft':ft,'pt':pt,'sp':s,'verdict':v})

print("=" * 72)
print("  ForTeX Quick Benchmark")
print("=" * 72)
print(f"  Threads: {NTHREADS}")

# ── 1. GEMM ──
print("\n--- 1. GEMM ---")
for m,n,k,lbl in [(256,256,256,"256³"),(512,512,512,"512³"),(1024,1024,1024,"1024³"),(2048,2048,2048,"2048³")]:
    a=np.random.randn(m,k); b=np.random.randn(k,n)
    at=torch.from_numpy(a); bt=torch.from_numpy(b)
    measure(lbl, "GEMM", lambda a=a,b=b: for_tex.gemm(a,b), lambda at=at,bt=bt: torch.matmul(at,bt))

# ── 2. Simple GEMM ──
print("\n--- 2. Simple GEMM ---")
m=n=k=512; a=np.random.randn(m,k); b=np.random.randn(k,n)
at=torch.from_numpy(a); bt=torch.from_numpy(b)
measure("512³", "Simple GEMM", lambda a=a,b=b: for_tex.simple_gemm(a,b), lambda at=at,bt=bt: torch.matmul(at,bt))

# ── 3. Linear+ReLU ──
print("\n--- 3. Fused Linear+ReLU ---")
for m,n,k,lbl in [(512,1024,512,"512×1024×512"),(1024,4096,1024,"1024×4096×1024")]:
    w=np.random.randn(m,k); b=np.random.randn(m); x=np.random.randn(k,n)
    xpt=torch.from_numpy(x.T.copy()); wpt=torch.from_numpy(w); bpt=torch.from_numpy(b)
    measure(lbl, "Linear+ReLU",
        lambda w=w,b=b,x=x: for_tex.linear_relu(w,b,x),
        lambda xpt=xpt,wpt=wpt,bpt=bpt: torch.relu(torch.nn.functional.linear(xpt,wpt,bpt)))

# ── 4. Conv2D ──
print("\n--- 4. Conv2D ---")
for n,ci,co,h,w,kh,kw,stride,pad,lbl in [
    (4,16,32,56,56,3,3,1,1,"16→32 56×56 k3p1"),
    (2,3,64,28,28,7,7,2,3,"3→64 28×28 k7s2p3"),
    (8,64,128,32,32,3,3,1,1,"64→128 32×32 k3p1")]:
    img=np.random.randn(n,ci,h,w); ker=np.random.randn(co,ci,kh,kw); bias=np.random.randn(co)
    it=torch.from_numpy(img); kt=torch.from_numpy(ker); bt=torch.from_numpy(bias)
    measure(lbl, "Conv2D",
        lambda img=img,k=ker,b=bias,s=stride,p=pad: for_tex.conv2d(img,k,b,stride=s,padding=p),
        lambda it=it,kt=kt,bt=bt,s=stride,p=pad: torch.nn.functional.conv2d(it,kt,bt,stride=s,padding=p))

# ── 5. Softmax ──
print("\n--- 5. Softmax ---")
for batch,dim,lbl in [(4,512,"4×512"),(64,512,"64×512"),(1024,512,"1024×512"),(4096,512,"4096×512"),
                       (1024,64,"1024×64"),(1024,4096,"1024×4096")]:
    x=np.random.randn(batch,dim); xt=torch.from_numpy(x)
    measure(lbl, "Softmax",
        lambda x=x: for_tex.softmax_fn(x),
        lambda xt=xt: torch.softmax(xt,dim=-1))

# ── 6. LayerNorm ──
print("\n--- 6. LayerNorm ---")
for batch,dim,lbl in [(4,512,"4×512"),(64,512,"64×512"),(1024,512,"1024×512"),(4096,512,"4096×512"),
                       (1024,64,"1024×64"),(1024,4096,"1024×4096")]:
    x=np.random.randn(batch,dim); g=np.ones(dim); b=np.zeros(dim)
    xt=torch.from_numpy(x); gt=torch.from_numpy(g); bt=torch.from_numpy(b)
    measure(lbl, "LayerNorm",
        lambda x=x,g=g,b=b: for_tex.layernorm(x,g,b,eps=1e-5),
        lambda xt=xt,gt=gt,bt=bt: torch.nn.functional.layer_norm(xt,(dim,),gt,bt,eps=1e-5))

# ── 7. GELU ──
print("\n--- 7. GELU ---")
for shape,lbl in [((512,512),"512×512"),((2048,2048),"2048×2048"),((4096,4096),"4096×4096")]:
    x=np.random.randn(*shape); xt=torch.from_numpy(x)
    measure(lbl, "GELU",
        lambda x=x: for_tex.gelu(x),
        lambda xt=xt: torch.nn.functional.gelu(xt))

# ── Summary ──
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
wins = sum(1 for r in all_results if r['sp'] > 1.05)
ties = sum(1 for r in all_results if 0.95 <= r['sp'] <= 1.05)
losses = sum(1 for r in all_results if r['sp'] < 0.95)
geo_mean = np.exp(np.mean(np.log([r['sp'] for r in all_results])))
print(f"  Total: {len(all_results)}  WIN: {wins}  TIE: {ties}  LOSS: {losses}")
print(f"  Geometric Mean Speedup: {geo_mean:.3f}x")
print(f"  Min: {min(r['sp'] for r in all_results):.3f}x  Max: {max(r['sp'] for r in all_results):.3f}x")

# Groups
from collections import defaultdict
groups = defaultdict(list)
for r in all_results: groups[r['group']].append(r)
print(f"\n  {'Group':<16s} {'Count':>5s} {'Min':>8s} {'Max':>8s} {'GeoMean':>8s}")
print(f"  {'─'*16}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
for g,items in groups.items():
    sps = [it['sp'] for it in items]
    gm = np.exp(np.mean(np.log(sps)))
    print(f"  {g:<16s}  {len(items):>5d}  {min(sps):>7.3f}x  {max(sps):>7.3f}x  {gm:>7.3f}x")
print("=" * 72)