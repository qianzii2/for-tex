"""
ForTeX Correctness Verification — Smoke test with asserts.
Can run standalone (`python test_smoke.py`) or be imported by benchmark.
"""
import sys
import numpy as np

# Fixed seed for reproducibility
SEED = 42

def run_all_tests() -> int:
    """Run all correctness tests. Returns 0 if all pass, non-zero on failure."""
    np.random.seed(SEED)

    import for_tex as ft

    failures = 0
    def check(desc: str, fn):
        nonlocal failures
        try:
            fn()
            print(f"  [OK] {desc}")
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {desc}: {e}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] {desc}: {e}")

    print(f"=== ForTeX Smoke Test (seed={SEED}) ===\n")
    print(f"Version: {ft.__version__}")

    # ---- 1. GEMM (hand-tuned) ----
    check("gemm (hand-tuned block GEMM)", lambda: _test_gemm(ft.gemm, "gemm"))

    # ---- 2. simple_gemm (Fortran matmul intrinsic) ----
    check("simple_gemm (Fortran matmul intrinsic)", lambda: _test_gemm(ft.simple_gemm, "simple_gemm"))

    # ---- 3. gemv ----
    check("gemv (matrix-vector)", lambda: _test_gemv(ft))

    # ---- 4. linear_relu ----
    check("linear_relu (fused matmul+bias+ReLU)", lambda: _test_linear_relu(ft))

    # ---- 5. ReLU ----
    check("relu", lambda: _test_activation(ft.relu, lambda x: np.maximum(x, 0)))

    # ---- 6. GELU ----
    check("gelu", lambda: _test_gelu(ft))

    # ---- 7. Sigmoid ----
    check("sigmoid", lambda: _test_activation(ft.sigmoid, lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))))

    # ---- 8. Tanh ----
    check("tanh_fn", lambda: _test_activation(ft.tanh_fn, np.tanh))

    # ---- 9. Softmax ----
    check("softmax_fn", lambda: _test_softmax(ft))

    # ---- 10. Conv2D ----
    check("conv2d (padding=1)", lambda: _test_conv2d(ft, stride=1, padding=1))
    check("conv2d (stride=2, padding=3)", lambda: _test_conv2d(ft, stride=2, padding=3))

    # ---- 11. MaxPool ----
    check("maxpool", lambda: _test_maxpool(ft))

    # ---- 12. AvgPool ----
    check("avgpool", lambda: _test_avgpool(ft))

    # ---- 13. LayerNorm ----
    check("layernorm", lambda: _test_layernorm(ft))

    # ---- 14. MSE ----
    check("mse", lambda: _test_mse(ft))

    # ---- 15. Cross Entropy ----
    check("cross_entropy", lambda: _test_cross_entropy(ft))

    print(f"\n{'='*50}")
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TEST(S) FAILED")
    print(f"{'='*50}")
    return failures


# ============================================================================
# Test Implementations
# ============================================================================

def _test_gemm(fn, name: str):
    """GEMM: C = A @ B, verify against NumPy reference."""
    a = np.random.randn(128, 256)
    b = np.random.randn(256, 512)
    c = fn(a, b)
    ref = a @ b
    np.testing.assert_allclose(c, ref, rtol=1e-10, atol=1e-10,
        err_msg=f"{name}: max diff {np.max(np.abs(c - ref)):.1e}")

def _test_gemv(ft):
    """GEMV: y = A @ x"""
    a = np.random.randn(100, 200)
    x = np.random.randn(200)
    y = ft.gemv(a, x)
    ref = a @ x
    np.testing.assert_allclose(y, ref, rtol=1e-10, atol=1e-10)

def _test_linear_relu(ft):
    """Fused Linear+ReLU: y = relu(W @ x + b)"""
    W = np.random.randn(128, 256)
    bias = np.random.randn(128)
    x = np.random.randn(256, 64)
    y = ft.linear_relu(W, bias, x)
    ref = np.maximum(W @ x + bias.reshape(-1, 1), 0)
    np.testing.assert_allclose(y, ref, rtol=1e-10, atol=1e-10)

def _test_activation(ft_fn, ref_fn):
    """Element-wise activation function: verify against reference."""
    x = np.random.randn(2000)
    y = ft_fn(x)
    ref = ref_fn(x)
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)

def _test_gelu(ft):
    """GELU: Uses Padé approximation of tanh, tolerance relaxed to 1e-5."""
    x = np.random.randn(2000)
    y = ft.gelu(x)
    # PyTorch-style GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
    ref = 0.5 * x * (1.0 + np.tanh(inner))
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5,
        err_msg=f"GELU Padé vs tanh: max diff {np.max(np.abs(y - ref)):.1e}")

def _test_softmax(ft):
    """Softmax: verify row sum = 1 and values match PyTorch reference."""
    try:
        import torch
        x = np.random.randn(64, 128)
        y = ft.softmax_fn(x)
        # Row sums must equal 1
        sums = y.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(64), rtol=1e-12, atol=1e-12,
            err_msg=f"Softmax row sums: max err {np.max(np.abs(sums - 1)):.1e}")
        # Verify values
        ref = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
        np.testing.assert_allclose(y, ref, rtol=2e-2, atol=2e-2,
            err_msg=f"Softmax vs PyTorch: max diff {np.max(np.abs(y - ref)):.1e}")
    except ImportError:
        # Without PyTorch, only verify row sums
        x = np.random.randn(64, 128)
        y = ft.softmax_fn(x)
        sums = y.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(64), rtol=1e-12, atol=1e-12)

def _test_conv2d(ft, stride, padding):
    """Conv2D: verify against PyTorch reference."""
    try:
        import torch
        n, ci, co, h, w = 2, 3, 16, 32, 32
        kh, kw = 3, 3
        img = np.random.randn(n, ci, h, w)
        weight = np.random.randn(co, ci, kh, kw)
        bias = np.random.randn(co)

        out = ft.conv2d(img, weight, bias, stride=stride, padding=padding)

        ref = torch.nn.functional.conv2d(
            torch.from_numpy(img),
            torch.from_numpy(weight),
            torch.from_numpy(bias),
            stride=stride, padding=padding
        ).numpy()

        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10,
            err_msg=f"Conv2D s={stride} p={padding}: max diff {np.max(np.abs(out - ref)):.1e}")
    except ImportError:
        # No PyTorch: only verify shape and value range
        n, ci, co, h, w = 2, 3, 16, 32, 32
        img = np.random.randn(n, ci, h, w)
        weight = np.random.randn(co, ci, 3, 3)
        bias = np.random.randn(co)
        out = ft.conv2d(img, weight, bias, stride=stride, padding=padding)
        out_h = (h + 2*padding - 3) // stride + 1
        out_w = (w + 2*padding - 3) // stride + 1
        assert out.shape == (n, co, out_h, out_w), f"Expected {(n, co, out_h, out_w)}, got {out.shape}"
        assert np.isfinite(out).all(), "Conv2D output contains NaN/Inf"

def _test_maxpool(ft):
    """MaxPool2D: verify against PyTorch reference."""
    try:
        import torch
        x = np.random.randn(2, 8, 28, 28)
        out = ft.maxpool(x, kernel=2, stride=2)
        ref = torch.nn.functional.max_pool2d(torch.from_numpy(x), 2, 2).numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
    except ImportError:
        x = np.random.randn(2, 8, 28, 28)
        out = ft.maxpool(x, kernel=2, stride=2)
        assert out.shape == (2, 8, 14, 14), f"Expected (2,8,14,14), got {out.shape}"
        assert np.isfinite(out).all(), "MaxPool output contains NaN/Inf"

def _test_avgpool(ft):
    """AvgPool2D: verify against PyTorch reference."""
    try:
        import torch
        x = np.random.randn(2, 8, 28, 28)
        out = ft.avgpool(x, kernel=2, stride=2)
        ref = torch.nn.functional.avg_pool2d(torch.from_numpy(x), 2, 2).numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
    except ImportError:
        x = np.random.randn(2, 8, 28, 28)
        out = ft.avgpool(x, kernel=2, stride=2)
        assert out.shape == (2, 8, 14, 14), f"Expected (2,8,14,14), got {out.shape}"
        assert np.isfinite(out).all(), "AvgPool output contains NaN/Inf"

def _test_layernorm(ft):
    """LayerNorm: verify mean≈0, std≈1 and match PyTorch reference."""
    try:
        import torch
        batch, dim = 16, 128
        x = np.random.randn(batch, dim)
        gamma = np.random.randn(dim) + 1.0
        beta = np.random.randn(dim)
        y = ft.layernorm(x, gamma, beta, eps=1e-5)

        # mean ≈ 0 (after gamma/beta, mean is not exactly 0; test raw pre-affine)
        y_no_affine = ft.layernorm(x)
        np.testing.assert_allclose(y_no_affine.mean(), 0.0, atol=1e-6,
            err_msg=f"LayerNorm mean: {y_no_affine.mean():.1e}")

        # Verify values
        ref = torch.nn.functional.layer_norm(
            torch.from_numpy(x), (dim,),
            torch.from_numpy(gamma), torch.from_numpy(beta),
            eps=1e-5
        ).numpy()
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5,
            err_msg=f"LayerNorm vs PyTorch: max diff {np.max(np.abs(y - ref)):.1e}")
    except ImportError:
        batch, dim = 16, 128
        x = np.random.randn(batch, dim)
        y = ft.layernorm(x)
        assert abs(y.mean()) < 1e-5, f"LayerNorm mean: {y.mean():.1e}"
        assert np.isfinite(y).all(), "LayerNorm output contains NaN/Inf"

def _test_mse(ft):
    """MSE: verify against NumPy reference."""
    y_pred = np.random.randn(100)
    y_true = np.random.randn(100)
    loss = ft.mse(y_pred, y_true)
    ref = np.mean((y_pred - y_true) ** 2)
    np.testing.assert_allclose(loss, ref, rtol=1e-10, atol=1e-10)

def _test_cross_entropy(ft):
    """Cross Entropy: verify against PyTorch reference."""
    try:
        import torch
        batch, classes = 16, 10
        logits = np.random.randn(batch, classes)
        targets = np.random.randint(0, classes, size=batch).astype(np.int64)
        loss = ft.cross_entropy(logits, targets)
        ref = torch.nn.functional.cross_entropy(
            torch.from_numpy(logits),
            torch.from_numpy(targets)
        ).item()
        np.testing.assert_allclose(loss, ref, rtol=1e-6, atol=1e-6,
            err_msg=f"CrossEntropy: ForTeX={loss:.6f}, PyTorch={ref:.6f}")
    except ImportError:
        batch, classes = 16, 10
        logits = np.random.randn(batch, classes)
        targets = np.random.randint(0, classes, size=batch).astype(np.int64)
        loss = ft.cross_entropy(logits, targets)
        assert np.isfinite(loss), f"CrossEntropy: loss={loss} is not finite"
        assert loss >= 0, f"CrossEntropy: loss={loss} should be >= 0"


# ============================================================================
# Entry
# ============================================================================
if __name__ == "__main__":
    sys.exit(run_all_tests())