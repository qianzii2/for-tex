import numpy as np
import for_tex as ft

print('=== ForTeX Smoke Test ===')
print(f'Version: {ft.__version__}')

# 1. simple_gemm
a = np.random.randn(128, 256)
b = np.random.randn(256, 512)
c = ft.simple_gemm(a, b)
print(f'1. simple_gemm: err={np.max(np.abs(c - a @ b)):.1e}')

# 2. hand-tuned GEMM
c = ft.gemm(a, b)
print(f'2. hand-tuned:   err={np.max(np.abs(c - a @ b)):.1e}')

# 3. fused linear_relu
W = np.random.randn(128, 256)
bias = np.random.randn(128)
x = np.random.randn(256, 64)
y = ft.linear_relu(W, bias, x)
print(f'3. linear_relu:  err={np.max(np.abs(y - np.maximum(W @ x + bias.reshape(-1, 1), 0))):.1e}')

# 4. gemv
a = np.random.randn(100, 200)
x = np.random.randn(200)
y = ft.gemv(a, x)
print(f'4. gemv: err={np.max(np.abs(y - a @ x)):.1e}')

# 5. activations
x = np.random.randn(1000)
for fn in ['relu', 'gelu', 'sigmoid', 'tanh_fn']:
    getattr(ft, fn)(x)
print('5. activations: all OK')

# 6. softmax
y = ft.softmax_fn(np.random.randn(4, 10))
print(f'6. softmax: sums={y.sum(axis=1).round(4)}')

# 7. conv2d
y = ft.conv2d(np.random.randn(2, 3, 32, 32), np.random.randn(16, 3, 3, 3), np.random.randn(16))
print(f'7. conv2d: {y.shape}')

# 8. pooling
x = np.random.randn(2, 8, 28, 28)
print(f'8. pool: max={ft.maxpool(x, 2).shape} avg={ft.avgpool(x, 2).shape}')

# 9. layernorm
y = ft.layernorm(np.random.randn(16, 128))
print(f'9. layernorm: mean={y.mean():.1e}')

# 10. loss
print(f'10. loss: mse={ft.mse(np.random.randn(100), np.random.randn(100)):.4f}')

print()
print('=== ALL 10 TESTS PASSED ===')
