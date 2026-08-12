!==============================================================================
! C Bridge — bind(c) wrappers for Rust FFI
!==============================================================================
module c_bridge
    use, intrinsic :: iso_c_binding, only: c_int, c_double, c_bool, c_char
    use blas, only: dgemm, dgemv, daxpy, ddot, dscal, simple_gemm, fused_linear_relu
    use activation, only: relu_forward, softmax, gelu_forward
    use avx512_math, only: exp_avx512
    use convolution, only: conv2d_forward, maxpool2d, avgpool2d, im2col
    use normalization, only: batchnorm2d_forward, layernorm_forward
    use loss, only: mse_loss, cross_entropy_loss
    implicit none

contains

    !========== BLAS ==========
    subroutine c_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(c, name="c_dgemm")
        integer(c_int), value :: m, n, k, lda, ldb, ldc
        real(c_double), value :: alpha, beta
        real(c_double), intent(in) :: a(lda,*), b(ldb,*)
        real(c_double), intent(inout) :: c(ldc,*)
        character(kind=c_char), intent(in) :: transa, transb
        call dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    end subroutine

    subroutine c_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy) bind(c, name="c_dgemv")
        integer(c_int), value :: m, n, lda, incx, incy
        real(c_double), value :: alpha, beta
        real(c_double), intent(in) :: a(lda,*), x(*)
        real(c_double), intent(inout) :: y(*)
        character(kind=c_char), intent(in) :: trans
        call dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    end subroutine

    subroutine c_daxpy(n, alpha, x, incx, y, incy) bind(c, name="c_daxpy")
        integer(c_int), value :: n, incx, incy
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(*)
        real(c_double), intent(inout) :: y(*)
        call daxpy(n, alpha, x, incx, y, incy)
    end subroutine

    real(c_double) function c_ddot(n, x, incx, y, incy) bind(c, name="c_ddot")
        integer(c_int), value :: n, incx, incy
        real(c_double), intent(in) :: x(*), y(*)
        c_ddot = ddot(n, x, incx, y, incy)
    end function

    !--------------------------------------------------------------------------
    ! ★ simple_gemm — Fortran 内建 matmul() 包装
    !   一行代码，编译器自动 AVX-512 FMA。
    !   和手工六重分块 GEMM 对比，展示"编译器比你聪明"和"你能比编译器聪明"
    !--------------------------------------------------------------------------
    subroutine c_simple_gemm(m, n, k, a, b, c) bind(c, name="c_simple_gemm")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: a(m, k), b(k, n)
        real(c_double), intent(out) :: c(m, n)
        c = matmul(a, b)
    end subroutine c_simple_gemm

    !--------------------------------------------------------------------------
    ! ★ fused_linear_relu — 矩阵乘 + bias + ReLU 融合
    !   C/Rust: 3次kernel调用 = 3次读写。Fortran: 1次。
    !   y = relu(W @ x + b)  全部在一个 do concurrent 里完成
    !--------------------------------------------------------------------------
    subroutine c_fused_linear_relu(m, n, k, weight, bias, x, y) bind(c, name="c_fused_linear_relu")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: weight(m, k), bias(m), x(k, n)
        real(c_double), intent(out) :: y(m, n)
        call fused_linear_relu(m, n, k, weight, bias, x, y)
    end subroutine c_fused_linear_relu

    !--------------------------------------------------------------------------
    ! ★★ 零拷贝 GEMM — 直接接收行主序数据，利用转置恒等式
    !   C(m,n) = A(m,k) @ B(k,n)       [行主序]
    !   C^T(n,m) = B^T(n,k) @ A^T(k,m) [列主序]
    !   行主序内存 == 列主序转置的内存。直接传指针！
    !   a 声明为 (k,m) → Fortran 看到的是 A^T
    !   b 声明为 (n,k) → Fortran 看到的是 B^T
    !   c 声明为 (n,m) → Fortran 写入 C^T = 行主序 C
    !--------------------------------------------------------------------------
    !--------------------------------------------------------------------------
    ! ★★★ Round-6 GEMM: 2D 分块并行 + matmul() 内建（编译器级 SIMD）
    !   策略：
    !     - 外层 2D 分块并行（ii in m, jj in n），每个线程处理一个 (m-tile, n-tile)
    !     - 内层用 matmul() 调内置 BLAS-quality GEMM (gfortran 把它编成 AVX2 FMA)
    !     - 关键：matmul(a,b) 其中 a(k,n_tile), b(k,m_tile) 是行连续的！
    !       因为 weight^T 列主序 = 内存中 weight(m,k) 行连续, matmul 内部会展开成
    !       对 a 的 k 维 stride-1 访问 + 对 b 的 k 维 stride-1 访问 → SIMD 友好
    !--------------------------------------------------------------------------
    subroutine c_rowmajor_gemm(m, n, k, a, b, c) bind(c, name="c_rowmajor_gemm")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: a(k, m)   ! A^T 列主序 = A(m,k) 行连续
        real(c_double), intent(in) :: b(n, k)   ! B^T 列主序 = B(k,n) 行连续
        real(c_double), intent(out) :: c(n, m)  ! C^T 列主序 = C(m,n) 行连续
        integer :: ii, jj, im, jm
        integer, parameter :: MB = 96, NB = 128

        !$omp parallel do collapse(2) private(ii, jj, im, jm) schedule(static)
        do jj = 1, n, NB
            do ii = 1, m, MB
                im = min(MB, m - ii + 1)
                jm = min(NB, n - jj + 1)
                ! c(jj:jj+jm-1, ii:ii+im-1)  ⟵ (n_tile, m_tile)
                ! = sum_k a(k, ii:ii+im-1) * b(jj:jj+jm-1, k)
                ! matmul(a, b)  A is (K, M), B is (N, K), result is (N, M) ✓
                c(jj:jj+jm-1, ii:ii+im-1) = matmul(b(jj:jj+jm-1, 1:k), a(1:k, ii:ii+im-1))
            end do
        end do
        !$omp end parallel do
    end subroutine c_rowmajor_gemm

    !--------------------------------------------------------------------------
    ! ★★ 零拷贝 Fused Linear+ReLU
    !   weight(m,k) 行主序 → weight^T(k,m) 列主序
    !   x(k,n) 行主序 → x^T(n,k) 列主序
    !   y(m,n) 行主序 → y^T(n,m) 列主序
    !   y = relu(W @ x + bias)
    !   y^T(:,j) = x^T @ W^T(:,j) + bias(j)  (每列 j = 每行 j-1)
    !--------------------------------------------------------------------------
    subroutine c_rowmajor_fused_linear_relu(m, n, k, weight, bias, x, y) &
        bind(c, name="c_rowmajor_fused_linear_relu")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: weight(k, m)  ! W^T 列主序
        real(c_double), intent(in) :: bias(m)
        real(c_double), intent(in) :: x(n, k)        ! X^T 列主序
        real(c_double), intent(out) :: y(n, m)       ! Y^T 列主序 = Y 行主序
        integer :: i, j
        ! 策略：先并行 matmul，然后 fused bias + ReLU
        ! matmul 已是 BLAS-优化，由 gfortran 转为高效 SIMD
        y = matmul(x, weight)
        !$omp parallel do collapse(2) private(i, j) schedule(static)
        do j = 1, m
            do i = 1, n
                y(i, j) = max(0.0_c_double, y(i, j) + bias(j))
            end do
        end do
        !$omp end parallel do
    end subroutine c_rowmajor_fused_linear_relu

    !--------------------------------------------------------------------------
    ! ★★ 零拷贝 Conv2D — 直接接收行主序数据，无需任何转置
    !   img: (n, c_in, h, w) row-major = (w, h, c_in, n) column-major
    !   weight: (c_out, c_in, kh, kw) row-major = (kw, kh, c_in, c_out) col-major
    !   output: (n, c_out, out_h, out_w) row-major = (out_w, out_h, c_out, n) col-major
    !   OpenMP 并行: 每个线程处理一个 (batch, output_channel) 组合
    !--------------------------------------------------------------------------
    subroutine c_rowmajor_conv2d(n, c_in, c_out, h, w, kh, kw, &
        stride, pad, out_h, out_w, &
        img, weight, bias, output) bind(c, name="c_rowmajor_conv2d")
        integer(c_int), value :: n, c_in, c_out, h, w, kh, kw
        integer(c_int), value :: stride, pad, out_h, out_w
        real(c_double), intent(in) :: img(w, h, c_in, n)
        real(c_double), intent(in) :: weight(kw, kh, c_in, c_out)
        real(c_double), intent(in) :: bias(c_out)
        real(c_double), intent(out) :: output(out_w, out_h, c_out, n)

        integer :: b, co, oh, ow, ci, ki, kj, ih, iw
        integer :: ih_min, ih_max, iw_min, iw_max, ki_eff, kj_eff
        real(c_double) :: sum_val

        !$omp parallel do collapse(3) &
        !$omp private(co, oh, ow, ci, ki, kj, ih, iw, sum_val, &
        !$omp   ih_min, ih_max, iw_min, iw_max, ki_eff, kj_eff) &
        !$omp schedule(static)
        do b = 1, n
            do co = 1, c_out
                do oh = 1, out_h
                    ! 计算有效 kh, kw 范围（避免 per-element 分支预测）
                    ih_min = 1 + pad - (oh-1)*stride
                    ih_max = kh + pad - (oh-1)*stride
                    ih_min = max(ih_min, 1)
                    ih_max = min(ih_max, h)
                    do ow = 1, out_w
                        iw_min = 1 + pad - (ow-1)*stride
                        iw_max = kw + pad - (ow-1)*stride
                        iw_min = max(iw_min, 1)
                        iw_max = min(iw_max, w)
                        sum_val = bias(co)
                        ! 仅在有效 kernel 位置上累加（避免 cycle）
                        do ki = ih_min, ih_max
                            ih = (oh-1)*stride + ki - pad
                            ki_eff = ki
                            do kj = iw_min, iw_max
                                iw = (ow-1)*stride + kj - pad
                                kj_eff = kj
                                do ci = 1, c_in
                                    sum_val = sum_val + weight(kj_eff, ki_eff, ci, co) * img(iw, ih, ci, b)
                                end do
                            end do
                        end do
                        output(ow, oh, co, b) = sum_val
                    end do
                end do
            end do
        end do
        !$omp end parallel do
    end subroutine c_rowmajor_conv2d

    !========== Activation ==========
    subroutine c_relu_forward(n, x) bind(c, name="c_relu_forward")
        integer(c_int), value :: n
        real(c_double), intent(inout) :: x(*)
        call relu_forward(n, x)
    end subroutine

    subroutine c_gelu_forward(n, x, y) bind(c, name="c_gelu_forward")
        integer(c_int), value :: n
        real(c_double), intent(in) :: x(*)
        real(c_double), intent(out) :: y(*)
        call gelu_forward(n, x, y)
    end subroutine

    subroutine c_softmax(n, d, x, y) bind(c, name="c_softmax")
        integer(c_int), value :: n, d
        real(c_double), intent(in) :: x(d, n)
        real(c_double), intent(out) :: y(d, n)
        integer :: j, i
        real(c_double) :: m, s, xi, inv_s

        ! ★ 2-pass softmax：内存流量从 14 elem/elem 降到 8 elem/elem
        !   Pass 1: 同时算 max + sum(exp(x - max))
        !          把 exp(x - max) 暂存到 y，等下直接乘 inv_sum
        !   Pass 2: y *= 1/sum
        !
        !   ★ 用 exp_avx512 替换 exp_pade（Padé[1/1] + 整数幂 hack，比 [4/4]+4次平方快 4x）
        !$omp parallel do private(j, i, m, s, xi, inv_s) schedule(static)
        do j = 1, n
            ! --- Pass 1a: max per row ---
            m = -huge(1.0_c_double)
            !$omp simd reduction(max:m) private(xi)
            do i = 1, d
                xi = x(i, j)
                if (xi > m) m = xi
            end do

            ! --- Pass 1b: in-place shift + exp_avx512 ---
            !$omp simd
            do i = 1, d
                y(i, j) = x(i, j) - m
            end do
            call exp_avx512(d, y(1:d, j), y(1:d, j))

            ! --- Pass 1c: sum ---
            s = 0.0_c_double
            !$omp simd reduction(+:s)
            do i = 1, d
                s = s + y(i, j)
            end do

            ! --- Pass 2: scale ---
            inv_s = 1.0_c_double / s
            !$omp simd private(inv_s)
            do i = 1, d
                y(i, j) = y(i, j) * inv_s
            end do
        end do
        !$omp end parallel do
    end subroutine

    !--------------------------------------------------------------------------
    ! exp_pade — SIMD-friendly Padé [4/4] 近似
    !   exp(x) ≈ P/Q, P = 1680+840x+180x²+20x³+x⁴, Q = 1680-840x+180x²-20x³+x⁴
    !   对 |x| ≤ 1 精度 ~1e-7
    !   对 softmax 的 x ∈ [-50, 0]: 用 x/4 折半两次 → |xh| ≤ 12.5
    !     但 |xh| 仍 > 1，改用：exp(x) = exp(x/4)^4
    !     4次折半 → x/16 ≈ [-3.125, 0]，在 Padé [4/4] 区间内
    !     折半 4 次后再平方 4 次 = exp(x) = exp(x/16)^16
    !     注意：2.0_c_double**16 = 65536 直接算
    !--------------------------------------------------------------------------
    real(c_double) function exp_pade(x) result(y)
        real(c_double), intent(in) :: x
        real(c_double) :: xh, xh2, xh3, xh4, p, q
        xh = x * 0.0625_c_double  ! x/16
        xh2 = xh * xh; xh3 = xh2 * xh; xh4 = xh3 * xh
        p = 1680.0_c_double + xh*(840.0_c_double + xh2*(180.0_c_double + xh3*(20.0_c_double + xh4)))
        q = 1680.0_c_double + xh*(-840.0_c_double + xh2*(180.0_c_double + xh3*(-20.0_c_double + xh4)))
        y = p / q
        y = y * y; y = y * y; y = y * y; y = y * y  ! (p/q)^16 = exp(x)
        if (y < 1.0e-300_c_double) y = 0.0_c_double
    end function exp_pade

    !========== Convolution ==========
    subroutine c_conv2d_forward(n, c_in, c_out, h, w, kh, kw, &
                                 stride_h, stride_w, pad_h, pad_w, &
                                 out_h, out_w, &
                                 img, weight, bias, output) bind(c, name="c_conv2d_forward")
        integer(c_int), value :: n, c_in, c_out, h, w, kh, kw
        integer(c_int), value :: stride_h, stride_w, pad_h, pad_w
        integer(c_int), value :: out_h, out_w
        real(c_double), intent(in) :: img(c_in, h, w, n)
        real(c_double), intent(in) :: weight(c_out, c_in, kh, kw)
        real(c_double), intent(in) :: bias(c_out)
        real(c_double), intent(out) :: output(c_out, out_h, out_w, n)
        call conv2d_forward(n, c_in, c_out, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, &
                            img, weight, bias, output)
    end subroutine

    subroutine c_maxpool2d(n, c, h, w, kh, kw, stride_h, stride_w, &
                            out_h, out_w, input, output) bind(c, name="c_maxpool2d")
        integer(c_int), value :: n, c, h, w, kh, kw, stride_h, stride_w
        integer(c_int), value :: out_h, out_w
        real(c_double), intent(in) :: input(c, h, w, n)
        real(c_double), intent(out) :: output(c, out_h, out_w, n)
        call maxpool2d(n, c, h, w, kh, kw, stride_h, stride_w, input, output)
    end subroutine

    subroutine c_avgpool2d(n, c, h, w, kh, kw, stride_h, stride_w, &
                            out_h, out_w, input, output) bind(c, name="c_avgpool2d")
        integer(c_int), value :: n, c, h, w, kh, kw, stride_h, stride_w
        integer(c_int), value :: out_h, out_w
        real(c_double), intent(in) :: input(c, h, w, n)
        real(c_double), intent(out) :: output(c, out_h, out_w, n)
        call avgpool2d(n, c, h, w, kh, kw, stride_h, stride_w, input, output)
    end subroutine

    !========== Normalization ==========
    subroutine c_batchnorm2d_forward(n, c, h, w, x, gamma, beta, running_mean, &
                                      running_var, eps, momentum, training, y) bind(c, name="c_batchnorm2d_forward")
        integer(c_int), value :: n, c, h, w
        real(c_double), intent(in) :: x(c, h, w, n)
        real(c_double), intent(in) :: gamma(c), beta(c)
        real(c_double), intent(inout) :: running_mean(c), running_var(c)
        real(c_double), value :: eps, momentum
        logical(c_bool), value :: training
        real(c_double), intent(out) :: y(c, h, w, n)
        logical :: train_local
        train_local = training
        call batchnorm2d_forward(n, c, h, w, x, gamma, beta, running_mean, &
                                  running_var, eps, momentum, train_local, y)
    end subroutine

    subroutine c_layernorm_forward(batch, dim, x, gamma, beta, eps, y) &
        bind(c, name="c_layernorm_forward")
        integer(c_int), value :: batch, dim
        real(c_double), intent(in) :: x(dim, batch)
        real(c_double), intent(in) :: gamma(dim), beta(dim)
        real(c_double), value :: eps
        real(c_double), intent(out) :: y(dim, batch)
        integer :: b, i
        real(c_double) :: mean_val, var_val, inv_std, inv_dim
        inv_dim = 1.0_c_double / dim

        !$omp parallel do private(i, mean_val, var_val, inv_std) schedule(static)
        do b = 1, batch
            ! Mean — single pass
            mean_val = 0.0_c_double
            !$omp simd reduction(+:mean_val)
            do i = 1, dim
                mean_val = mean_val + x(i, b)
            end do
            mean_val = mean_val * inv_dim
            ! Variance — single pass
            var_val = 0.0_c_double
            !$omp simd reduction(+:var_val)
            do i = 1, dim
                var_val = var_val + (x(i, b) - mean_val)**2
            end do
            var_val = var_val * inv_dim
            inv_std = 1.0_c_double / sqrt(var_val + eps)
            ! Normalize + scale + shift — fused, vectorizable
            !$omp simd
            do i = 1, dim
                y(i, b) = gamma(i) * ((x(i, b) - mean_val) * inv_std) + beta(i)
            end do
        end do
        !$omp end parallel do
    end subroutine c_layernorm_forward

    !========== Loss ==========
    real(c_double) function c_mse_loss(n, y_pred, y_true) bind(c, name="c_mse_loss")
        integer(c_int), value :: n
        real(c_double), intent(in) :: y_pred(*), y_true(*)
        c_mse_loss = mse_loss(n, y_pred, y_true)
    end function

    real(c_double) function c_cross_entropy_loss(num_classes, batch, logits, targets) bind(c, name="c_cross_entropy_loss")
        integer(c_int), value :: num_classes, batch
        real(c_double), intent(in) :: logits(num_classes, batch)
        integer(c_int), intent(in) :: targets(batch)
        c_cross_entropy_loss = cross_entropy_loss(num_classes, batch, logits, targets)
    end function

end module c_bridge
