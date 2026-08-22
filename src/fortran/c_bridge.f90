!==============================================================================
! C Bridge — bind(c) wrappers for Rust FFI
!==============================================================================
module c_bridge
    use, intrinsic :: iso_c_binding, only: c_int, c_double, c_bool, c_char
    use blas, only: simple_gemm
    use activation, only: gelu_forward
    use avx512_math, only: exp_avx512
    use scipy_openblas, only: scipy_dgemm, scipy_dgemm_generic
    use convolution, only: maxpool2d, avgpool2d
    use normalization, only: layernorm_forward
    use loss, only: mse_loss, cross_entropy_loss
    implicit none

contains

    !========== GEMM ==========
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
        integer, parameter :: MB = 96, NB = 128  ! unused

        call scipy_dgemm(m, n, k, a, b, c)
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
        call scipy_dgemm_generic(n, m, k, x, n, weight, k, y, n)
        !$omp parallel do collapse(2) private(i, j) schedule(static)
        do j = 1, m
            do i = 1, n
                y(i, j) = max(0.0_c_double, y(i, j) + bias(j))
            end do
        end do
        !$omp end parallel do
    end subroutine c_rowmajor_fused_linear_relu

    !--------------------------------------------------------------------------
    ! ★★ 零拷贝 Conv2D — im2col + matmul 策略
    !   img: (n, c_in, h, w) row-major = (w, h, c_in, n) column-major
    !   weight: (c_out, c_in, kh, kw) row-major = (kw, kh, c_in, c_out) col-major
    !   output: (n, c_out, out_h, out_w) row-major = (out_w, out_h, c_out, n) col-major
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

        integer :: patch_size, num_patches
        real(c_double), allocatable :: col(:,:), out_col(:,:), w_reshaped(:,:)
        integer :: b, co, oh, ow, ci, ki, kj, ix, iy, pe, pi, ki_min, ki_max, kj_min, kj_max

        patch_size = c_in * kh * kw
        num_patches = out_h * out_w * n

        allocate(col(patch_size, num_patches))
        allocate(out_col(c_out, num_patches))
        col = 0.0_c_double  ! ★ 必须清零：padding 位置不写入

        ! ── im2col: 自适应 OpenMP ──
        if (num_patches > 1024) then
            !$omp parallel do collapse(3) private(b, oh, ow, ci, ki, kj, ix, iy, pe, pi, ki_min, ki_max, kj_min, kj_max) &
            !$omp schedule(static)
        do b = 1, n
            do oh = 1, out_h
                do ow = 1, out_w
                    pi = (b-1)*out_h*out_w + (oh-1)*out_w + ow
                    ki_min = max(1, 1+pad-(oh-1)*stride)
                    ki_max = min(kh, h+pad-(oh-1)*stride)
                    kj_min = max(1, 1+pad-(ow-1)*stride)
                    kj_max = min(kw, w+pad-(ow-1)*stride)
                    do ci = 1, c_in
                        do ki = ki_min, ki_max
                            iy = (oh-1)*stride + ki - pad
                            do kj = kj_min, kj_max
                                ix = (ow-1)*stride + kj - pad
                                pe = (ci-1)*kh*kw + (ki-1)*kw + kj
                                col(pe, pi) = img(ix, iy, ci, b)
                            end do
                        end do
                    end do
                end do
            end do
        end do
        !$omp end parallel do
        else
            do b = 1, n
                do oh = 1, out_h
                    do ow = 1, out_w
                        pi = (b-1)*out_h*out_w + (oh-1)*out_w + ow
                        do ci = 1, c_in
                            do ki = 1, kh
                                iy = (oh-1)*stride + ki - pad
                                if (iy < 1 .or. iy > h) cycle
                                do kj = 1, kw
                                    ix = (ow-1)*stride + kj - pad
                                    if (ix < 1 .or. ix > w) cycle
                                    pe = (ci-1)*kh*kw + (ki-1)*kw + kj
                                    col(pe, pi) = img(ix, iy, ci, b)
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end if

        ! GEMM via OpenBLAS cblas_dgemm wrapper
        allocate(w_reshaped(c_out, patch_size))
        do co = 1, c_out
            do ci = 1, c_in
                do ki = 1, kh
                    do kj = 1, kw
                        pe = (ci-1)*kh*kw + (ki-1)*kw + kj
                        w_reshaped(co, pe) = weight(kj, ki, ci, co)
                    end do
                end do
            end do
        end do
        call scipy_dgemm_generic(c_out, num_patches, patch_size, w_reshaped, c_out, col, patch_size, out_col, c_out)
        deallocate(w_reshaped)

        ! ── add bias + reshape output ──
        do b = 1, n
            do oh = 1, out_h
                do ow = 1, out_w
                    pi = (b-1)*out_h*out_w + (oh-1)*out_w + ow
                    do co = 1, c_out
                        output(ow, oh, co, b) = out_col(co, pi) + bias(co)
                    end do
                end do
            end do
        end do

        deallocate(col, out_col)
    end subroutine c_rowmajor_conv2d

    !========== Activation ==========
    subroutine c_gelu_forward(n, x, y) bind(c, name="c_gelu_forward")
        integer(c_int), value :: n
        real(c_double), intent(in) :: x(*)
        real(c_double), intent(out) :: y(*)
        call gelu_forward(n, x, y)
    end subroutine c_gelu_forward

    !========== Convolution ==========
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
    subroutine c_layernorm_forward(batch, dim, x, gamma, beta, eps, y) &
        bind(c, name="c_layernorm_forward")
        integer(c_int), value :: batch, dim
        real(c_double), intent(in) :: x(dim, batch)
        real(c_double), intent(in) :: gamma(dim), beta(dim)
        real(c_double), value :: eps
        real(c_double), intent(out) :: y(dim, batch)
        integer :: b, i
        real(c_double) :: sx, sx2, mean_val, var_val, inv_std, inv_dim
        inv_dim = 1.0_c_double / dim

        ! ★ 2-pass: 融合 sum+sum_sq 减少内存遍历
        !$omp parallel do private(b, i, sx, sx2, mean_val, var_val, inv_std) schedule(static)
        do b = 1, batch
            ! Pass 1: sum + sum_sq (fused, 1 memory traversal)
            sx = 0.0_c_double
            sx2 = 0.0_c_double
            !$omp simd reduction(+:sx, sx2)
            do i = 1, dim
                sx = sx + x(i, b)
                sx2 = sx2 + x(i, b) * x(i, b)
            end do
            mean_val = sx * inv_dim
            var_val = sx2 * inv_dim - mean_val * mean_val
            inv_std = 1.0_c_double / sqrt(var_val + eps)

            ! Pass 2: normalize + affine
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
