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
    ! ★ simple_gemm — Fortran built-in matmul() wrapper
    !   One line of code, compiler auto AVX-512 FMA.
    !   Compare with hand-tiled six-nested GEMM, showcasing "compiler is smarter than you" and "you can be smarter than the compiler"
    !--------------------------------------------------------------------------
    subroutine c_simple_gemm(m, n, k, a, b, c) bind(c, name="c_simple_gemm")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: a(m, k), b(k, n)
        real(c_double), intent(out) :: c(m, n)
        c = matmul(a, b)
    end subroutine c_simple_gemm

    !--------------------------------------------------------------------------
    ! ★★ Zero-copy GEMM — directly receives row-major data, using transpose identity
    !   C(m,n) = A(m,k) @ B(k,n)       [row-major]
    !   C^T(n,m) = B^T(n,k) @ A^T(k,m) [column-major]
    !   Row-major memory == column-major transposed memory. Pass pointers directly!
    !   a declared as (k,m) → Fortran sees A^T
    !   b declared as (n,k) → Fortran sees B^T
    !   c declared as (n,m) → Fortran writes C^T = row-major C
    !--------------------------------------------------------------------------
    !--------------------------------------------------------------------------
    ! ★★★ Round-6 GEMM: 2D tiled parallel + matmul() built-in (compiler-level SIMD)
    !   Strategy:
    !     - Outer 2D tiled parallel (ii in m, jj in n), each thread handles one (m-tile, n-tile)
    !     - Inner uses matmul() to call built-in BLAS-quality GEMM (gfortran compiles to AVX2 FMA)
    !     - Key: matmul(a,b) where a(k,n_tile), b(k,m_tile) is row-contiguous!
    !       Because weight^T column-major = weight(m,k) row-contiguous in memory, matmul internally unrolls into
    !       stride-1 access on a's k dimension + stride-1 access on b's k dimension → SIMD-friendly
    !--------------------------------------------------------------------------
    subroutine c_rowmajor_gemm(m, n, k, a, b, c) bind(c, name="c_rowmajor_gemm")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: a(k, m)   ! A^T col-major = A(m,k) row-contiguous
        real(c_double), intent(in) :: b(n, k)   ! B^T col-major = B(k,n) row-contiguous
        real(c_double), intent(out) :: c(n, m)  ! C^T col-major = C(m,n) row-contiguous
        integer :: ii, jj, im, jm
        integer, parameter :: MB = 96, NB = 128  ! unused

        call scipy_dgemm(m, n, k, a, b, c)
    end subroutine c_rowmajor_gemm

    !--------------------------------------------------------------------------
    ! ★★ Zero-copy Fused Linear+ReLU
    !   weight(m,k) row-major → weight^T(k,m) column-major
    !   x(k,n) row-major → x^T(n,k) column-major
    !   y(m,n) row-major → y^T(n,m) column-major
    !   y = relu(W @ x + bias)
    !   y^T(:,j) = x^T @ W^T(:,j) + bias(j)  (each column j = each row j-1)
    !--------------------------------------------------------------------------
    subroutine c_rowmajor_fused_linear_relu(m, n, k, weight, bias, x, y) &
        bind(c, name="c_rowmajor_fused_linear_relu")
        integer(c_int), value :: m, n, k
        real(c_double), intent(in) :: weight(k, m)  ! W^T col-major
        real(c_double), intent(in) :: bias(m)
        real(c_double), intent(in) :: x(n, k)        ! X^T col-major
        real(c_double), intent(out) :: y(n, m)       ! Y^T col-major = Y row-major
        integer :: i, j
        ! Strategy: parallel matmul first, then fused bias + ReLU
        ! matmul is already BLAS-optimized, converted to efficient SIMD by gfortran
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
    ! ★★ Zero-copy Conv2D — im2col + matmul strategy
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
        col = 0.0_c_double  ! ★ Must zero: padding positions are not written

        ! ── im2col: adaptive OpenMP ──
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

        ! ★ 2-pass: fused sum+sum_sq reduces memory traversal
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