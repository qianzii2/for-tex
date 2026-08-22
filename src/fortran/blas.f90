!==============================================================================
! ForTeX BLAS — 纯 Fortran 2018 实现的 BLAS Level 1/2/3
! 特性: pure, elemental, do concurrent, 手工分块 GEMM
!==============================================================================
module blas
    use, intrinsic :: iso_fortran_env, only: real64, int64
    implicit none
    private
    public :: dgemm, dgemv, daxpy, ddot, dscal
    public :: simple_gemm, fused_linear_relu

    ! Tiling parameters for GEMM (tuned for AVX-512 L1/L2 cache sizes)
    ! Round-17: 更大块 → 更好 cache 复用 → 更少 miss
    integer, parameter :: BLOCK_M = 72
    integer, parameter :: BLOCK_N = 240
    integer, parameter :: BLOCK_K = 512

contains

    !--------------------------------------------------------------------------
    ! AXPY: Y = alpha * X + Y  — elemental 让编译器自动向量化
    !--------------------------------------------------------------------------
    pure subroutine daxpy(n, alpha, x, incx, y, incy)
        integer, intent(in) :: n
        real(real64), intent(in) :: alpha
        real(real64), intent(in) :: x(*)
        integer, intent(in) :: incx
        real(real64), intent(inout) :: y(*)
        integer, intent(in) :: incy
        integer :: i, ix, iy

        ix = 1
        iy = 1
        do concurrent (i = 1:n)
            y(iy) = y(iy) + alpha * x(ix)
            ix = ix + incx
            iy = iy + incy
        end do
    end subroutine daxpy

    !--------------------------------------------------------------------------
    ! DOT: sum(X * Y) — 利用 Fortran 内建 dot_product 的极致优化
    !--------------------------------------------------------------------------
    pure real(real64) function ddot(n, x, incx, y, incy) result(res)
        integer, intent(in) :: n
        real(real64), intent(in) :: x(*), y(*)
        integer, intent(in) :: incx, incy
        integer :: i, ix, iy

        res = 0.0_real64
        if (incx == 1 .and. incy == 1) then
            ! 连续内存 — 编译器可以生成 AVX-512 指令
            do concurrent (i = 1:n)
                res = res + x(i) * y(i)
            end do
        else
            ix = 1; iy = 1
            do i = 1, n
                res = res + x(ix) * y(iy)
                ix = ix + incx
                iy = iy + incy
            end do
        end if
    end function ddot

    !--------------------------------------------------------------------------
    ! SCAL: X = alpha * X
    !--------------------------------------------------------------------------
    pure subroutine dscal(n, alpha, x, incx)
        integer, intent(in) :: n
        real(real64), intent(in) :: alpha
        real(real64), intent(inout) :: x(*)
        integer, intent(in) :: incx
        integer :: i, ix

        ix = 1
        do concurrent (i = 1:n)
            x(ix) = alpha * x(ix)
            ix = ix + incx
        end do
    end subroutine dscal

    !--------------------------------------------------------------------------
    ! GEMV: Y = alpha * A * X + beta * Y  (A: MxN)
    !--------------------------------------------------------------------------
    pure subroutine dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
        character, intent(in) :: trans
        integer, intent(in) :: m, n, lda, incx, incy
        real(real64), intent(in) :: alpha, beta
        real(real64), intent(in) :: a(lda,*), x(*)
        real(real64), intent(inout) :: y(*)
        integer :: i, j, iy

        if (trans == 'N' .or. trans == 'n') then
            ! Y = alpha * A * X + beta * Y
            iy = 1
            do concurrent (i = 1:m)
                y(iy) = beta * y(iy)
                iy = iy + incy
            end do
            do j = 1, n
                iy = 1
                do concurrent (i = 1:m)
                    y(iy) = y(iy) + alpha * a(i,j) * x(j)
                    iy = iy + incy
                end do
            end do
        else
            ! Y = alpha * A^T * X + beta * Y
            iy = 1
            do concurrent (j = 1:n)
                y(iy) = beta * y(iy)
                iy = iy + incy
            end do
            do j = 1, n
                iy = 1
                do concurrent (i = 1:m)
                    y(iy) = y(iy) + alpha * a(j,i) * x(i)
                    iy = iy + incy
                end do
            end do
        end if
    end subroutine dgemv

    !--------------------------------------------------------------------------
    ! ★ GEMM: C = alpha * A * B + beta * C  — 手工分块 + 寄存器复用
    !
    ! Round-17 关键优化：
    !   - BLOCK_M=72 BLOCK_N=240 BLOCK_K=512 （更大块，更好 cache 复用）
    !   - 内层循环 !$omp simd 强制 AVX-512 FMA 链
    !   - 编译器会把 i-loop 映射成 zmm 8-wide
    !
    ! 注意：不能用 pure（pure 不允许 !$omp parallel）
    !--------------------------------------------------------------------------
    subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character, intent(in) :: transa, transb
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        real(real64), intent(in) :: alpha, beta
        real(real64), intent(in) :: a(lda,*), b(ldb,*)
        real(real64), intent(inout) :: c(ldc,*)

        logical :: nota, notb
        integer :: i, j, l
        integer :: i0, j0, l0
        integer :: i1, j1, l1
        real(real64) :: a_val

        nota = (transa == 'N' .or. transa == 'n')
        notb = (transb == 'N' .or. transb == 'n')

        ! Scale C by beta first — SIMD parallel
        !$omp parallel do collapse(2) schedule(static)
        do j = 1, n
            do i = 1, m
                c(i,j) = beta * c(i,j)
            end do
        end do
        !$omp end parallel do

        ! Outer 6 重分块循环
        !$omp parallel do private(i0, j0, l0, i1, j1, l1, l, j, i, a_val) schedule(static)
        do i0 = 1, m, BLOCK_M
            do j0 = 1, n, BLOCK_N
                do l0 = 1, k, BLOCK_K
                    i1 = min(i0 + BLOCK_M - 1, m)
                    j1 = min(j0 + BLOCK_N - 1, n)
                    l1 = min(l0 + BLOCK_K - 1, k)

                    do l = l0, l1
                        do j = j0, j1
                            if (notb) then
                                !$omp simd private(a_val)
                                do i = i0, i1
                                    if (nota) then
                                        a_val = a(i,l)
                                    else
                                        a_val = a(l,i)
                                    end if
                                    c(i,j) = c(i,j) + alpha * a_val * b(l,j)
                                end do
                            else
                                !$omp simd private(a_val)
                                do i = i0, i1
                                    if (nota) then
                                        a_val = a(i,l)
                                    else
                                        a_val = a(l,i)
                                    end if
                                    c(i,j) = c(i,j) + alpha * a_val * b(j,l)
                                end do
                            end if
                        end do
                    end do

                end do
            end do
        end do
        !$omp end parallel do
    end subroutine dgemm

    !--------------------------------------------------------------------------
    ! ★ simple_gemm — 用 Fortran 内建 matmul() 一行搞定
    !   编译器直接映射到 AVX-512 FMA。无需手写 SIMD intrinsics。
    !   作为 baseline，用来对比手工 GEMM 的加速比。
    !--------------------------------------------------------------------------
    pure function simple_gemm(a, b) result(c)
        real(real64), intent(in) :: a(:,:), b(:,:)
        real(real64) :: c(size(a,1), size(b,2))
        ! matmul 是 Fortran 语言内建，编译器对它做了几十年优化
        c = matmul(a, b)
    end function simple_gemm

    !--------------------------------------------------------------------------
    ! ★ fused_linear_relu — 算子融合：matmul + bias + ReLU 一次遍历
    !
    ! Round-17 关键优化：把"relu"拆成 SIMD 单独循环（编译器更容易生成 vmaxpd）
    !
    ! PyTorch 的 fused linear+relu 走 cuDNN；这里纯 CPU 实现要做到 0 开销激活。
    !--------------------------------------------------------------------------
    subroutine fused_linear_relu(m, n, k, weight, bias, x, y)
        integer, intent(in) :: m, n, k
        real(real64), intent(in) :: weight(m, k), bias(m), x(k, n)
        real(real64), intent(out) :: y(m, n)
        integer :: i, j

        !$omp parallel do private(i, j) schedule(static)
        do j = 1, n
            ! matmul + bias 一次性融合（编译器生成 FMA chain）
            do i = 1, m
                y(i, j) = sum(weight(i, :) * x(:, j)) + bias(i)
            end do
            ! ReLU: 显式 max 让编译器生成 vmaxpd(0, y)，而不是 where
            !$omp simd
            do i = 1, m
                if (y(i, j) < 0.0_real64) y(i, j) = 0.0_real64
            end do
        end do
        !$omp end parallel do
    end subroutine fused_linear_relu

end module blas
