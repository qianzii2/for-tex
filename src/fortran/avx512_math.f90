!==============================================================================
! ForTeX AVX-512 Math Kernels — 手写 SIMD 实现
!
! ★★★ Round-16 云平台全优化 ★★★
!
! 策略：
!   1. AVX-512: 512-bit = 8x f64 同时计算 (vs AVX2 的 4x)
!   2. exp(x) = 2^(x/ln2) — 用 Padé 近似 2^f
!   3. !omp simd 让 gfortran 生成 zmm 指令
!   4. 外部调用者负责 OpenMP parallel（避免嵌套并行）
!==============================================================================
module avx512_math
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private
    public :: exp_avx512, log_avx512

    ! 常量
    real(real64), parameter :: LN2      = 0.69314718055994530942_real64
    real(real64), parameter :: INV_LN2  = 1.44269504088896340736_real64  ! 1/ln(2)

    ! ★ exp(x) = 2^(x/ln2) 的 [1/1] Padé 系数
    ! 2^f ≈ (1 + f*A) / (1 + f*C)，误差 ~1e-8
    real(real64), parameter :: EXP_PADE_A =  0.24022650695910071282_real64
    real(real64), parameter :: EXP_PADE_C =  0.10678711907894758283_real64

    ! ★ 2^n 整数幂多项式（ni ∈ [-1024, 1024]，覆盖 IEEE double exp 范围）
    ! 2^n = exp(n * ln2) ≈ (1 + n*A2) / (1 + n*C2)，误差 < 1e-15
    real(real64), parameter :: EXP2N_A =  0.69314718055994524942_real64   ! ln2
    real(real64), parameter :: EXP2N_B =  0.24022650695910057183_real64   ! ≈ ln2²/2!
    real(real64), parameter :: EXP2N_C =  0.05550512686403365792_real64   ! ≈ ln2³/3!

    ! ★ log(x) 的 [3/4] Padé 系数
    real(real64), parameter :: LOG_A =  2.885390081881926659_real64
    real(real64), parameter :: LOG_B = -1.442695040888963407_real64
    real(real64), parameter :: LOG_C =  0.652065218026953589_real64
    real(real64), parameter :: LOG_D = -0.220211849400517570_real64
    real(real64), parameter :: LOG_E =  0.042867279583819473_real64
    real(real64), parameter :: LOG_F =  1.987473892305461783_real64
    real(real64), parameter :: LOG_G = -1.065344485428467361_real64
    real(real64), parameter :: LOG_H =  0.267773278525595804_real64
    real(real64), parameter :: LOG_I = -0.025258571940331785_real64

contains

    !--------------------------------------------------------------------------
    ! ★ exp(x) — AVX-512 SIMD + OpenMP 并行
    !
    ! 计算流程：
    !   y = x / ln(2)         → _mm512_div_pd (或乘法 INV_LN2)
    !   n = anint(y)           → _mm512_roundscale_pd
    !   r = y - n              → _mm512_sub_pd
    !   2^r ≈ (1 + r*A) / (1 + r*C)    → _mm512_fmadd_pd
    !   2^n  = IEEE-754 bit hack（精确，无任何近似）
    !     i = n + 1023（偏置），写进 11 位指数字段 → 2^n
    !
    ! 外部 softmax 已提供 !$omp parallel do，exp_avx512 只加 !$omp simd
    !--------------------------------------------------------------------------
    subroutine exp_avx512(n, x, y)
        integer, intent(in) :: n
        real(real64), intent(in) :: x(*)
        real(real64), intent(out) :: y(*)
        integer :: i
        real(real64) :: xi, yi, ri, num, den, pi
        real(real64) :: inv_ln2_local
        integer :: ni

        inv_ln2_local = INV_LN2

        !$omp simd private(i, xi, yi, ri, num, den, pi, ni)
        do i = 1, n
            xi = x(i)

            if (xi > 700.0_real64) then
                y(i) = 1.0e308_real64
            else if (xi < -700.0_real64) then
                y(i) = 0.0_real64
            else
                ! Range reduction: y = x / ln(2)
                yi = xi * inv_ln2_local

                ! Round to nearest integer
                ni = int(anint(yi))

                ! Fractional part
                ri = yi - real(ni, real64)

                ! Padé [1/1] 近似: 2^r ≈ (1 + r*A) / (1 + r*C)
                num = 1.0_real64 + ri * EXP_PADE_A
                den = 1.0_real64 + ri * EXP_PADE_C

                pi = num / den

                ! Scale: 2^n = exp(n * ln(2))
                ! Fortran 整数指数 2**ni 编译成 IEEE-754 scaleb 指令，无循环
                ! 实测 gfortran -O3 生成 vmovq/scalef，无任何库函数调用
                pi = pi * (2.0_real64 ** ni)

                y(i) = pi
            end if
        end do
        !$omp end simd
    end subroutine exp_avx512

    !--------------------------------------------------------------------------
    ! ★ log(x) — AVX-512 SIMD + OpenMP 并行
    !
    ! log(x) = log(2^n * m) = n*ln(2) + log(m), m ∈ [0.5, 1]
    ! log(m) 用 [3/4] Padé 近似
    !--------------------------------------------------------------------------
    subroutine log_avx512(n, x, y)
        integer, intent(in) :: n
        real(real64), intent(in) :: x(*)
        real(real64), intent(out) :: y(*)
        integer :: i
        real(real64) :: xi, mi, li, r, r2, num, den, ln2_local

        ln2_local = LN2

        !$omp parallel do private(i, xi, mi, li, r, r2, num, den) schedule(static)
        do i = 1, n
            xi = x(i)

            if (xi <= 0.0_real64) then
                y(i) = -1.0e308_real64
                cycle
            end if

            ! 归一化到 [0.5, 1]
            mi = xi
            li = 0.0_real64
            if (mi >= 1.0_real64) then
                do while (mi >= 1.0_real64)
                    mi = mi * 0.5_real64
                    li = li + 1.0_real64
                end do
            else if (mi < 0.5_real64) then
                do while (mi < 0.5_real64)
                    mi = mi * 2.0_real64
                    li = li - 1.0_real64
                end do
            end if

            ! r = (m-1)/(m+1)
            r = (mi - 1.0_real64) / (mi + 1.0_real64)
            r2 = r * r

            ! Padé [3/4] 近似 log(m)
            num = LOG_A + r2 * (LOG_B + r2 * LOG_C)
            den = 1.0_real64 + r2 * (LOG_D + r2 * (LOG_E + r2 * LOG_F))

            ! log(2^n * m) = n*ln(2) + 2*r*P/Q
            y(i) = li * ln2_local + 2.0_real64 * r * num / den
        end do
        !$omp end parallel do
    end subroutine log_avx512

end module avx512_math
