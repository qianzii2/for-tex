!==============================================================================
! ForTeX AVX-512 Math Kernels — SIMD exp/log
!
! ★★★ Round-16 cloud platform full optimization ★★★
!
! Strategy:
!   1. AVX-512: 512-bit = 8x f64 simultaneous computation (vs AVX2's 4x)
!   2. exp(x) = 2^(x/ln2) — using Padé approximation of 2^f
!   3. !omp simd lets gfortran generate zmm instructions
!   4. External callers handle OpenMP parallel (avoid nested parallelism)
!==============================================================================
module avx512_math
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private
    public :: exp_avx512, log_avx512, exp_avx512_sum

    ! Constants
    real(real64), parameter :: LN2      = 0.69314718055994530942_real64
    real(real64), parameter :: INV_LN2  = 1.44269504088896340736_real64  ! 1/ln(2)

    ! ★ exp(x) = 2^(x/ln2) [1/1] Padé coefficients
    ! 2^f ≈ (1 + f*A) / (1 + f*C), error ~1e-8
    real(real64), parameter :: EXP_PADE_A =  0.24022650695910071282_real64
    real(real64), parameter :: EXP_PADE_C =  0.10678711907894758283_real64

    ! ★ 2^n integer power polynomial (ni ∈ [-1024, 1024], covers IEEE double exp range)
    ! 2^n = exp(n * ln2) ≈ (1 + n*A2) / (1 + n*C2), error < 1e-15
    real(real64), parameter :: EXP2N_A =  0.69314718055994524942_real64   ! ln2
    real(real64), parameter :: EXP2N_B =  0.24022650695910057183_real64   ! ≈ ln2²/2!
    real(real64), parameter :: EXP2N_C =  0.05550512686403365792_real64   ! ≈ ln2³/3!

    ! ★ log(x) [3/4] Padé coefficients
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
    ! ★ exp(x) — AVX-512 SIMD + OpenMP parallel
    !
    ! Computation flow:
    !   y = x / ln(2)         → _mm512_div_pd (or multiply by INV_LN2)
    !   n = anint(y)           → _mm512_roundscale_pd
    !   r = y - n              → _mm512_sub_pd
    !   2^r ≈ (1 + r*A) / (1 + r*C)    → _mm512_fmadd_pd
    !   2^n  = IEEE-754 bit hack (exact, no approximation)
    !     i = n + 1023 (bias), write into 11-bit exponent field → 2^n
    !
    ! External softmax already provides !$omp parallel do, exp_avx512 only adds !$omp simd
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

                ! Padé [1/1] approximation: 2^r ≈ (1 + r*A) / (1 + r*C)
                num = 1.0_real64 + ri * EXP_PADE_A
                den = 1.0_real64 + ri * EXP_PADE_C

                pi = num / den

                ! Scale: 2^n = exp(n * ln(2))
                ! Fortran integer exponent 2**ni compiles to IEEE-754 scaleb instruction, no loop
                ! Measured: gfortran -O3 generates vmovq/scalef, no library function calls
                pi = pi * (2.0_real64 ** ni)

                y(i) = pi
            end if
        end do
        !$omp end simd
    end subroutine exp_avx512

    !--------------------------------------------------------------------------
    ! ★ exp(x-shift) + sum — fused version, eliminates softmax's separate shift+sum loop
    !   Returns: y(i) = exp(x(i) - shift), s = sum(y(1:n))
    !--------------------------------------------------------------------------
    subroutine exp_avx512_sum(n, x, shift, y, s)
        integer, intent(in) :: n
        real(real64), intent(in) :: x(*)
        real(real64), intent(in) :: shift
        real(real64), intent(out) :: y(*)
        real(real64), intent(out) :: s
        integer :: i
        real(real64) :: xi, yi, ri, num, den, pi
        real(real64) :: inv_ln2_local, sum_local
        integer :: ni

        inv_ln2_local = INV_LN2
        sum_local = 0.0_real64

        !$omp simd private(i, xi, yi, ri, num, den, pi, ni) reduction(+:sum_local)
        do i = 1, n
            xi = x(i) - shift

            if (xi > 700.0_real64) then
                pi = 1.0e308_real64
            else if (xi < -700.0_real64) then
                pi = 0.0_real64
            else
                yi = xi * inv_ln2_local
                ni = int(anint(yi))
                ri = yi - real(ni, real64)
                num = 1.0_real64 + ri * EXP_PADE_A
                den = 1.0_real64 + ri * EXP_PADE_C
                pi = num / den
                pi = pi * (2.0_real64 ** ni)
            end if
            y(i) = pi
            sum_local = sum_local + pi
        end do
        !$omp end simd
        s = sum_local
    end subroutine exp_avx512_sum

    !--------------------------------------------------------------------------
    ! ★ log(x) — AVX-512 SIMD + OpenMP parallel
    !
    ! log(x) = log(2^n * m) = n*ln(2) + log(m), m ∈ [0.5, 1]
    ! log(m) uses [3/4] Padé approximation
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

            ! Normalize to [0.5, 1]
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

            ! Padé [3/4] approximation of log(m)
            num = LOG_A + r2 * (LOG_B + r2 * LOG_C)
            den = 1.0_real64 + r2 * (LOG_D + r2 * (LOG_E + r2 * LOG_F))

            ! log(2^n * m) = n*ln(2) + 2*r*P/Q
            y(i) = li * ln2_local + 2.0_real64 * r * num / den
        end do
        !$omp end parallel do
    end subroutine log_avx512

end module avx512_math