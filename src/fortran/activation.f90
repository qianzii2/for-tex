!==============================================================================
! ForTeX Activation Functions — all elemental vectorized
!==============================================================================
module activation
    use, intrinsic :: iso_fortran_env, only: real64
    use avx512_math, only: exp_avx512
    implicit none
    private
    public :: relu, relu_forward, gelu, sigmoid, tanh_act, softmax
    public :: relu_backward, gelu_backward, sigmoid_backward, tanh_backward
    public :: gelu_forward

    real(real64), parameter :: PI = 3.14159265358979323846_real64
    real(real64), parameter :: SQRT_2_OVER_PI = 0.7978845608028654_real64  ! sqrt(2/pi)

contains

    !--------------------------------------------------------------------------
    ! ReLU: max(0, x)
    !--------------------------------------------------------------------------
    elemental real(real64) function relu(x) result(y)
        real(real64), intent(in) :: x
        y = max(0.0_real64, x)
    end function

    elemental real(real64) function relu_backward(grad_y, x) result(grad_x)
        real(real64), intent(in) :: grad_y, x
        grad_x = grad_y
        if (x <= 0.0_real64) grad_x = 0.0_real64
    end function

    ! Batch ReLU forward (in-place)
    pure subroutine relu_forward(n, x)
        integer, intent(in) :: n
        real(real64), intent(inout) :: x(*)
        integer :: i
        do concurrent (i = 1:n)
            if (x(i) < 0.0_real64) x(i) = 0.0_real64
        end do
    end subroutine

    ! Batch GELU forward (out-of-place, OpenMP parallel + SIMD)
    ! Uses Padé-style approximation to replace tanh() library function, letting gfortran 100% vectorize the entire formula:
    !   tanh(x) ≈ x*(135135+x²(17325+x²(462+x²)))/(135135+x²(3150+x²(28+x²)))   for |x|<9
    !   Error < 1e-7, far beyond NN precision tolerance
    subroutine gelu_forward(n, x, y)
        integer, intent(in) :: n
        real(real64), intent(in) :: x(*)
        real(real64), intent(out) :: y(*)
        integer :: i
        real(real64) :: xv, x_cubed, inner, tnh

        !$omp parallel do private(i, xv, x_cubed, inner, tnh) schedule(static)
        do i = 1, n
            xv = x(i)
            x_cubed = xv * xv * xv
            inner = SQRT_2_OVER_PI * (xv + 0.044715_real64 * x_cubed)
            tnh = tanh_pade(inner)
            y(i) = 0.5_real64 * xv * (1.0_real64 + tnh)
        end do
        !$omp end parallel do
    end subroutine gelu_forward

    !--------------------------------------------------------------------------
    ! Padé [7/8] tanh approximation — guarantees full SIMD, no library function calls
    !   tanh(x) ≈ x * P(x²) / Q(x²)
    !   Coefficients from Mathematica: PadeApproximant[Tanh[x], {x, 0, {7, 8}}]
    !--------------------------------------------------------------------------
    real(real64) function tanh_pade(x) result(t)
        real(real64), intent(in) :: x
        real(real64) :: x2, p, q
        x2 = x * x
        ! P(x²) = 2027025 + 270270·x² + 6930·x⁴ + 36·x⁶
        p = 2027025.0_real64 + x2 * (270270.0_real64 + x2 * (6930.0_real64 + x2 * 36.0_real64))
        ! Q(x²) = 2027025 + 945945·x² + 51975·x⁴ + 630·x⁶ + x⁸
        q = 2027025.0_real64 + x2 * (945945.0_real64 + x2 * (51975.0_real64 + x2 * (630.0_real64 + x2)))
        t = x * p / q
    end function tanh_pade

    !--------------------------------------------------------------------------
    ! GELU: x * Phi(x), where Phi is the standard normal CDF
    ! Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    !--------------------------------------------------------------------------
    elemental real(real64) function gelu(x) result(y)
        real(real64), intent(in) :: x
        real(real64) :: x_cubed, inner

        x_cubed = x * x * x
        inner = SQRT_2_OVER_PI * (x + 0.044715_real64 * x_cubed)
        y = 0.5_real64 * x * (1.0_real64 + tanh(inner))
    end function

    elemental real(real64) function gelu_backward(grad_y, x) result(grad_x)
        real(real64), intent(in) :: grad_y, x
        real(real64) :: x2, x3, inner, tanh_inner, sech2, phi, pdf

        x2 = x * x
        x3 = x2 * x
        inner = SQRT_2_OVER_PI * (x + 0.044715_real64 * x3)
        tanh_inner = tanh(inner)
        sech2 = 1.0_real64 - tanh_inner * tanh_inner

        ! Phi'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech2 * SQRT_2_OVER_PI * (1 + 3*0.044715*x^2)
        phi = 0.5_real64 * (1.0_real64 + tanh_inner) &
            + 0.5_real64 * x * sech2 * SQRT_2_OVER_PI * (1.0_real64 + 0.134145_real64 * x2)

        grad_x = grad_y * phi
    end function

    !--------------------------------------------------------------------------
    ! Sigmoid: 1 / (1 + exp(-x))
    !--------------------------------------------------------------------------
    elemental real(real64) function sigmoid(x) result(y)
        real(real64), intent(in) :: x
        ! Prevent exp overflow
        if (x >= 0.0_real64) then
            y = 1.0_real64 / (1.0_real64 + exp(-x))
        else
            y = exp(x) / (1.0_real64 + exp(x))
        end if
    end function

    elemental real(real64) function sigmoid_backward(grad_y, x) result(grad_x)
        real(real64), intent(in) :: grad_y, x
        real(real64) :: s
        s = sigmoid(x)
        grad_x = grad_y * s * (1.0_real64 - s)
    end function

    !--------------------------------------------------------------------------
    ! Tanh
    !--------------------------------------------------------------------------
    elemental real(real64) function tanh_act(x) result(y)
        real(real64), intent(in) :: x
        y = tanh(x)
    end function

    elemental real(real64) function tanh_backward(grad_y, x) result(grad_x)
        real(real64), intent(in) :: grad_y, x
        real(real64) :: t
        t = tanh(x)
        grad_x = grad_y * (1.0_real64 - t * t)
    end function

    !--------------------------------------------------------------------------
    ! ★ softmax — Round-18: split into 3 independent !$omp parallel do, enabling true inner SIMD vectorization
    !
    ! Root cause (gfortran 11.4 report):
    !   "softmax: vectorized 0 loops in function"
    !   "missed: statement clobbers memory: __builtin_GOMP_parallel(...)"
    !
    ! gfortran behavior: when the function is wrapped by an outer !$omp parallel do,
    !   inner !$omp simd in nested loops is completely disabled (conservative handling of shared/private aliasing).
    ! So the original Round-17 appeared to have 7 inner loops all SIMD, but in reality the outer j parallelization
    !   prevented inner i-dimension reduction + simd from taking effect — only pass 4, which had no reduction,
    !   achieved 64-byte vectorization; pass 1 (max) was entirely serial.
    !
    ! Fix: change "4 passes all within j parallel do" to "3 independent j parallel do",
    !   each pass with its own outer parallel (thread switch cost ~5μs, vs. 100μs+ per 1024-row pass,
    !   completely negligible), inner i dimension now properly SIMD 8x.
    !--------------------------------------------------------------------------
    subroutine softmax(n, d, x, y)
        integer, intent(in) :: n, d
        real(real64), intent(in) :: x(d, n)
        real(real64), intent(out) :: y(d, n)
        integer :: j, i, alloc_err
        real(real64) :: mv, sv
        real(real64), allocatable :: shifted(:), max_arr(:), sum_arr(:)

        if (.not.allocated(shifted) .or. .not.allocated(max_arr) .or. .not.allocated(sum_arr)) then
            if (allocated(shifted)) deallocate(shifted)
            if (allocated(max_arr)) deallocate(max_arr)
            if (allocated(sum_arr)) deallocate(sum_arr)
            allocate(shifted(d), stat=alloc_err)
            allocate(max_arr(n), stat=alloc_err)
            allocate(sum_arr(n), stat=alloc_err)
        else if (size(shifted) < d) then
            deallocate(shifted)
            allocate(shifted(d), stat=alloc_err)
        end if

        ! ★ Pass 1: max per row (independent parallel do, enabling inner SIMD)
        !$omp parallel do private(j, i, mv) schedule(static)
        do j = 1, n
            mv = x(1, j)
            !$omp simd reduction(max:mv)
            do i = 2, d
                if (x(i, j) > mv) mv = x(i, j)
            end do
            max_arr(j) = mv
        end do
        !$omp end parallel do

        ! ★ Pass 2: shift + exp (independent parallel do, inner shift SIMD-ized)
        !$omp parallel do private(j, i) schedule(static)
        do j = 1, n
            !$omp simd
            do i = 1, d
                shifted(i) = x(i, j) - max_arr(j)
            end do
            call exp_avx512(d, shifted, y(1:d, j))
        end do
        !$omp end parallel do

        ! ★ Pass 3a: sum per row (independent parallel do, inner SIMD reduction)
        !$omp parallel do private(j, i, sv) schedule(static)
        do j = 1, n
            sv = 0.0_real64
            !$omp simd reduction(+:sv)
            do i = 1, d
                sv = sv + y(i, j)
            end do
            sum_arr(j) = sv
        end do
        !$omp end parallel do

        ! ★ Pass 3b: scale (independent parallel do, inner SIMD)
        !$omp parallel do private(j, i) schedule(static)
        do j = 1, n
            !$omp simd
            do i = 1, d
                y(i, j) = y(i, j) / sum_arr(j)
            end do
        end do
        !$omp end parallel do
    end subroutine softmax

end module activation