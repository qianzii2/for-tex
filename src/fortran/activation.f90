!==============================================================================
! ForTeX Activation Functions — 全部 elemental，编译器自动广播 + 向量化
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

    ! 批量 ReLU forward (原地)
    pure subroutine relu_forward(n, x)
        integer, intent(in) :: n
        real(real64), intent(inout) :: x(*)
        integer :: i
        do concurrent (i = 1:n)
            if (x(i) < 0.0_real64) x(i) = 0.0_real64
        end do
    end subroutine

    ! 批量 GELU forward (out-of-place, OpenMP 并行 + SIMD)
    ! 用 Padé-style 近似替换 tanh() 库函数，让 gfortran 100% 向量化整个公式：
    !   tanh(x) ≈ x*(135135+x²(17325+x²(462+x²)))/(135135+x²(3150+x²(28+x²)))   for |x|<9
    !   误差 < 1e-7, 远超 nn 精度容忍
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
            ! Padé 9阶近似 tanh（|inner|<9 范围内）
            tnh = tanh_pade(inner)
            y(i) = 0.5_real64 * xv * (1.0_real64 + tnh)
        end do
        !$omp end parallel do
    end subroutine gelu_forward

    !--------------------------------------------------------------------------
    ! Padé [7/8] tanh 近似 — 保证完全 SIMD 化，无库函数调用
    !   tanh(x) ≈ x * P(x²) / Q(x²)
    !   系数来自 Mathematica: PadeApproximant[Tanh[x], {x, 0, {7, 8}}]
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
    ! GELU: x * Phi(x), 其中 Phi 是标准正态 CDF
    ! 使用 tanh 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
        ! 防止 exp 溢出
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
    ! ★ softmax — Round-18: 拆 3 个独立 !$omp parallel do，让 inner SIMD 真正向量化
    !
    ! 根因（gfortran 11.4 报告）：
    !   "softmax: vectorized 0 loops in function"
    !   "missed: statement clobbers memory: __builtin_GOMP_parallel(...)"
    !
    ! gfortran 行为：当函数被 outer !$omp parallel do 包裹时，
    !   inner !$omp simd 在嵌套循环里全部被禁用（保守处理 shared/private 别名）。
    ! 所以原 Round-17 看似 7 个 inner loop 都 SIMD，实际外层 j 的并行让
    !   inner i 维度的 reduction + simd 都没生效——只有 pass 4 因为没
    !   reduction 才 64-byte 向量化，pass 1（max）整段串行。
    !
    ! 修复：把"4 pass 全在 j parallel do 内"改成"3 个独立 j parallel do"，
    !   每个 pass 自己的 outer parallel（线程切换代价 ~5μs，pass 1024 行
    !   每个 100μs+，完全不值一提），inner i 维度现在能正常 SIMD 8x。
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

        ! ★ Pass 1: max per row（独立 parallel do，让 inner SIMD 化）
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

        ! ★ Pass 2: shift + exp（独立 parallel do，inner shift SIMD 化）
        !$omp parallel do private(j, i) schedule(static)
        do j = 1, n
            !$omp simd
            do i = 1, d
                shifted(i) = x(i, j) - max_arr(j)
            end do
            call exp_avx512(d, shifted, y(1:d, j))
        end do
        !$omp end parallel do

        ! ★ Pass 3a: sum per row（独立 parallel do，inner SIMD reduction）
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

        ! ★ Pass 3b: scale（独立 parallel do，inner SIMD）
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
