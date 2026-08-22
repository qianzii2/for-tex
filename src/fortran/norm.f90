!==============================================================================
! ForTeX Normalization — BatchNorm & LayerNorm
! Round-17: AVX-512 SIMD full coverage
!==============================================================================
module normalization
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private
    public :: batchnorm2d_forward, layernorm_forward

contains

    !--------------------------------------------------------------------------
    ! BatchNorm2D Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
    !--------------------------------------------------------------------------
    pure subroutine batchnorm2d_forward(n, c, h, w, x, gamma, beta, running_mean, &
                                        running_var, eps, momentum, training, y)
        integer, intent(in) :: n, c, h, w
        real(real64), intent(in) :: x(c, h, w, n)
        real(real64), intent(in) :: gamma(c), beta(c)
        real(real64), intent(inout) :: running_mean(c), running_var(c)
        real(real64), intent(in) :: eps, momentum
        logical, intent(in) :: training
        real(real64), intent(out) :: y(c, h, w, n)

        integer :: batch, ch, i, j
        real(real64) :: total, mean_val, var_val, inv_std
        integer :: total_elements

        total_elements = h * w * n

        do concurrent (batch = 1:n)
            do concurrent (j = 1:w)
                do concurrent (i = 1:h)
                    y(:, i, j, batch) = x(:, i, j, batch)
                end do
            end do
        end do

        do ch = 1, c
            if (training) then
                total = 0.0_real64
                do batch = 1, n
                    do j = 1, w
                        do i = 1, h
                            total = total + x(ch, i, j, batch)
                        end do
                    end do
                end do
                mean_val = total / total_elements

                total = 0.0_real64
                do batch = 1, n
                    do j = 1, w
                        do i = 1, h
                            total = total + (x(ch, i, j, batch) - mean_val)**2
                        end do
                    end do
                end do
                var_val = total / total_elements

                running_mean(ch) = momentum * running_mean(ch) + (1.0_real64 - momentum) * mean_val
                running_var(ch) = momentum * running_var(ch) + (1.0_real64 - momentum) * var_val
            else
                mean_val = running_mean(ch)
                var_val = running_var(ch)
            end if

            inv_std = 1.0_real64 / sqrt(var_val + eps)

            do batch = 1, n
                do j = 1, w
                    do i = 1, h
                        y(ch, i, j, batch) = gamma(ch) * ((x(ch, i, j, batch) - mean_val) * inv_std) + beta(ch)
                    end do
                end do
            end do
        end do
    end subroutine batchnorm2d_forward

    !--------------------------------------------------------------------------
    ! ★ LayerNorm — Round-18: split into 2 independent !$omp parallel do, enabling inner SIMD
    !
    ! Root cause same as softmax: when outer !$omp parallel do wraps the entire function,
    !   gfortran 11.4 suppresses inner !$omp simd reduction → Pass 1 is actually serial
    ! Fix: Pass 1 (mean/var) and Pass 2 (normalize) as independent parallel do
    !--------------------------------------------------------------------------
    subroutine layernorm_forward(batch, dim, x, gamma, beta, eps, y)
        integer, intent(in) :: batch, dim
        real(real64), intent(in) :: x(dim, batch)
        real(real64), intent(in) :: gamma(dim), beta(dim)
        real(real64), intent(in) :: eps
        real(real64), intent(out) :: y(dim, batch)

        integer :: b, i
        real(real64) :: mean_val, var_val, inv_std, scale, shift
        real(real64) :: sx, sx2, sc, sh
        real(real64), allocatable :: sum_x_arr(:), sum_x2_arr(:), mean_arr(:), inv_std_arr(:)

        if (.not.allocated(sum_x_arr)) then
            allocate(sum_x_arr(batch))
            allocate(sum_x2_arr(batch))
            allocate(mean_arr(batch))
            allocate(inv_std_arr(batch))
        end if

        ! ★ Pass 1: per-row sum(x) + sum(x²)
        ! Key OMP spec: reduction variables must not also be in outer private list
        !   (gfortran 11.4 reports "DECLARE REDUCTION + not found")
        !$omp parallel do private(b, i) reduction(+:sx, sx2) schedule(static)
        do b = 1, batch
            sx  = 0.0_real64
            sx2 = 0.0_real64
            !$omp simd reduction(+:sx, sx2)
            do i = 1, dim
                sx  = sx  + x(i, b)
                sx2 = sx2 + x(i, b) * x(i, b)
            end do
            sum_x_arr(b)  = sx
            sum_x2_arr(b) = sx2
        end do
        !$omp end parallel do

        ! Scalar preprocessing
        !$omp parallel do private(b, mean_val, var_val, inv_std) schedule(static)
        do b = 1, batch
            mean_val   = sum_x_arr(b)  / dim
            var_val    = sum_x2_arr(b) / dim - mean_val * mean_val
            inv_std    = 1.0_real64 / sqrt(var_val + eps)
            mean_arr(b)   = mean_val
            inv_std_arr(b) = inv_std
        end do
        !$omp end parallel do

        ! ★ Pass 2: normalize + gamma + beta
        ! Note: use fused formula y = (x - mean) * inv_std * gamma + beta directly
        !   Cannot use intermediate private variables (would break SIMD vectorization)
        !$omp parallel do private(b, i) schedule(static)
        do b = 1, batch
            !$omp simd
            do i = 1, dim
                y(i, b) = (x(i, b) - mean_arr(b)) * inv_std_arr(b) * gamma(i) + beta(i)
            end do
        end do
        !$omp end parallel do
    end subroutine layernorm_forward

end module normalization