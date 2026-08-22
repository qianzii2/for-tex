!==============================================================================
! ForTeX Loss Functions — MSE & CrossEntropy
!==============================================================================
module loss
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private
    public :: mse_loss, cross_entropy_loss

contains

    !--------------------------------------------------------------------------
    ! MSE Loss = mean((y_pred - y_true)^2)
    !--------------------------------------------------------------------------
    pure real(real64) function mse_loss(n, y_pred, y_true) result(loss_val)
        integer, intent(in) :: n
        real(real64), intent(in) :: y_pred(*), y_true(*)
        integer :: i

        loss_val = 0.0_real64
        do concurrent (i = 1:n)
            loss_val = loss_val + (y_pred(i) - y_true(i))**2
        end do
        loss_val = loss_val / real(n, real64)
    end function mse_loss

    !--------------------------------------------------------------------------
    ! Cross Entropy Loss (带 softmax)
    ! logits: (num_classes, batch) — 列主序: 每列是一个样本的各类别 logits
    ! targets: (batch) — 整数标签，1-based
    !--------------------------------------------------------------------------
    pure real(real64) function cross_entropy_loss(num_classes, batch, logits, targets) result(loss_val)
        integer, intent(in) :: num_classes, batch
        real(real64), intent(in) :: logits(num_classes, batch)
        integer, intent(in) :: targets(batch)

        real(real64) :: max_val, sum_exp
        integer :: i, b

        loss_val = 0.0_real64

        do b = 1, batch
            ! Softmax — 先找最大值防止溢出
            max_val = logits(1, b)
            do i = 2, num_classes
                if (logits(i, b) > max_val) max_val = logits(i, b)
            end do

            sum_exp = 0.0_real64
            do i = 1, num_classes
                sum_exp = sum_exp + exp(logits(i, b) - max_val)
            end do

            ! NLL of the correct class
            loss_val = loss_val - (logits(targets(b), b) - max_val - log(sum_exp))
        end do

        loss_val = loss_val / real(batch, real64)
    end function cross_entropy_loss

end module loss
