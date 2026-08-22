!==============================================================================
! ForTeX Convolution — im2col + matmul 策略
!==============================================================================
module convolution
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private
    public :: im2col, conv2d_forward, maxpool2d, avgpool2d

contains

    !--------------------------------------------------------------------------
    ! im2col: 将 4D 图像张量 (N, C, H, W) 展开为 2D 矩阵
    ! 输入: img(C, H, W, N)  — Fortran 列主序
    ! 输出: col(patch_size, num_patches)
    !--------------------------------------------------------------------------
    pure subroutine im2col(n, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, img, col)
        integer, intent(in) :: n, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w
        real(real64), intent(in) :: img(c, h, w, n)
        real(real64), intent(out) :: col(:,:)  ! (c*kh*kw, out_h*out_w*n)

        integer :: out_h, out_w
        integer :: batch, ch, i, j, ki, kj, in_i, in_j
        integer :: col_row, col_col

        out_h = (h + 2*pad_h - kh) / stride_h + 1
        out_w = (w + 2*pad_w - kw) / stride_w + 1

        col_row = c * kh * kw
        col_col = out_h * out_w * n

        ! 初始化
        do concurrent (j = 1:col_col)
            do concurrent (i = 1:col_row)
                col(i, j) = 0.0_real64
            end do
        end do

        do batch = 1, n
            do i = 1, out_h
                do j = 1, out_w
                    col_col = (batch - 1) * out_h * out_w + (i - 1) * out_w + j
                    do ch = 1, c
                        do ki = 1, kh
                            in_i = (i-1)*stride_h + ki - pad_h
                            if (in_i < 1 .or. in_i > h) cycle
                            do kj = 1, kw
                                in_j = (j-1)*stride_w + kj - pad_w
                                if (in_j < 1 .or. in_j > w) cycle
                                col_row = (ch - 1) * kh * kw + (ki - 1) * kw + kj
                                col(col_row, col_col) = img(ch, in_i, in_j, batch)
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end subroutine im2col

    !--------------------------------------------------------------------------
    ! Conv2D Forward: output = Conv(input, weight) + bias
    ! 输入:  img(C_in, H, W, N)
    ! 权重: weight(C_out, C_in, Kh, Kw)
    ! 输出: output(C_out, OutH, OutW, N)
    !--------------------------------------------------------------------------
    pure subroutine conv2d_forward(n, c_in, c_out, h, w, kh, kw, &
                                    stride_h, stride_w, pad_h, pad_w, &
                                    img, weight, bias, output)
        integer, intent(in) :: n, c_in, c_out, h, w, kh, kw
        integer, intent(in) :: stride_h, stride_w, pad_h, pad_w
        real(real64), intent(in) :: img(c_in, h, w, n)
        real(real64), intent(in) :: weight(c_out, c_in, kh, kw)
        real(real64), intent(in) :: bias(c_out)
        real(real64), intent(out) :: output(:,:,:,:)  ! (c_out, out_h, out_w, n)

        integer :: out_h, out_w, patch_size, num_patches
        real(real64), allocatable :: col(:,:), w_reshaped(:,:), out_reshaped(:,:)
        integer :: i, j, batch, col_idx

        out_h = (h + 2*pad_h - kh) / stride_h + 1
        out_w = (w + 2*pad_w - kw) / stride_w + 1
        patch_size = c_in * kh * kw
        num_patches = out_h * out_w * n

        allocate(col(patch_size, num_patches))
        allocate(w_reshaped(c_out, patch_size))
        allocate(out_reshaped(c_out, num_patches))

        ! im2col
        call im2col(n, c_in, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w, img, col)

        ! 将 weight(C_out, C_in, Kh, Kw) reshape 为 (C_out, patch_size)
        w_reshaped = reshape(weight, [c_out, patch_size])

        ! GEMM: out_reshaped = w_reshaped * col
        ! 用 matmul intrinsic（pure-friendly，编译器映射 BLAS）
        out_reshaped = matmul(w_reshaped, col)

        ! 加 bias + reshape 回 (C_out, OutH, OutW, N)
        do batch = 1, n
            do j = 1, out_w
                do i = 1, out_h
                    col_idx = (batch - 1) * out_h * out_w + (i - 1) * out_w + j
                    output(:, i, j, batch) = out_reshaped(:, col_idx) + bias(:)
                end do
            end do
        end do

        deallocate(col, w_reshaped, out_reshaped)
    end subroutine conv2d_forward

    !--------------------------------------------------------------------------
    ! MaxPool2D
    !--------------------------------------------------------------------------
    pure subroutine maxpool2d(n, c, h, w, kh, kw, stride_h, stride_w, input, output)
        integer, intent(in) :: n, c, h, w, kh, kw, stride_h, stride_w
        real(real64), intent(in) :: input(c, h, w, n)
        real(real64), intent(out) :: output(:,:,:,:)  ! (c, out_h, out_w, n)

        integer :: out_h, out_w
        integer :: batch, ch, i, j, ki, kj, in_i, in_j
        real(real64) :: max_val

        out_h = (h - kh) / stride_h + 1
        out_w = (w - kw) / stride_w + 1

        do batch = 1, n
            do ch = 1, c
                do i = 1, out_h
                    do j = 1, out_w
                        in_i = (i - 1) * stride_h + 1
                        in_j = (j - 1) * stride_w + 1
                        max_val = input(ch, in_i, in_j, batch)
                        do ki = 1, kh
                            do kj = 1, kw
                                max_val = max(max_val, input(ch, in_i+ki-1, in_j+kj-1, batch))
                            end do
                        end do
                        output(ch, i, j, batch) = max_val
                    end do
                end do
            end do
        end do
    end subroutine maxpool2d

    !--------------------------------------------------------------------------
    ! AvgPool2D
    !--------------------------------------------------------------------------
    pure subroutine avgpool2d(n, c, h, w, kh, kw, stride_h, stride_w, input, output)
        integer, intent(in) :: n, c, h, w, kh, kw, stride_h, stride_w
        real(real64), intent(in) :: input(c, h, w, n)
        real(real64), intent(out) :: output(:,:,:,:)  ! (c, out_h, out_w, n)

        integer :: out_h, out_w
        integer :: batch, ch, i, j, ki, kj, in_i, in_j
        real(real64) :: sum_val, inv_area

        inv_area = 1.0_real64 / real(kh * kw, real64)
        out_h = (h - kh) / stride_h + 1
        out_w = (w - kw) / stride_w + 1

        do batch = 1, n
            do ch = 1, c
                do i = 1, out_h
                    do j = 1, out_w
                        in_i = (i - 1) * stride_h + 1
                        in_j = (j - 1) * stride_w + 1
                        sum_val = 0.0_real64
                        do ki = 1, kh
                            do kj = 1, kw
                                sum_val = sum_val + input(ch, in_i+ki-1, in_j+kj-1, batch)
                            end do
                        end do
                        output(ch, i, j, batch) = sum_val * inv_area
                    end do
                end do
            end do
        end do
    end subroutine avgpool2d

end module convolution
