!==============================================================================
! scipy OpenBLAS wrapper — calls scipy's bundled OpenBLAS
!==============================================================================
module scipy_openblas
    use, intrinsic :: iso_c_binding
    implicit none
    private
    public :: scipy_dgemm, scipy_dgemm_generic

    ! CBLAS constants
    integer(c_int), parameter :: CblasRowMajor = 101
    integer(c_int), parameter :: CblasColMajor = 102
    integer(c_int), parameter :: CblasNoTrans   = 111
    integer(c_int), parameter :: CblasTrans     = 112

    interface
        subroutine scipy_cblas_dgemm(order, transa, transb, m, n, k, &
            alpha, a, lda, b, ldb, beta, c, ldc) &
            bind(c, name="scipy_cblas_dgemm")
            import :: c_int, c_double
            integer(c_int), value :: order, transa, transb, m, n, k, lda, ldb, ldc
            real(c_double), value :: alpha, beta
            real(c_double), intent(in) :: a(*), b(*)
            real(c_double), intent(inout) :: c(*)
        end subroutine
    end interface

contains

    !--------------------------------------------------------------------------
    ! Row-major GEMM: C(m,n) = A(m,k) @ B(k,n)
    ! Uses zero-copy transpose identity: C^T = B^T @ A^T (col-major = row-major)
    !--------------------------------------------------------------------------
    subroutine scipy_dgemm(m, n, k, a, b, c)
        integer, intent(in) :: m, n, k
        real(c_double), intent(in) :: a(k, m)  ! A^T col-major = A row-major
        real(c_double), intent(in) :: b(n, k)  ! B^T col-major = B row-major
        real(c_double), intent(out) :: c(n, m) ! C^T col-major = C row-major
        real(c_double) :: alpha, beta
        alpha = 1.0_c_double
        beta  = 0.0_c_double
        ! C^T(n,m) = B^T(n,k) @ A^T(k,m)  in column-major
        call scipy_cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, &
            n, m, k, alpha, b, n, a, k, beta, c, n)
    end subroutine scipy_dgemm

    !--------------------------------------------------------------------------
    ! Generic GEMM: C = A @ B (column-major, NoTrans)
    !--------------------------------------------------------------------------
    subroutine scipy_dgemm_generic(m, n, k, a, lda, b, ldb, c, ldc)
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        real(c_double), intent(in) :: a(lda,*), b(ldb,*)
        real(c_double), intent(out) :: c(ldc,*)
        call scipy_cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, &
            m, n, k, 1.0_c_double, a, lda, b, ldb, 0.0_c_double, c, ldc)
    end subroutine scipy_dgemm_generic

end module scipy_openblas