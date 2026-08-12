//! FFI bindings to Fortran compiled library.

extern "C" {
    // BLAS
    pub fn c_dgemm(
        transa: u8, transb: u8,
        m: i32, n: i32, k: i32,
        alpha: f64,
        a: *const f64, lda: i32,
        b: *const f64, ldb: i32,
        beta: f64,
        c: *mut f64, ldc: i32,
    );
    pub fn c_dgemv(
        trans: u8,
        m: i32, n: i32,
        alpha: f64,
        a: *const f64, lda: i32,
        x: *const f64, incx: i32,
        beta: f64,
        y: *mut f64, incy: i32,
    );
    pub fn c_daxpy(
        n: i32,
        alpha: f64,
        x: *const f64, incx: i32,
        y: *mut f64, incy: i32,
    );
    pub fn c_ddot(
        n: i32,
        x: *const f64, incx: i32,
        y: *const f64, incy: i32,
    ) -> f64;
    // ★ Fortran matmul() 内建 — 一行 GEMM，编译器级优化
    pub fn c_simple_gemm(
        m: i32, n: i32, k: i32,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
    );
    // ★ 算子融合: y = relu(W @ x + b) — 一次内存遍历
    pub fn c_fused_linear_relu(
        m: i32, n: i32, k: i32,
        weight: *const f64,
        bias: *const f64,
        x: *const f64,
        y: *mut f64,
    );
    // ★★ 零拷贝 GEMM — 直接传行主序指针，Fortran 内部用转置恒等式
    pub fn c_rowmajor_gemm(
        m: i32, n: i32, k: i32,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
    );
    // ★★ 零拷贝 Fused Linear+ReLU
    pub fn c_rowmajor_fused_linear_relu(
        m: i32, n: i32, k: i32,
        weight: *const f64,
        bias: *const f64,
        x: *const f64,
        y: *mut f64,
    );

    // Activation
    pub fn c_relu_forward(n: i32, x: *mut f64);
    pub fn c_gelu_forward(n: i32, x: *const f64, y: *mut f64);
    pub fn c_softmax(n: i32, d: i32, x: *const f64, y: *mut f64);

    // Convolution
    pub fn c_conv2d_forward(
        n: i32, c_in: i32, c_out: i32, h: i32, w: i32, kh: i32, kw: i32,
        stride_h: i32, stride_w: i32, pad_h: i32, pad_w: i32,
        out_h: i32, out_w: i32,
        img: *const f64, weight: *const f64, bias: *const f64,
        output: *mut f64,
    );
    pub fn c_maxpool2d(
        n: i32, c: i32, h: i32, w: i32,
        kh: i32, kw: i32, stride_h: i32, stride_w: i32,
        out_h: i32, out_w: i32,
        input: *const f64, output: *mut f64,
    );
    pub fn c_avgpool2d(
        n: i32, c: i32, h: i32, w: i32,
        kh: i32, kw: i32, stride_h: i32, stride_w: i32,
        out_h: i32, out_w: i32,
        input: *const f64, output: *mut f64,
    );

    // ★★ 零拷贝 Conv2D — 直接卷积，无需转置
    pub fn c_rowmajor_conv2d(
        n: i32, c_in: i32, c_out: i32, h: i32, w: i32, kh: i32, kw: i32,
        stride: i32, pad: i32, out_h: i32, out_w: i32,
        img: *const f64, weight: *const f64, bias: *const f64,
        output: *mut f64,
    );
    // Normalization
    pub fn c_batchnorm2d_forward(
        n: i32, c: i32, h: i32, w: i32,
        x: *const f64, gamma: *const f64, beta: *const f64,
        running_mean: *mut f64, running_var: *mut f64,
        eps: f64, momentum: f64, training: bool,
        y: *mut f64,
    );
    pub fn c_layernorm_forward(
        batch: i32, dim: i32,
        x: *const f64, gamma: *const f64, beta: *const f64,
        eps: f64,
        y: *mut f64,
    );

    // Loss
    pub fn c_mse_loss(n: i32, y_pred: *const f64, y_true: *const f64) -> f64;
    pub fn c_cross_entropy_loss(num_classes: i32, batch: i32, logits: *const f64, targets: *const i32) -> f64;
}
