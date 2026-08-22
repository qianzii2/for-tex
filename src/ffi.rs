//! FFI bindings to Fortran library

extern "C" {
    // ★ Zero-copy GEMM — pass row-major pointers directly
    pub fn c_rowmajor_gemm(
        m: i32, n: i32, k: i32,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
    );
    // ★ Zero-copy Fused Linear+ReLU
    pub fn c_rowmajor_fused_linear_relu(
        m: i32, n: i32, k: i32,
        weight: *const f64,
        bias: *const f64,
        x: *const f64,
        y: *mut f64,
    );
    // ★ Fortran matmul() built-in
    pub fn c_simple_gemm(
        m: i32, n: i32, k: i32,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
    );

    // Activation
    pub fn c_gelu_forward(n: i32, x: *const f64, y: *mut f64);

    // Convolution
    pub fn c_rowmajor_conv2d(
        n: i32, c_in: i32, c_out: i32, h: i32, w: i32, kh: i32, kw: i32,
        stride: i32, pad: i32, out_h: i32, out_w: i32,
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

    // Normalization
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