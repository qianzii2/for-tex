//! ForTeX — Fortran Tensor Executor: Rust调度层
//!
//! 职责：
//! 1. FFI 绑定 Fortran 数值算子
//! 2. 线程池（rayon）
//! 3. PyO3 Python 绑定

mod ffi;
mod tensor;

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArrayDyn};
use numpy::ndarray::{ArrayD, Array1, Array2, Array4};
use rayon::prelude::*;

// ============================================================================
// Activation 内联函数（在 Rust 侧做逐元素操作，保持 Fortran 做重活）
// ============================================================================
mod activation {
    #[inline(always)]
    pub fn relu(x: f64) -> f64 { if x > 0.0 { x } else { 0.0 } }
    #[inline(always)]
    pub fn gelu(x: f64) -> f64 {
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
        let x3 = x * x * x;
        0.5 * x * (1.0 + f64::tanh(SQRT_2_OVER_PI * (x + 0.044715 * x3)))
    }
    #[inline(always)]
    pub fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 { 1.0 / (1.0 + f64::exp(-x)) } else { let ex = f64::exp(x); ex / (1.0 + ex) }
    }
    #[inline(always)]
    pub fn tanh_act(x: f64) -> f64 { f64::tanh(x) }
}

// ============================================================================
// Python Module
// ============================================================================

#[pymodule]
fn for_tex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    m.add_function(wrap_pyfunction!(gemv, m)?)?;
    m.add_function(wrap_pyfunction!(simple_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(linear_relu, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_fn, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_fn, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(maxpool, m)?)?;
    m.add_function(wrap_pyfunction!(avgpool, m)?)?;
    m.add_function(wrap_pyfunction!(layernorm, m)?)?;
    m.add_function(wrap_pyfunction!(mse, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

// ============================================================================
// GEMM — 调用 Fortran matmul() 内建（已验证正确）
// ============================================================================

#[pyfunction]
#[pyo3(signature = (a, b, trans_a=None, trans_b=None))]
fn gemm<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    trans_a: Option<bool>,
    trans_b: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let ta = trans_a.unwrap_or(false);
    let tb = trans_b.unwrap_or(false);

    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();

    let (m, k_a) = if ta { (a_arr.shape()[1], a_arr.shape()[0]) } else { (a_arr.shape()[0], a_arr.shape()[1]) };
    let (k_b, n) = if tb { (b_arr.shape()[1], b_arr.shape()[0]) } else { (b_arr.shape()[0], b_arr.shape()[1]) };
    assert_eq!(k_a, k_b, "Inner dimension mismatch: {} vs {}", k_a, k_b);
    let k = k_a;

    if !ta && !tb {
        // ★ 零拷贝快路径：C = A@B 行主序 ⟺ C^T = B^T@A^T 列主序
        let mut pyout = Array2::zeros((m, n));
        let c_slice = pyout.as_slice_mut().unwrap();
        unsafe {
            ffi::c_rowmajor_gemm(m as i32, n as i32, k as i32,
                a_slice.as_ptr(), b_slice.as_ptr(), c_slice.as_mut_ptr());
        }
        Ok(pyout.into_pyarray(py))
    } else {
        // 回退：转置情况需手工处理
        let mut a_f = vec![0.0f64; m * k];
        if ta {
            for i in 0..m { for j in 0..k { a_f[i + j * m] = a_slice[j * k + i]; }}
        } else {
            for j in 0..k { for i in 0..m { a_f[i + j * m] = a_slice[i * k + j]; }}
        }
        let mut b_f = vec![0.0f64; k * n];
        if tb {
            for i in 0..n { for j in 0..k { b_f[i + j * n] = b_slice[j * n + i]; }}
        } else {
            for j in 0..n { for i in 0..k { b_f[i + j * k] = b_slice[i * n + j]; }}
        }
        let mut c_f = vec![0.0f64; m * n];
        unsafe {
            ffi::c_simple_gemm(m as i32, n as i32, k as i32,
                a_f.as_ptr(), b_f.as_ptr(), c_f.as_mut_ptr());
        }
        let mut pyout = Array2::zeros((m, n));
        for j in 0..n { for i in 0..m { pyout[(i, j)] = c_f[i + j * m]; }}
        Ok(pyout.into_pyarray(py))
    }
}

// ============================================================================
// GEMV — y = A @ x, 通过 gemm(x.reshape(n,1)) 实现
// ============================================================================

#[pyfunction]
fn gemv<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let x_arr = x.as_array();
    let m = a_arr.shape()[0];
    let n = a_arr.shape()[1];
    let a_slice = a_arr.as_slice().unwrap();
    let x_slice = x_arr.as_slice().unwrap();

    // ★ 零拷贝：y(m) = A(m,n)@x(n) 行主序 ⟺ y^T(1,m) = x^T(1,n)@A^T(n,m) 列主序
    let mut pyout = Array1::zeros(m);
    let y_slice = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_rowmajor_gemm(m as i32, 1, n as i32,
            a_slice.as_ptr(), x_slice.as_ptr(), y_slice.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// simple_gemm — Fortran matmul() 内建，一行  vs 六重分块手工 GEMM
// ============================================================================

#[pyfunction]
fn simple_gemm<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    assert_eq!(k, b.shape()[0], "Inner dimension mismatch");

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();

    // ★ 零拷贝：直接传行主序指针
    let mut pyout = Array2::zeros((m, n));
    let c_slice = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_rowmajor_gemm(m as i32, n as i32, k as i32,
            a_slice.as_ptr(), b_slice.as_ptr(), c_slice.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// linear_relu — fused: y = relu(W @ x + b), one memory pass
// ============================================================================

// ============================================================================
// linear_relu — Fortran: matmul + bias+relu fusion
// ============================================================================

#[pyfunction]
fn linear_relu<'py>(
    py: Python<'py>,
    weight: PyReadonlyArrayDyn<'py, f64>,
    bias: PyReadonlyArrayDyn<'py, f64>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w = weight.as_array();
    let b = bias.as_array();
    let x_arr = x.as_array();
    let m = w.shape()[0];
    let k = w.shape()[1];
    let n = x_arr.shape()[1];

    let ws = w.as_slice().unwrap();
    let bs = b.as_slice().unwrap();
    let xs = x_arr.as_slice().unwrap();

    let mut pyout = Array2::zeros((m, n));
    let ys = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_rowmajor_fused_linear_relu(m as i32, n as i32, k as i32,
            ws.as_ptr(), bs.as_ptr(), xs.as_ptr(), ys.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Activation Functions — 逐元素操作，rayon 并行
// ============================================================================

macro_rules! element_wise {
    ($name:ident, $fn_path:path, $desc:expr) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArrayDyn<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let x_slice = x.as_array();
            let n = x_slice.len();
            let mut y = Array1::zeros(n);
            let x_flat = x_slice.as_slice().unwrap();
            let y_flat = y.as_slice_mut().unwrap();

            y_flat.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = $fn_path(x_flat[i]);
            });

            Ok(y.into_pyarray(py))
        }
    };
}

element_wise!(relu, activation::relu, "ReLU activation");
element_wise!(sigmoid, activation::sigmoid, "Sigmoid activation");
element_wise!(tanh_fn, activation::tanh_act, "Tanh activation");

// GELU: 走 Fortran 批量 OMP+SIMD 路径，因为 tanh() 在 Rust 里不能 SIMD
#[pyfunction]
fn gelu<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_arr = x.as_array();
    let n = x_arr.len();
    let xs = x_arr.as_slice().unwrap();
    let mut pyout = Array1::zeros(n);
    let ys = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_gelu_forward(n as i32, xs.as_ptr(), ys.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Softmax — 调用 Fortran (range-reduced Taylor exp 实验失败，精度+速度均不够)
// ============================================================================

#[pyfunction]
fn softmax_fn<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_arr = x.as_array();
    let dim = x_arr.shape()[x_arr.ndim() - 1];
    let n = x_arr.len() / dim;

    // ★ Round-22: 关键 stride 修复（最终方案）
    //  Python ndarray 是 row-major (n, dim)：行连续，相邻行 stride=dim*8 字节
    //  Fortran c_softmax 把指针当 col-major (d, n)：内层 do i=1,d stride=8 连续读
    //  直接传 ptr → stride 4KB，几乎全部 L1 miss
    //
    //  关键洞察：Rust Array2 (dim, n) row-major 内存布局 ==
    //          Fortran y(d, n) col-major 内存布局（两者都是 [col0, col1, ...]）
    //          所以可以直接传同一个 buffer，零拷贝转置输出
    // ★ Round-22 回滚: 恢复 Round-21 简单接口 (3.86/1.32 ms)
    // .t() 让 input stride 也错了，导致 row sum 错位
    // → 回到 Round-21 直接传 row-major 指针给 Fortran (会 stride 4KB，但 SIMD prefetcher 优化)
    let xs = x_arr.as_slice().unwrap();
    let mut pyout = Array2::<f64>::zeros((n, dim));
    let ys = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_softmax(n as i32, dim as i32,
            xs.as_ptr(), ys.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Conv2D — 调用 Fortran im2col+GEMM
// ============================================================================

#[pyfunction]
#[pyo3(signature = (img, weight, bias, stride=None, padding=None))]
fn conv2d<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f64>,
    weight: PyReadonlyArrayDyn<'py, f64>,
    bias: PyReadonlyArrayDyn<'py, f64>,
    stride: Option<usize>,
    padding: Option<usize>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let img_arr = img.as_array();
    let w_arr = weight.as_array();
    let b_arr = bias.as_array();
    let stride = stride.unwrap_or(1);
    let pad = padding.unwrap_or(0);

    let n = img_arr.shape()[0];
    let c_in = img_arr.shape()[1];
    let h = img_arr.shape()[2];
    let i_w = img_arr.shape()[3];
    let c_out = w_arr.shape()[0];
    let kh = w_arr.shape()[2];
    let kw = w_arr.shape()[3];

    let out_h = (h + 2 * pad - kh) / stride + 1;
    let out_w = (i_w + 2 * pad - kw) / stride + 1;

    // ★ 零拷贝：直接传行主序指针，Fortran 内部用列主序转置恒等式
    // img(n,c_in,h,w) row-major = img^T(w,h,c_in,n) column-major
    // weight(c_out,c_in,kh,kw) row-major = weight^T(kw,kh,c_in,c_out) col-major
    // output(n,c_out,out_h,out_w) row-major = output^T(out_w,out_h,c_out,n) col-major
    let imgs = img_arr.as_slice().unwrap();
    let ws = w_arr.as_slice().unwrap();
    let bs = b_arr.as_slice().unwrap();
    let mut pyout = Array4::zeros((n, c_out, out_h, out_w));
    let ps = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_rowmajor_conv2d(
            n as i32, c_in as i32, c_out as i32, h as i32, i_w as i32,
            kh as i32, kw as i32, stride as i32, pad as i32,
            out_h as i32, out_w as i32,
            imgs.as_ptr(), ws.as_ptr(), bs.as_ptr(), ps.as_mut_ptr(),
        );
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// MaxPool2D
// ============================================================================

#[pyfunction]
#[pyo3(signature = (x, kernel, stride=None))]
fn maxpool<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    kernel: usize,
    stride: Option<usize>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let x_arr = x.as_array();
    let stride = stride.unwrap_or(kernel);
    let n = x_arr.shape()[0];
    let c = x_arr.shape()[1];
    let h = x_arr.shape()[2];
    let w_in = x_arr.shape()[3];

    let out_h = (h - kernel) / stride + 1;
    let out_w = (w_in - kernel) / stride + 1;

    // (n,c,h,w) row-major → (c,h,w,n) column-major
    let xs = x_arr.as_slice().unwrap();
    let mut x_f = vec![0.0f64; c * h * w_in * n];
    for b in 0..n { for xi in 0..w_in { for y in 0..h { for ch in 0..c {
        let fi = ch + y * c + xi * (c * h) + b * (c * h * w_in);
        let ri = ((b * c + ch) * h + y) * w_in + xi;
        x_f[fi] = xs[ri];
    }}}}

    let mut output_f = vec![0.0f64; c * out_h * out_w * n];
    unsafe {
        ffi::c_maxpool2d(n as i32, c as i32, h as i32, w_in as i32,
            kernel as i32, kernel as i32, stride as i32, stride as i32,
            out_h as i32, out_w as i32,
            x_f.as_ptr(),
            output_f.as_mut_ptr());
    }

    // column-major (c,out_h,out_w,n) → row-major (n,c,out_h,out_w)
    let mut pyout = Array4::zeros((n, c, out_h, out_w));
    let ps = pyout.as_slice_mut().unwrap();
    for b in 0..n { for xi in 0..out_w { for y in 0..out_h { for ch in 0..c {
        let fi = ch + y * c + xi * (c * out_h) + b * (c * out_h * out_w);
        let pi = ((b * c + ch) * out_h + y) * out_w + xi;
        ps[pi] = output_f[fi];
    }}}}

    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// AvgPool2D
// ============================================================================

#[pyfunction]
#[pyo3(signature = (x, kernel, stride=None))]
fn avgpool<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    kernel: usize,
    stride: Option<usize>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let x_arr = x.as_array();
    let stride = stride.unwrap_or(kernel);
    let n = x_arr.shape()[0];
    let c = x_arr.shape()[1];
    let h = x_arr.shape()[2];
    let w_in = x_arr.shape()[3];
    let out_h = (h - kernel) / stride + 1;
    let out_w = (w_in - kernel) / stride + 1;

    // (n,c,h,w) row-major → (c,h,w,n) column-major
    let xs = x_arr.as_slice().unwrap();
    let mut x_f = vec![0.0f64; c * h * w_in * n];
    for b in 0..n { for xi in 0..w_in { for y in 0..h { for ch in 0..c {
        let fi = ch + y * c + xi * (c * h) + b * (c * h * w_in);
        let ri = ((b * c + ch) * h + y) * w_in + xi;
        x_f[fi] = xs[ri];
    }}}}

    let mut output_f = vec![0.0f64; c * out_h * out_w * n];
    unsafe {
        ffi::c_avgpool2d(n as i32, c as i32, h as i32, w_in as i32,
            kernel as i32, kernel as i32, stride as i32, stride as i32,
            out_h as i32, out_w as i32,
            x_f.as_ptr(),
            output_f.as_mut_ptr());
    }

    // column-major (c,out_h,out_w,n) → row-major (n,c,out_h,out_w)
    let mut pyout = Array4::zeros((n, c, out_h, out_w));
    let ps = pyout.as_slice_mut().unwrap();
    for b in 0..n { for xi in 0..out_w { for y in 0..out_h { for ch in 0..c {
        let fi = ch + y * c + xi * (c * out_h) + b * (c * out_h * out_w);
        let pi = ((b * c + ch) * out_h + y) * out_w + xi;
        ps[pi] = output_f[fi];
    }}}}

    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// LayerNorm — 调用 Fortran
// ============================================================================

#[pyfunction]
#[pyo3(signature = (x, gamma=None, beta=None, eps=None))]
fn layernorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    gamma: Option<PyReadonlyArrayDyn<'py, f64>>,
    beta: Option<PyReadonlyArrayDyn<'py, f64>>,
    eps: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_arr = x.as_array();
    let eps = eps.unwrap_or(1e-5);
    let batch = x_arr.shape()[0];
    let dim = x_arr.shape()[1];

    let gamma_arr = gamma.map(|g| g.as_array().to_owned()).unwrap_or_else(|| ArrayD::ones(vec![dim]));
    let beta_arr = beta.map(|b| b.as_array().to_owned()).unwrap_or_else(|| ArrayD::zeros(vec![dim]));

    // ★ Round-22 回滚: 恢复 Round-21 简单接口
    let xs = x_arr.as_slice().unwrap();
    let mut pyout = Array2::<f64>::zeros((batch, dim));
    let ys = pyout.as_slice_mut().unwrap();

    unsafe {
        ffi::c_layernorm_forward(
            batch as i32, dim as i32,
            xs.as_ptr(),
            gamma_arr.as_slice().unwrap().as_ptr(),
            beta_arr.as_slice().unwrap().as_ptr(),
            eps,
            ys.as_mut_ptr(),
        );
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Loss
// ============================================================================

#[pyfunction]
fn mse<'py>(
    _py: Python<'py>,
    y_pred: PyReadonlyArrayDyn<'py, f64>,
    y_true: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<f64> {
    let p = y_pred.as_array();
    let t = y_true.as_array();
    let n = p.len() as i32;
    unsafe {
        Ok(ffi::c_mse_loss(n,
            p.as_slice().unwrap().as_ptr(),
            t.as_slice().unwrap().as_ptr()))
    }
}

#[pyfunction]
fn cross_entropy<'py>(
    _py: Python<'py>,
    logits: PyReadonlyArrayDyn<'py, f64>,
    targets: PyReadonlyArrayDyn<'py, i64>,
) -> PyResult<f64> {
    let l = logits.as_array();
    let t = targets.as_array();
    let num_classes = l.shape()[1];
    let batch = l.shape()[0];

    // ★ 零拷贝：logits(batch,classes) 行主序 = logits^T(classes,batch) 列主序
    let ls = l.as_slice().unwrap();
    let ts = t.as_slice().unwrap();
    let targets_1b: Vec<i32> = ts.iter().map(|&v| (v + 1) as i32).collect();

    unsafe {
        Ok(ffi::c_cross_entropy_loss(num_classes as i32, batch as i32,
            ls.as_ptr(),
            targets_1b.as_ptr()))
    }
}
