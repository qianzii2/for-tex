//! ForTeX — Rust scheduling layer: PyO3 + rayon + FFI
//!
//! Responsibilities:
//! 1. FFI bindings to Fortran numerical operators
//! 2. Thread pool (rayon)
//! 3. PyO3 Python bindings

mod ffi;

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArrayDyn};
use numpy::ndarray::{ArrayD, Array1, Array2, Array4};
use rayon::prelude::*;

// ============================================================================
// Activation inline functions (element-wise ops in Rust, Fortran does heavy work)
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
// GEMM — Calls Fortran matmul() built-in (verified correct)
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
        // ★ Zero-copy fast path: C = A@B row-major ⟺ C^T = B^T@A^T column-major
        let mut pyout = Array2::zeros((m, n));
        let c_slice = pyout.as_slice_mut().unwrap();
        unsafe {
            ffi::c_rowmajor_gemm(m as i32, n as i32, k as i32,
                a_slice.as_ptr(), b_slice.as_ptr(), c_slice.as_mut_ptr());
        }
        Ok(pyout.into_pyarray(py))
    } else {
        // Fallback: transpose case requires manual handling
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
// GEMV — y = A @ x, implemented via gemm(x.reshape(n,1))
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

    // ★ Zero-copy: y(m) = A(m,n)@x(n) row-major ⟺ y^T(1,m) = x^T(1,n)@A^T(n,m) column-major
    let mut pyout = Array1::zeros(m);
    let y_slice = pyout.as_slice_mut().unwrap();
    unsafe {
        ffi::c_rowmajor_gemm(m as i32, 1, n as i32,
            a_slice.as_ptr(), x_slice.as_ptr(), y_slice.as_mut_ptr());
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// simple_gemm — Fortran matmul() built-in, one-liner vs six-nested-block hand GEMM
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

    // ★ Zero-copy: pass row-major pointers directly
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
// Activation Functions — element-wise ops, rayon parallel
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

// GELU: small matrices use Rust scalar (avoid OMP thread overhead), large use Fortran OMP+SIMD
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

    if n < 262144 {
        // Small matrices: Rust scalar (rayon parallel, no OMP overhead)
        use activation::gelu as gelu_scalar;
        ys.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v = gelu_scalar(xs[i]);
        });
    } else {
        // Large matrices: Fortran OMP + SIMD Padé tanh
        unsafe {
            ffi::c_gelu_forward(n as i32, xs.as_ptr(), ys.as_mut_ptr());
        }
    }
    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Softmax — Rust rayon parallel + fast approximate exp (Padé [1/1])
// ============================================================================

// ============================================================================
// Softmax — AVX2 SIMD (max+exp+scale)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast exp(x) approximation, Padé [1/1]: 2^r ≈ (1+r*A)/(1+r*C), 2^n via bit manipulation
/// Pure arithmetic, no libm calls, compiler can auto-vectorize
#[inline(always)]
fn fast_exp(x: f64) -> f64 {
    const INV_LN2: f64 = 1.4426950408889634;
    const A: f64 = 0.24022650695910071;
    const C: f64 = 0.10678711907894758;
    let y = x * INV_LN2;
    let n = y.round();
    let r = y - n;
    let num = 1.0 + r * A;
    let den = 1.0 + r * C;
    let p = num / den;
    let ni = n as i64;
    p * if ni > 1023 { f64::INFINITY } else if ni < -1023 { 0.0 } else { f64::from_bits(((ni + 1023) as u64) << 52) }
}

/// AVX2 SIMD exp: 4×f64 computed simultaneously
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_avx2_4(x: __m256d) -> __m256d {
    const INV_LN2: f64 = 1.4426950408889634;
    const A: f64 = 0.24022650695910071;
    const C: f64 = 0.10678711907894758;
    let v_inv_ln2 = _mm256_set1_pd(INV_LN2);
    let v_a = _mm256_set1_pd(A);
    let v_c = _mm256_set1_pd(C);
    let v_one = _mm256_set1_pd(1.0);
    let v_1023 = _mm256_set1_pd(1023.0);
    let v_shift = _mm256_set1_pd(4503599627370496.0); // 2^52 for rounding trick

    // y = x / ln(2)
    let y = _mm256_mul_pd(x, v_inv_ln2);
    // n = round(y) via add-shift-sub trick
    let n = _mm256_sub_pd(_mm256_add_pd(y, v_shift), v_shift);
    // r = y - n
    let r = _mm256_sub_pd(y, n);
    // num = 1 + r*A
    let num = _mm256_fmadd_pd(r, v_a, v_one);
    // den = 1 + r*C
    let den = _mm256_fmadd_pd(r, v_c, v_one);
    // p = num / den
    let p = _mm256_div_pd(num, den);
    // 2^n: shift exponent by n
    let n_bits = _mm256_slli_epi64::<52>(_mm256_add_epi64(
        _mm256_castpd_si256(n),
        _mm256_castpd_si256(v_1023)
    ));
    let two_n = _mm256_castsi256_pd(n_bits);
    _mm256_mul_pd(p, two_n)
}

/// AVX2 SIMD softmax on a slice of dim elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_row_avx2(dim: usize, src: *const f64, dst: *mut f64) {
    let d4 = dim / 4;
    let rem = dim % 4;

    // Constants for exp
    const INV_LN2: f64 = 1.4426950408889634;
    const A: f64 = 0.24022650695910071;
    const C: f64 = 0.10678711907894758;
    let v_inv_ln2 = _mm256_set1_pd(INV_LN2);
    let v_a = _mm256_set1_pd(A);
    let v_c = _mm256_set1_pd(C);
    let v_one = _mm256_set1_pd(1.0);
    let v_1023 = _mm256_set1_pd(1023.0);
    let v_shift = _mm256_set1_pd(4503599627370496.0); // 2^52

    // Pass 1: SIMD find max
    let mut m = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut j = 0;
    for _ in 0..d4 {
        let v = _mm256_loadu_pd(src.add(j));
        m = _mm256_max_pd(m, v);
        j += 4;
    }
    let m2 = _mm256_max_pd(m, _mm256_permute4x64_pd::<0b10>(m));
    let m3 = _mm256_max_pd(m2, _mm256_permute4x64_pd::<0b01>(m2));
    let mut max_val = _mm256_cvtsd_f64(m3);
    for jj in 0..rem { let v = *src.add(j + jj); if v > max_val { max_val = v; } }

    let v_m = _mm256_set1_pd(max_val);

    // Pass 2: exp + sum (4-wide SIMD, fused, inlined exp)
    let mut sum = _mm256_setzero_pd();
    j = 0;
    for _ in 0..d4 {
        let x = _mm256_sub_pd(_mm256_loadu_pd(src.add(j)), v_m);
        // Inlined exp_avx2_4
        let y = _mm256_mul_pd(x, v_inv_ln2);
        let n = _mm256_sub_pd(_mm256_add_pd(y, v_shift), v_shift);
        let r = _mm256_sub_pd(y, n);
        let num = _mm256_fmadd_pd(r, v_a, v_one);
        let den = _mm256_fmadd_pd(r, v_c, v_one);
        let p = _mm256_div_pd(num, den);
        let n_bits = _mm256_slli_epi64::<52>(_mm256_add_epi64(
            _mm256_castpd_si256(n), _mm256_castpd_si256(v_1023)));
        let e = _mm256_mul_pd(p, _mm256_castsi256_pd(n_bits));
        _mm256_storeu_pd(dst.add(j), e);
        sum = _mm256_add_pd(sum, e);
        j += 4;
    }
    let sum2 = _mm256_hadd_pd(sum, sum);
    let mut s = _mm256_cvtsd_f64(sum2) + _mm256_cvtsd_f64(_mm256_permute4x64_pd::<0b10>(sum2));
    for jj in 0..rem { let v = fast_exp(*src.add(j + jj) - max_val); *dst.add(j + jj) = v; s += v; }

    let inv = 1.0 / s;
    let v_inv = _mm256_set1_pd(inv);
    j = 0;
    for _ in 0..d4 {
        let y = _mm256_mul_pd(_mm256_loadu_pd(dst.add(j)), v_inv);
        _mm256_storeu_pd(dst.add(j), y);
        j += 4;
    }
    for jj in 0..rem { *dst.add(j + jj) *= inv; }
}

#[pyfunction]
fn softmax_fn<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_arr = x.as_array();
    let dim = x_arr.shape()[x_arr.ndim() - 1];
    let n = x_arr.len() / dim;
    let xs = x_arr.as_slice().unwrap();

    let mut pyout = Array2::<f64>::zeros((n, dim));
    let ys = pyout.as_slice_mut().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && dim >= 4 {
            ys.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
                let offset = i * dim;
                unsafe {
                    softmax_row_avx2(dim, xs.as_ptr().add(offset), row.as_mut_ptr());
                }
            });
            return Ok(pyout.into_pyarray(py));
        }
    }

    // Fallback: scalar fast_exp
    ys.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let offset = i * dim;
        let mut m = f64::NEG_INFINITY;
        for j in 0..dim { let v = xs[offset + j]; if v > m { m = v; } }
        let mut s = 0.0f64;
        for j in 0..dim {
            let v = fast_exp(xs[offset + j] - m);
            row[j] = v;
            s += v;
        }
        let inv = 1.0 / s;
        for j in 0..dim { row[j] *= inv; }
    });

    Ok(pyout.into_pyarray(py))
}

// ============================================================================
// Conv2D — Calls Fortran im2col+GEMM
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

    // ★ Zero-copy: pass row-major pointers directly; Fortran uses column-major transpose identity internally
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
// LayerNorm — Hybrid strategy: small batch Rust 2-pass, large batch Fortran 2-pass SIMD
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

    // ★ Large batch uses Fortran (OpenMP SIMD), small batch uses Rust rayon
    if batch * dim > 8192 {
        let gamma_arr = gamma.map(|g| g.as_array().to_owned()).unwrap_or_else(|| ArrayD::ones(vec![dim]));
        let beta_arr = beta.map(|b| b.as_array().to_owned()).unwrap_or_else(|| ArrayD::zeros(vec![dim]));
        let xs = x_arr.as_slice().unwrap();
        let mut pyout = Array2::<f64>::zeros((batch, dim));
        let ys = pyout.as_slice_mut().unwrap();
        unsafe {
            ffi::c_layernorm_forward(batch as i32, dim as i32, xs.as_ptr(),
                gamma_arr.as_slice().unwrap().as_ptr(), beta_arr.as_slice().unwrap().as_ptr(),
                eps, ys.as_mut_ptr());
        }
        return Ok(pyout.into_pyarray(py));
    }

    let xs = x_arr.as_slice().unwrap();
    let gamma_slice = gamma.map(|g| g.as_array().as_slice().unwrap().to_vec())
        .unwrap_or_else(|| vec![1.0f64; dim]);
    let beta_slice = beta.map(|b| b.as_array().as_slice().unwrap().to_vec())
        .unwrap_or_else(|| vec![0.0f64; dim]);
    let mut pyout = Array2::<f64>::zeros((batch, dim));
    let ys = pyout.as_slice_mut().unwrap();
    let inv_dim = 1.0 / dim as f64;

    // ★ 2-pass: Pass1 = sum + sum_sq → mean/var, Pass2 = normalize
    ys.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let offset = i * dim;
        let mut sx = 0.0f64;
        let mut sx2 = 0.0f64;
        for j in 0..dim {
            let v = xs[offset + j];
            sx += v;
            sx2 += v * v;
        }
        let mean = sx * inv_dim;
        let var = sx2 * inv_dim - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..dim {
            row[j] = gamma_slice[j] * ((xs[offset + j] - mean) * inv_std) + beta_slice[j];
        }
    });
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

    // ★ Zero-copy: logits(batch,classes) row-major = logits^T(classes,batch) column-major
    let ls = l.as_slice().unwrap();
    let ts = t.as_slice().unwrap();
    let targets_1b: Vec<i32> = ts.iter().map(|&v| (v + 1) as i32).collect();

    unsafe {
        Ok(ffi::c_cross_entropy_loss(num_classes as i32, batch as i32,
            ls.as_ptr(),
            targets_1b.as_ptr()))
    }
}