//! Tensor data structure — 64-byte aligned memory for AVX-512

use std::alloc::{alloc, dealloc, Layout};

#[repr(C, align(64))]
pub struct Tensor {
    pub data: *mut f64,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
}

impl Tensor {
    pub fn new(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let layout = Layout::from_size_align(size * 8, 64).unwrap();
        let data = unsafe { alloc(layout) as *mut f64 };

        // Fortran column-major strides
        let mut strides = vec![1usize];
        for &s in shape.iter().take(shape.len() - 1) {
            strides.push(strides.last().unwrap() * s);
        }

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            size,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let t = Self::new(shape);
        unsafe {
            std::ptr::write_bytes(t.data, 0, t.size);
        }
        t
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = Layout::from_size_align(self.size * 8, 64).unwrap();
            unsafe { dealloc(self.data as *mut u8, layout) };
        }
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}
