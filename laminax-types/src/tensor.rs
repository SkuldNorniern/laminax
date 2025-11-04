//! Tensor data structures powered by Numina

use std::fmt;
use numina::{CpuBytesArray, NdArray, Shape, Strides, DType};
use numina::{add as numina_add, mul as numina_mul};
use numina::{sum as numina_sum, mean as numina_mean};
use numina::{exp as numina_exp, log as numina_log, sqrt as numina_sqrt};

// Re-export types that are part of the laminax-types API
pub use numina::{BFloat16, QuantizedU8, QuantizedI4};

/// Main tensor structure powered by Numina
#[derive(Debug)]
pub struct Tensor {
    storage: Box<dyn NdArray>,
}

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        Tensor {
            storage: CpuBytesArray::new(data, shape, dtype).into_boxed(),
        }
    }

    /// Create tensor from slice (copies data)
    pub fn from_slice<T>(data: &[T], shape: Shape) -> Self
    where
        T: Copy + Into<DType>,
    {
        let dtype = data[0].into();
        let expected_len = shape.len();

        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} does not match shape {}",
            data.len(),
            shape
        );

        let byte_len = data.len() * std::mem::size_of::<T>();
        let mut bytes = Vec::<u8>::with_capacity(byte_len);
        unsafe {
            bytes.set_len(byte_len);
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }

        Self::new(bytes, shape, dtype)
    }

    /// Create tensor filled with zeros
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        Tensor {
            storage: CpuBytesArray::zeros(dtype, shape).into_boxed(),
        }
    }

    /// Create tensor filled with ones
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        Tensor {
            storage: CpuBytesArray::ones(dtype, shape).into_boxed(),
        }
    }

    /// Create identity matrix
    pub fn eye(dtype: DType, n: usize) -> Self {
        Tensor {
            storage: CpuBytesArray::eye(dtype, n).into_boxed(),
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &Shape {
        self.storage.shape()
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.storage.len() == 0
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.storage.shape().ndim()
    }

    /// Get strides
    pub fn strides(&self) -> &Strides {
        self.storage.strides()
    }

    /// Extract tensor data as f32 vector (for debugging/verification)
    /// This is a convenience method for small tensors - not efficient for large ones
    pub fn to_vec_f32(&self) -> Result<Vec<f32>, String> {
        if self.dtype() != crate::F32 {
            return Err(format!("to_vec_f32 only supported for F32 tensors, got {:?}", self.dtype()));
        }

        let byte_len = self.len() * 4; // f32 = 4 bytes
        if byte_len != self.storage.byte_len() {
            return Err("Byte length mismatch".to_string());
        }

        let mut result = vec![0.0f32; self.len()];
        unsafe {
            let bytes = self.storage.as_bytes();
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        Ok(result)
    }

    /// Set tensor data from f32 slice (for testing/computation results)
    /// This is a temporary method until proper tensor mutation API is implemented
    pub fn set_from_f32_slice(&mut self, data: &[f32]) -> Result<(), String> {
        if self.dtype() != crate::F32 {
            return Err(format!("set_from_f32_slice only supported for F32 tensors, got {:?}", self.dtype()));
        }

        if data.len() != self.len() {
            return Err(format!("Data length {} does not match tensor length {}", data.len(), self.len()));
        }

        let byte_len = data.len() * 4; // f32 = 4 bytes
        unsafe {
            let dest_bytes = self.storage.as_mut_bytes();
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                dest_bytes.as_mut_ptr(),
                byte_len,
            );
        }
        Ok(())
    }

    /// Create a tensor from any NdArray implementation
    pub fn from_ndarray(array: Box<dyn NdArray>) -> Self {
        Tensor { storage: array }
    }

    /// Clone this tensor with its storage
    pub fn clone_tensor(&self) -> Self {
        Tensor {
            storage: self.storage.clone_array(),
        }
    }

    /// Reshape tensor (creates new storage if supported)
    pub fn reshape(self, new_shape: Shape) -> Result<Self, String> {
        let reshaped_storage = self.storage.reshape(new_shape)?;
        Ok(Tensor {
            storage: reshaped_storage,
        })
    }

    /// Transpose tensor (2D only, creates new storage if supported)
    pub fn transpose(self) -> Result<Self, String> {
        let transposed_storage = self.storage.transpose()?;
        Ok(Tensor {
            storage: transposed_storage,
        })
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        let result = numina_add(self, other)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        let result = numina_mul(self, other)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Tensor, String> {
        let result = numina_exp(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise logarithm
    pub fn log(&self) -> Result<Tensor, String> {
        let result = numina_log(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor, String> {
        let result = numina_sqrt(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Sum reduction
    pub fn sum(&self, axis: Option<usize>) -> Result<Tensor, String> {
        let result = numina_sum(self, axis)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Mean reduction
    pub fn mean(&self, axis: Option<usize>) -> Result<Tensor, String> {
        let result = numina_mean(self, axis)?;
        Ok(Tensor::from_ndarray(result))
    }
}

impl NdArray for Tensor {
    fn shape(&self) -> &Shape {
        self.storage.shape()
    }

    fn strides(&self) -> &Strides {
        self.storage.strides()
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { self.storage.as_bytes() }
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { self.storage.as_mut_bytes() }
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        self.storage.clone_array()
    }

    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.storage.reshape(new_shape)
    }

    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        self.storage.transpose()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {})",
            self.storage.shape(),
            self.storage.dtype(),
            self.storage
                .strides()
                .as_slice()
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32;

    #[test]
    fn tensor_zeros() {
        let shape = Shape::from([2, 3]);
        let tensor = Tensor::zeros(F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn tensor_ones() {
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::ones(F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::from_slice(&data, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_reshape() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = Shape::from([2, 3]);
        let tensor1 = Tensor::from_slice(&data, shape1);

        let shape2 = Shape::from([3, 2]);
        let tensor2 = tensor1.reshape(shape2.clone()).unwrap();
        assert_eq!(tensor2.shape(), &shape2);
        assert_eq!(tensor2.len(), 6);
    }

    #[test]
    fn tensor_display() {
        let shape = Shape::from([2, 3]);
        let tensor = Tensor::zeros(F32, shape);
        let display = format!("{}", tensor);
        assert!(display.contains("Tensor([2, 3], f32"));
    }
}
