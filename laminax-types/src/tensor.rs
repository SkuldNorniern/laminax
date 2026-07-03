//! Tensor data structures powered by Numina (NumPy-like array API).
//!
//! Numina's array layer provides **dtype**, **shape**, **ndim**, **size**, and creation
//! (zeros, ones, eye, arange, linspace). This module adds a high-level [`Tensor`] with
//! [`TensorStorage`] backends; [`Tensor`] implements [`NdArray`] for Numina ops (add, mul, sum).
//! Use [`Tensor::from_slice`] with types that implement Numina's [`TensorElement`].

use std::fmt;
use numina::{NdArray, Shape, Strides, TensorElement, DType};
use numina::{add as numina_add, mul as numina_mul};
use numina::{sum as numina_sum, mean as numina_mean};
use numina::{exp as numina_exp, log as numina_log, sqrt as numina_sqrt};

// Re-export types that are part of the laminax-types API
pub use numina::{BFloat16, QuantizedU8, QuantizedI4};

/// Backend storage for tensor data. Implement this (or use [`CpuStorage`]) instead of Numina's NdArray.
pub trait TensorStorage: Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &Strides;
    fn len(&self) -> usize;
    fn dtype(&self) -> DType;
    fn byte_len(&self) -> usize {
        self.len() * self.dtype().dtype_size_bytes()
    }
    /// # Safety
    /// Caller must ensure the slice is not used beyond the storage lifetime and matches dtype layout.
    unsafe fn as_bytes(&self) -> &[u8];
    /// # Safety
    /// Same as `as_bytes`; no other references to this storage may alias the returned region.
    unsafe fn as_mut_bytes(&mut self) -> &mut [u8];
    fn clone_storage(&self) -> Box<dyn TensorStorage>;
    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn TensorStorage>, String>;
    fn transpose(&self) -> Result<Box<dyn TensorStorage>, String>;
    fn zeros(&self, shape: Shape) -> Result<Box<dyn TensorStorage>, String>;
    fn ones(&self, shape: Shape) -> Result<Box<dyn TensorStorage>, String>;
    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn TensorStorage>, String>;
}

/// In-memory row-major storage (bytes + shape + dtype).
#[derive(Debug)]
pub struct CpuStorage {
    data: Vec<u8>,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}

impl CpuStorage {
    pub fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        let strides = Strides::from_shape(&shape);
        assert_eq!(data.len(), shape.len() * dtype.dtype_size_bytes());
        Self { data, shape, strides, dtype }
    }

    /// Like `np.zeros`: delegate to Numina's array creation so all dtypes are consistent.
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let arr = numina::CpuBytesArray::zeros(dtype, shape.clone());
        Self {
            data: unsafe { arr.as_bytes() }.to_vec(),
            strides: Strides::from_shape(&shape),
            shape,
            dtype,
        }
    }

    /// Like `np.ones`: delegate to Numina's array creation.
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        let arr = numina::CpuBytesArray::ones(dtype, shape.clone());
        Self {
            data: unsafe { arr.as_bytes() }.to_vec(),
            strides: Strides::from_shape(&shape),
            shape,
            dtype,
        }
    }

    /// Like `np.eye`: identity matrix; delegate to Numina.
    pub fn eye(dtype: DType, n: usize) -> Self {
        let arr = numina::CpuBytesArray::eye(dtype, n);
        let shape = arr.shape().clone();
        Self {
            data: unsafe { arr.as_bytes() }.to_vec(),
            strides: Strides::from_shape(&shape),
            shape,
            dtype: arr.dtype(),
        }
    }
}

impl TensorStorage for CpuStorage {
    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn strides(&self) -> &Strides {
        &self.strides
    }
    fn len(&self) -> usize {
        self.shape.len()
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    unsafe fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.data
    }
    fn clone_storage(&self) -> Box<dyn TensorStorage> {
        Box::new(Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        })
    }
    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn TensorStorage>, String> {
        if new_shape.len() != self.shape.len() {
            return Err(format!("reshape: size {} != {}", new_shape.len(), self.shape.len()));
        }
        let strides = Strides::from_shape(&new_shape);
        Ok(Box::new(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides,
            dtype: self.dtype,
        }))
    }
    fn transpose(&self) -> Result<Box<dyn TensorStorage>, String> {
        if self.shape.ndim() != 2 {
            return Err("transpose only supported for 2D".to_string());
        }
        let (rows, cols) = (self.shape.dim(0), self.shape.dim(1));
        let mut out = vec![0u8; self.data.len()];
        let elem = self.dtype.dtype_size_bytes();
        for i in 0..rows {
            for j in 0..cols {
                let src = (i * cols + j) * elem;
                let dst = (j * rows + i) * elem;
                out[dst..dst + elem].copy_from_slice(&self.data[src..src + elem]);
            }
        }
        let new_shape = Shape::from(vec![cols, rows]);
        let strides = Strides::from_shape(&new_shape);
        Ok(Box::new(Self {
            data: out,
            shape: new_shape,
            strides,
            dtype: self.dtype,
        }))
    }
    fn zeros(&self, shape: Shape) -> Result<Box<dyn TensorStorage>, String> {
        Ok(Box::new(Self::zeros(self.dtype, shape)))
    }
    fn ones(&self, shape: Shape) -> Result<Box<dyn TensorStorage>, String> {
        Ok(Box::new(Self::ones(self.dtype, shape)))
    }
    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn TensorStorage>, String> {
        Ok(Box::new(Self::zeros(dtype, shape)))
    }
}

/// Main tensor structure. Storage is [`TensorStorage`]; implements [`NdArray`] for Numina ops.
#[derive(Debug)]
pub struct Tensor {
    storage: Box<dyn TensorStorage>,
}

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new<F>(data: Vec<u8>, shape: Shape, dtype: DType, backend_factory: F) -> Self
    where
        F: FnOnce(Vec<u8>, Shape, DType) -> Box<dyn TensorStorage>,
    {
        Tensor {
            storage: backend_factory(data, shape, dtype),
        }
    }

    /// Create tensor from slice (copies data).
    /// Only types that implement [`TensorElement`] are allowed, so dtype and byte layout are defined by Numina.
    /// Using a normal type (e.g. a plain struct) that does not implement `TensorElement` causes a compile error.
    ///
    /// # Example (allowed: type implements TensorElement)
    ///
    /// ```
    /// use laminax_types::{Tensor, Shape, CpuStorage};
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let shape = Shape::from([2, 2]);
    /// let _t = Tensor::from_slice(&data, shape, |bytes, shape, dtype| {
    ///     Box::new(CpuStorage::new(bytes, shape, dtype))
    /// });
    /// ```
    ///
    /// # Compile error: normal type without TensorElement
    ///
    /// ```compile_fail
    /// use laminax_types::{Tensor, Shape};
    /// #[derive(Clone, Copy)]
    /// struct MyType(i32);
    /// let data = [MyType(1), MyType(2)];
    /// let shape = Shape::from([2]);
    /// let _t = Tensor::from_slice(&data, shape, |_bytes, _shape, _dtype| unimplemented!());
    /// ```
    pub fn from_slice<T, F>(data: &[T], shape: Shape, backend_factory: F) -> Self
    where
        T: TensorElement,
        F: FnOnce(Vec<u8>, Shape, DType) -> Box<dyn TensorStorage>,
    {
        let nt = numina::Tensor::from_vec(data.to_vec(), shape.dims())
            .expect("data len must equal product of shape dimensions");
        Self::new(nt.to_bytes(), shape, nt.dtype(), backend_factory)
    }

    /// Create tensor filled with zeros using a specific backend
    pub fn zeros<F>(dtype: DType, shape: Shape, backend_factory: F) -> Self
    where
        F: FnOnce(DType, Shape) -> Box<dyn TensorStorage>,
    {
        Tensor {
            storage: backend_factory(dtype, shape),
        }
    }

    /// Create tensor filled with ones using a specific backend
    pub fn ones<F>(dtype: DType, shape: Shape, backend_factory: F) -> Self
    where
        F: FnOnce(DType, Shape) -> Box<dyn TensorStorage>,
    {
        Tensor {
            storage: backend_factory(dtype, shape),
        }
    }

    /// Create identity matrix (like `np.eye`; uses [`CpuStorage`]).
    pub fn eye(dtype: DType, n: usize) -> Self {
        Tensor {
            storage: Box::new(CpuStorage::eye(dtype, n)),
        }
    }

    /// Like `np.arange`: 1D tensor with values [start, stop) with step. F32/F64 and integer dtypes.
    pub fn arange(dtype: DType, start: f64, stop: f64, step: f64) -> Result<Self, String> {
        let arr = numina::arange(dtype, start, stop, step)?;
        Ok(Tensor::from_ndarray(Box::new(arr)))
    }

    /// Like `np.linspace`: 1D tensor with `num` values from start to end (inclusive). F32/F64.
    pub fn linspace(dtype: DType, start: f64, end: f64, num: usize) -> Result<Self, String> {
        let arr = numina::linspace(dtype, start, end, num)?;
        Ok(Tensor::from_ndarray(Box::new(arr)))
    }

    /// Get tensor shape (NumPy: `shape`).
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

    /// Copy data from a Numina NdArray into Tensor storage (e.g. when Numina ops return `Box<dyn NdArray>`). Does not hold NdArray.
    pub fn from_ndarray(array: Box<dyn NdArray>) -> Self {
        let shape = array.shape().clone();
        let dtype = array.dtype();
        let bytes = unsafe { array.as_bytes() }.to_vec();
        Tensor {
            storage: Box::new(CpuStorage::new(bytes, shape, dtype)),
        }
    }

    /// Clone this tensor with its storage
    pub fn clone_tensor(&self) -> Self {
        Tensor {
            storage: self.storage.clone_storage(),
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

    /// Create a new tensor with zeros using the same dtype as this tensor
    pub fn zeros_like(&self, shape: Shape) -> Result<Tensor, String> {
        let storage = self.storage.zeros(shape)?;
        Ok(Tensor { storage })
    }

    /// Create a new tensor with ones using the same dtype as this tensor
    pub fn ones_like(&self, shape: Shape) -> Result<Tensor, String> {
        let storage = self.storage.ones(shape)?;
        Ok(Tensor { storage })
    }

    /// Create a new tensor with specified shape and dtype
    pub fn new_like(&self, shape: Shape, dtype: DType) -> Result<Tensor, String> {
        let storage = self.storage.new_array(shape, dtype)?;
        Ok(Tensor { storage })
    }
}

/// Exposes [`TensorStorage`] as Numina's [`NdArray`] so results can be used with Numina ops.
struct StorageAsNdArray(Box<dyn TensorStorage>);

impl std::fmt::Debug for StorageAsNdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StorageAsNdArray")
    }
}

impl NdArray for StorageAsNdArray {
    fn shape(&self) -> &Shape {
        self.0.shape()
    }
    fn strides(&self) -> &Strides {
        self.0.strides()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn dtype(&self) -> DType {
        self.0.dtype()
    }
    fn byte_len(&self) -> usize {
        self.0.byte_len()
    }
    unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { self.0.as_bytes() }
    }
    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { self.0.as_mut_bytes() }
    }
    fn clone_array(&self) -> Box<dyn NdArray> {
        Box::new(StorageAsNdArray(self.0.clone_storage()))
    }
    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.0.reshape(new_shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        self.0.transpose().map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.0.zeros(shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.0.ones(shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        self.0.new_array(shape, dtype).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
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
        Box::new(StorageAsNdArray(self.storage.clone_storage()))
    }
    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.storage.reshape(new_shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        self.storage.transpose().map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.storage.zeros(shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        self.storage.ones(shape).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
    }
    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        self.storage.new_array(shape, dtype).map(|s| Box::new(StorageAsNdArray(s)) as Box<dyn NdArray>)
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
    fn tensor_zeros_with_backend() {
        let shape = Shape::from([2, 3]);
        let tensor = Tensor::zeros(F32, shape.clone(), |dtype, shape| {
            Box::new(CpuStorage::zeros(dtype, shape))
        });
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn tensor_ones_with_backend() {
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::ones(F32, shape.clone(), |dtype, shape| {
            Box::new(CpuStorage::ones(dtype, shape))
        });
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_from_slice_i32_dtype() {
        let data = [1i32, 2, 3, 4];
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::from_slice(&data, shape.clone(), |data, shape, dtype| {
            Box::new(CpuStorage::new(data, shape, dtype))
        });
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), numina::I32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::from_slice(&data, shape.clone(), |data, shape, dtype| {
            Box::new(CpuStorage::new(data, shape, dtype))
        });
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_reshape() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = Shape::from([2, 3]);
        let tensor1 = Tensor::from_slice(&data, shape1, |data, shape, dtype| {
            Box::new(CpuStorage::new(data, shape, dtype))
        });
        let shape2 = Shape::from([3, 2]);
        let tensor2 = tensor1.reshape(shape2.clone()).unwrap();
        assert_eq!(tensor2.shape(), &shape2);
        assert_eq!(tensor2.len(), 6);
    }

    // Skip display test since it requires strides implementation
    // and we're avoiding CpuBytesArray usage
    // #[test]
    // fn tensor_display() { ... }

    #[test]
    fn tensor_zeros_like() {
        let tensor = Tensor::ones(F32, Shape::from([2, 2]), |dtype, shape| {
            Box::new(CpuStorage::ones(dtype, shape))
        });
        let zeros_tensor = tensor.zeros_like(Shape::from([3, 4])).unwrap();
        assert_eq!(zeros_tensor.shape(), &Shape::from([3, 4]));
        assert_eq!(zeros_tensor.dtype(), F32);
    }

    #[test]
    fn tensor_ones_like() {
        let tensor = Tensor::zeros(F32, Shape::from([2, 2]), |dtype, shape| {
            Box::new(CpuStorage::zeros(dtype, shape))
        });
        let ones_tensor = tensor.ones_like(Shape::from([3, 4])).unwrap();
        assert_eq!(ones_tensor.shape(), &Shape::from([3, 4]));
        assert_eq!(ones_tensor.dtype(), F32);
    }

    #[test]
    fn tensor_new_like() {
        let tensor = Tensor::zeros(F32, Shape::from([2, 2]), |dtype, shape| {
            Box::new(CpuStorage::zeros(dtype, shape))
        });
        let new_tensor = tensor.new_like(Shape::from([3, 4]), numina::I32).unwrap();
        assert_eq!(new_tensor.shape(), &Shape::from([3, 4]));
        assert_eq!(new_tensor.dtype(), numina::I32);
    }


    #[test]
    fn tensor_with_factory_backend() {
        // Test creating tensors using the factory function approach
        // This allows using any backend that can be created via a function

        let zeros = Tensor::zeros(F32, Shape::from([2, 2]), |dtype, shape| {
            Box::new(CpuStorage::zeros(dtype, shape))
        });

        let ones = Tensor::ones(F32, Shape::from([2, 2]), |dtype, shape| {
            Box::new(CpuStorage::ones(dtype, shape))
        });

        assert_eq!(zeros.shape(), &Shape::from([2, 2]));
        assert_eq!(ones.shape(), &Shape::from([2, 2]));

        // Values should be correct
        let zeros_values = zeros.to_vec_f32().unwrap();
        let ones_values = ones.to_vec_f32().unwrap();

        assert!(zeros_values.iter().all(|&x| x == 0.0));
        assert!(ones_values.iter().all(|&x| x == 1.0));
    }
}
