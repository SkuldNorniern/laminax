//! Laminax - Rust-first DSL and compiler for high-performance kernels
//!
//! Laminax provides a high-level, declarative way to express tensor computations
//! while giving fine-grained control over execution through scheduling primitives.
//! It compiles to optimized machine code via Lamina IR for CPU and CUDA/HIP for GPU.

pub mod dsl;
pub mod lcir;

// Re-export everything from our foundation crates for unified API
pub use laminax_types::*; // Tensor API and types

// Re-export from numina (excluding matmul which conflicts with DSL version)
pub use numina::{
    Array, BFloat16, Bool, CpuBytesArray, DType, DTypeCandidate, DTypeLike, F32, F64, I8, I16, I32,
    I64, NdArray, QuantizedI4, QuantizedU8, Shape, Strides, U8, U16, U32, U64, abs, acos, add,
    add_scalar, argsort, asin, atan, cos, exp, log, max, mean, min, mul, pow, prod, sign, sin,
    sort, sqrt, sum, tan, where_condition,
};

// Re-export DSL items
pub use dsl::{Computation, Schedule};

// Error types for Laminax operations
#[derive(Debug, Clone, PartialEq)]
pub enum LaminaxError {
    /// Shape mismatch between tensors
    ShapeMismatch(String),
    /// Data type mismatch
    DTypeMismatch(String),
    /// Invalid operation for given inputs
    InvalidOperation(String),
    /// Compilation error
    CompilationError(String),
    /// Runtime execution error
    RuntimeError(String),
    /// Backend-specific error
    BackendError(String),
}

impl std::fmt::Display for LaminaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaminaxError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            LaminaxError::DTypeMismatch(msg) => write!(f, "Data type mismatch: {}", msg),
            LaminaxError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            LaminaxError::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
            LaminaxError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            LaminaxError::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for LaminaxError {}

pub type Result<T> = std::result::Result<T, LaminaxError>;

// Test backend factory for lib tests
#[cfg(test)]
fn lib_test_backend_factory(data: Vec<u8>, shape: Shape, dtype: DType) -> Box<dyn NdArray> {
    struct LibTestArray { data: Vec<u8>, shape: Shape, dtype: DType }
    impl std::fmt::Debug for LibTestArray {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "LibTestArray{{ shape: {:?}, dtype: {:?} }}", self.shape, self.dtype)
        }
    }
    impl NdArray for LibTestArray {
        fn shape(&self) -> &Shape { &self.shape }
        fn strides(&self) -> &Strides {
            panic!("strides not implemented for test backend")
        }
        fn len(&self) -> usize { self.shape.len() }
        fn dtype(&self) -> DType { self.dtype }
        unsafe fn as_bytes(&self) -> &[u8] { &self.data }
        unsafe fn as_mut_bytes(&mut self) -> &mut [u8] { unimplemented!() }
        fn clone_array(&self) -> Box<dyn NdArray> {
            Box::new(LibTestArray {
                data: self.data.clone(),
                shape: self.shape.clone(),
                dtype: self.dtype,
            })
        }
        fn reshape(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
        fn transpose(&self) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
        fn zeros(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
        fn ones(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
        fn new_array(&self, _: Shape, _: DType) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
    }
    Box::new(LibTestArray { data, shape, dtype })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_reexports() {
        // Test that our re-exports work
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::zeros(F32, shape.clone(), |dtype, shape| {
            struct TestArray { shape: Shape, dtype: DType }
            impl std::fmt::Debug for TestArray {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "TestArray{{ shape: {:?}, dtype: {:?} }}", self.shape, self.dtype)
                }
            }
            impl NdArray for TestArray {
                fn shape(&self) -> &Shape { &self.shape }
                fn strides(&self) -> &Strides {
                    panic!("strides not implemented for test backend")
                }
                fn len(&self) -> usize { self.shape.len() }
                fn dtype(&self) -> DType { self.dtype }
                unsafe fn as_bytes(&self) -> &[u8] { unimplemented!() }
                unsafe fn as_mut_bytes(&mut self) -> &mut [u8] { unimplemented!() }
                fn clone_array(&self) -> Box<dyn NdArray> {
                    Box::new(TestArray {
                        shape: self.shape.clone(),
                        dtype: self.dtype,
                    })
                }
                fn reshape(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
                fn transpose(&self) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
                fn zeros(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
                fn ones(&self, _: Shape) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
                fn new_array(&self, _: Shape, _: DType) -> std::result::Result<Box<dyn NdArray>, String> { unimplemented!() }
            }
            Box::new(TestArray { shape, dtype })
        });
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
    }

    #[test]
    fn tensor_operations() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]), lib_test_backend_factory);
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]), lib_test_backend_factory);

        // Just test that tensors can be created with the new API
        assert_eq!(a.shape(), &Shape::from([2]));
        assert_eq!(b.shape(), &Shape::from([2]));
        assert_eq!(a.dtype(), F32);
        assert_eq!(b.dtype(), F32);
    }
}
