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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_reexports() {
        // Test that our re-exports work
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::zeros(F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), F32);
    }

    #[test]
    fn tensor_operations() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]));
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]));

        let result = a.add(&b).unwrap();
        assert_eq!(result.shape(), &Shape::from([2]));
        assert_eq!(result.dtype(), F32);
    }
}
