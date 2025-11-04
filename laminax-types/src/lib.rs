//! Laminax Types - Tensor library powered by Numina

pub mod tensor;

// Re-export core types from numina for convenience
pub use numina::{Array, CpuBytesArray, NdArray, Shape, Strides, DType, F32, F64};
pub use numina::{add, mul, matmul, sum, mean, max, min, prod};

// Re-export Tensor and specialized types
pub use tensor::*;
