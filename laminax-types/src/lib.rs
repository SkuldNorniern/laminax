//! Laminax Types - Tensor library powered by Numina

pub mod tensor;
pub mod array;
pub mod device;
pub mod graph_node;

// Re-export core types from numina for convenience
pub use numina::{
    Array, CpuBytesArray, NdArray, Shape, Strides, DType, DTypeId, DTypeInfo, F32, F64, I32,
};
pub use numina::{add, mul, matmul, sum, mean, max, min, prod};

// Re-export Tensor and specialized types
pub use tensor::*;

// Re-export GPU and specialized array types
pub use array::*;

// Re-export device abstraction layer
pub use device::*;

// Re-export graph node types for use with laminax-dag
pub use graph_node::{Node, Op, TensorDesc};
