//! Laminax Types - Tensor library powered by Numina

pub mod array;
pub mod device;
pub mod graph_node;
pub mod tensor;

// Re-export core types from numina for convenience
pub use numina::{
    Array, CpuBytesArray, DType, DTypeId, DTypeInfo, F32, F64, I32, NdArray, Shape, Strides,
};
pub use numina::{add, matmul, max, mean, min, mul, prod, sum};

// Re-export Tensor and specialized types
pub use tensor::*;

// Re-export GPU and specialized array types
pub use array::*;

// Re-export device abstraction layer
pub use device::*;

// Re-export graph node types for use with laminax-dag
pub use graph_node::{Node, Op, TensorDesc};
