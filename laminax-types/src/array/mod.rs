//! Backend-specific array implementations.
//!
//! This module provides concrete `NdArray` implementations for various compute backends:
//! - **GPU Arrays**: CUDA/HIP, Metal, ROCm for GPU acceleration
//! - **TPU Arrays**: Google TPU, Coral TPU for ML acceleration
//! - **Specialized Arrays**: FPGA, custom accelerators
//!
//! Each backend provides different capabilities, memory models, and performance characteristics.
//! The `NdArray` trait ensures unified interface across all backends.

pub mod coral;
pub mod cuda;
pub mod gpu;
pub mod metal;
pub mod rocm;
pub mod tpu;

// Re-export common array types
pub use coral::{CoralArray, CoralDevice};
pub use cuda::{CudaArray, CudaDevice};
pub use gpu::{GpuArray, GpuDevice};
pub use metal::{MetalArray, MetalDevice};
pub use rocm::{RocmArray, RocmDevice};
pub use tpu::{TpuArray, TpuDevice};

// Re-export device types from the device module
pub use crate::device::{Device, DeviceCapabilities, DeviceType};
