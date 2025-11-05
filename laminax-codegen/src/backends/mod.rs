//! Backend implementations for different compute targets.
//!
//! Each backend provides compilation and execution capabilities for
//! specific hardware platforms (CPU, GPU, accelerators, etc.).

pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod vulkan;
pub mod opencl;
pub mod webgpu;
pub mod apple;

use crate::{CodegenError, Result};

/// Backend capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub supports_fp64: bool,
    pub supports_fp16: bool,
    pub supports_int64: bool,
    pub supports_int16: bool,
    pub supports_int8: bool,
    pub supports_async: bool,
    pub unified_memory: bool,
    pub shared_memory: bool,
}

/// Common trait for all backends
pub trait Backend {
    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Check if backend is available on this system
    fn is_available(&self) -> bool;

    /// Get backend name
    fn name(&self) -> &'static str;
}
