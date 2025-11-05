//! WebGPU backend for web and cross-platform compute.

use crate::backends::{Backend, BackendCapabilities};
use crate::{CodegenError, Result};

/// WebGPU backend implementation
pub struct WebGpuBackend;

impl WebGpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile WGSL shader
    pub fn compile_wgsl(&self, _source: &str) -> Result<Vec<u8>> {
        Err(CodegenError::NotImplemented("WGSL compilation not yet implemented"))
    }

    /// Compile from LCIR to WGSL
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
        let wgsl = crate::lowering::spirv::lower_lcir_to_wgsl(_kernel)?;
        self.compile_wgsl(&wgsl)
    }
}

impl Backend for WebGpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: false, // WebGPU doesn't support FP64
            supports_fp16: true,  // WGSL supports f16
            supports_int64: false, // Limited 64-bit support
            supports_int16: false,
            supports_int8: false,
            supports_async: true,  // WebGPU is async
            unified_memory: false, // Separate host/device memory
            shared_memory: true,   // WebGPU workgroup memory
        }
    }

    fn is_available(&self) -> bool {
        // WebGPU is available in modern browsers and native via wgpu
        true // Assume available for now
    }

    fn name(&self) -> &'static str {
        "WebGPU"
    }
}
