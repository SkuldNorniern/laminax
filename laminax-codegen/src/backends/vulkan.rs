//! Vulkan backend for cross-platform GPU compute.

use crate::backends::{Backend, BackendCapabilities};
use crate::{CodegenError, Result};

/// Vulkan backend implementation
pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile SPIR-V shader
    pub fn compile_spirv(&self, _spirv_bytes: &[u8]) -> Result<Vec<u8>> {
        Err(CodegenError::NotImplemented("SPIR-V compilation not yet implemented"))
    }

    /// Compile from LCIR to SPIR-V
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
        let spirv = crate::lowering::spirv::lower_lcir_to_spirv(_kernel)?;
        self.compile_spirv(spirv.as_bytes())
    }
}

impl Backend for VulkanBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: true,  // Vulkan supports FP64
            supports_fp16: true,
            supports_int64: true,
            supports_int16: true,
            supports_int8: true,
            supports_async: true,  // Vulkan queues
            unified_memory: false, // Separate host/device memory
            shared_memory: true,   // Vulkan shared memory
        }
    }

    fn is_available(&self) -> bool {
        // TODO: Check for Vulkan runtime availability
        false // Placeholder
    }

    fn name(&self) -> &'static str {
        "Vulkan"
    }
}
