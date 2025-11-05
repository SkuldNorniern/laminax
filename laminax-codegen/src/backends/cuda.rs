//! CUDA/HIP backend for NVIDIA/AMD GPUs.

use crate::backends::{Backend, BackendCapabilities};
use crate::{CodegenError, Result};

/// CUDA/HIP backend implementation
pub struct CudaBackend;

impl CudaBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile CUDA/HIP kernel source
    pub fn compile_kernel(&self, _source: &str) -> Result<Vec<u8>> {
        Err(CodegenError::NotImplemented(
            "CUDA kernel compilation not yet implemented",
        ))
    }

    /// Compile from LCIR to CUDA/HIP kernel
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
        let source = crate::lowering::cuda::lower_lcir_to_cuda(_kernel)?;
        self.compile_kernel(&source)
    }
}

impl Backend for CudaBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: true,
            supports_fp16: true,
            supports_int64: true,
            supports_int16: false, // Limited support
            supports_int8: false,  // Limited support
            supports_async: true,  // CUDA streams
            unified_memory: false, // Separate host/device memory
            shared_memory: true,   // CUDA shared memory
        }
    }

    fn is_available(&self) -> bool {
        // TODO: Check for CUDA/HIP runtime availability
        false // Placeholder
    }

    fn name(&self) -> &'static str {
        "CUDA"
    }
}
