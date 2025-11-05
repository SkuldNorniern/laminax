//! Apple Silicon backend (Neural Engine/CoreML).

use crate::backends::{Backend, BackendCapabilities};
use crate::{CodegenError, Result};

/// Apple Neural Engine/CoreML backend
pub struct AppleBackend;

impl AppleBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile for Apple Neural Engine via CoreML
    pub fn compile_coreml(&self, _model_data: &[u8]) -> Result<Vec<u8>> {
        Err(CodegenError::NotImplemented("CoreML compilation not yet implemented"))
    }

    /// Compile from LCIR to CoreML model
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
        // For now, this would convert LCIR to CoreML format
        // Future: Use CoreML tools or MLCompute framework
        Err(CodegenError::NotImplemented("LCIR → CoreML conversion not yet implemented"))
    }
}

impl Backend for AppleBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: false, // ANE typically uses FP16/FP32
            supports_fp16: true,
            supports_int64: false, // Limited integer support
            supports_int16: false,
            supports_int8: true,   // 8-bit quantization support
            supports_async: true,  // Async neural network execution
            unified_memory: true,  // Apple Silicon unified memory
            shared_memory: false,  // No traditional shared memory
        }
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos") // CoreML/ANE only on macOS
    }

    fn name(&self) -> &'static str {
        "Apple Neural Engine"
    }
}


