//! Metal backend for Apple GPUs.

use crate::backends::{Backend, BackendCapabilities};
use crate::{CodegenError, Result};

/// Metal backend implementation
pub struct MetalBackend;

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile Metal shader source to Metal library
    pub fn compile_shader(&self, _source: &str) -> Result<Vec<u8>> {
        Err(CodegenError::NotImplemented("Metal shader compilation not yet implemented"))
    }

    /// Compile from LCIR to Metal shader
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
        let source = crate::lowering::metal::lower_lcir_to_metal(_kernel)?;
        self.compile_shader(&source)
    }
}

impl Backend for MetalBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: false, // Metal doesn't support FP64
            supports_fp16: true,
            supports_int64: true,
            supports_int16: true,
            supports_int8: true,
            supports_async: true,  // Metal supports async compute
            unified_memory: cfg!(target_os = "macos"), // Unified memory on Apple Silicon
            shared_memory: true,   // Metal shared memory
        }
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos") // Metal is only available on macOS
    }

    fn name(&self) -> &'static str {
        "Metal"
    }
}


