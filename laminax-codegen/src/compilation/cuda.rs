//! CUDA/HIP compilation.

use crate::CodegenError;
use crate::compilation::Compiler;

/// CUDA/HIP compiler implementation
pub struct CudaCompiler;

impl CudaCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl Compiler for CudaCompiler {
    fn compile(&self, _source: &str) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        // TODO: Implement CUDA compilation using NVRTC or nvcc
        Err(CodegenError::NotImplemented(
            "CUDA compilation not yet implemented",
        ))
    }

    fn name(&self) -> &'static str {
        "CUDA Compiler"
    }
}
