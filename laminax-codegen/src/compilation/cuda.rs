//! CUDA/HIP compilation.

use crate::compilation::Compiler;
use crate::{CodegenError, Result};

/// CUDA/HIP compiler implementation
pub struct CudaCompiler;

impl CudaCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl Compiler for CudaCompiler {
    fn compile(&self, _source: &str) -> Result<Vec<u8>> {
        // TODO: Implement CUDA compilation using NVRTC or nvcc
        Err(CodegenError::NotImplemented("CUDA compilation not yet implemented"))
    }

    fn name(&self) -> &'static str {
        "CUDA Compiler"
    }
}
