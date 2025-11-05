//! LCIR → CUDA/HIP lowering.

use crate::lowering::LowerToTarget;
use crate::{CodegenError, Result};

/// CUDA/HIP lowering implementation
pub struct CudaLowerer;

impl CudaLowerer {
    pub fn new() -> Self {
        Self
    }
}

impl LowerToTarget for CudaLowerer {
    fn lower_lcir(&self, kernel: &laminax::lcir::Kernel) -> Result<String> {
        lower_lcir_to_cuda(kernel)
    }

    fn target_name(&self) -> &'static str {
        "CUDA"
    }
}

/// Lower LCIR kernel to CUDA/HIP kernel source
pub fn lower_lcir_to_cuda(_kernel: &laminax::lcir::Kernel) -> Result<String> {
    // TODO: Implement CUDA lowering
    // This would convert LCIR operations to CUDA kernel syntax
    Err(CodegenError::NotImplemented("CUDA lowering not yet implemented"))
}
