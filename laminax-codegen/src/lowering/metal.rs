//! LCIR → Metal lowering.

use crate::lowering::LowerToTarget;
use crate::CodegenError;

/// Metal lowering implementation
pub struct MetalLowerer;

impl MetalLowerer {
    pub fn new() -> Self {
        Self
    }
}

impl LowerToTarget for MetalLowerer {
    fn lower_lcir(&self, kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
        lower_lcir_to_metal(kernel)
    }

    fn target_name(&self) -> &'static str {
        "Metal"
    }
}

/// Lower LCIR kernel to Metal shader source
pub fn lower_lcir_to_metal(_kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
    // TODO: Implement Metal lowering
    // This would convert LCIR operations to Metal Shading Language (MSL)
    Err(CodegenError::NotImplemented(
        "Metal lowering not yet implemented",
    ))
}

