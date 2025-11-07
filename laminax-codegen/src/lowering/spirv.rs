//! LCIR → SPIR-V/OpenCL/WGSL lowering.

use crate::lowering::LowerToTarget;
use crate::CodegenError;

/// SPIR-V lowering implementation
pub struct SpirvLowerer;

impl SpirvLowerer {
    pub fn new() -> Self {
        Self
    }
}

impl LowerToTarget for SpirvLowerer {
    fn lower_lcir(&self, kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
        lower_lcir_to_spirv(kernel)
    }

    fn target_name(&self) -> &'static str {
        "SPIR-V"
    }
}

/// Lower LCIR kernel to SPIR-V binary (as hex string for now)
pub fn lower_lcir_to_spirv(_kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
    // TODO: Implement SPIR-V lowering
    // This would convert LCIR operations to SPIR-V bytecode
    Err(CodegenError::NotImplemented(
        "SPIR-V lowering not yet implemented",
    ))
}

/// Lower LCIR kernel to OpenCL kernel source
pub fn lower_lcir_to_opencl(_kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
    // TODO: Implement OpenCL lowering
    // This would convert LCIR operations to OpenCL C kernel syntax
    Err(CodegenError::NotImplemented(
        "OpenCL lowering not yet implemented",
    ))
}

/// Lower LCIR kernel to WGSL shader source
pub fn lower_lcir_to_wgsl(_kernel: &laminax_lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
    // TODO: Implement WGSL lowering
    // This would convert LCIR operations to WebGPU Shading Language
    Err(CodegenError::NotImplemented(
        "WGSL lowering not yet implemented",
    ))
}

