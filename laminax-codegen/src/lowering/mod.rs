//! LCIR lowering implementations for different target formats.
//!
//! Each module provides lowering from Laminax Compute IR (LCIR)
//! to specific backend formats (Lamina IR, CUDA, Metal, SPIR-V, etc.).

pub mod cuda;
pub mod lamina;
pub mod metal;
pub mod spirv;

use crate::CodegenError;

/// Common trait for LCIR lowering targets
pub trait LowerToTarget {
    /// Lower LCIR kernel to target format
    fn lower_lcir(&self, kernel: &laminax_lcir::Kernel) -> std::result::Result<String, CodegenError>;

    /// Get target format name
    fn target_name(&self) -> &'static str;
}

