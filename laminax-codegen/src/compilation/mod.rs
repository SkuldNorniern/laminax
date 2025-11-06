//! Compilation pipelines for different backends.
//!
//! Each module handles the compilation of lowered code
//! to executable artifacts for specific compute platforms.

pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod shader;


/// Common trait for compilation targets
pub trait Compiler {
    /// Compile source code to binary artifact
    fn compile(&self, source: &str) -> std::result::Result<Vec<u8>, crate::CodegenError>;

    /// Get compiler name
    fn name(&self) -> &'static str;
}

