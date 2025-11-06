//! Metal compilation.

use crate::compilation::Compiler;
use crate::CodegenError;

/// Metal compiler implementation
pub struct MetalCompiler;

impl MetalCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl Compiler for MetalCompiler {
    fn compile(&self, _source: &str) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        // TODO: Implement Metal compilation using Metal shader compiler
        Err(CodegenError::NotImplemented(
            "Metal compilation not yet implemented",
        ))
    }

    fn name(&self) -> &'static str {
        "Metal Compiler"
    }
}

