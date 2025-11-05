//! CPU compilation using Lamina backend.

use crate::Result;
use crate::compilation::Compiler;

/// CPU compiler implementation
pub struct CpuCompiler;

impl CpuCompiler {
    pub fn new() -> Self {
        Self
    }
}

impl Compiler for CpuCompiler {
    fn compile(&self, source: &str) -> Result<Vec<u8>> {
        // Use Lamina to compile to assembly
        let mut out = Vec::new();
        lamina::compile_lamina_ir_to_assembly(source, &mut out)?;
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "CPU Compiler"
    }
}
