//! CPU backend using Lamina IR compiler.

use crate::backends::{Backend, BackendCapabilities};

/// CPU backend implementation
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile Lamina IR to textual assembly for the current host CPU.
    pub fn compile_assembly(&self, ir: &str) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        let mut out = Vec::new();
        lamina::compile_lamina_ir_to_assembly(ir, &mut out)?;
        Ok(out)
    }

    /// Compile from LCIR to host assembly.
    pub fn compile_from_lcir(&self, kernel: &laminax::lcir::Kernel) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        let ir = crate::lowering::lamina::lower_lcir_to_lamina(kernel)?;
        self.compile_assembly(&ir)
    }
}

impl Backend for CpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: true,
            supports_fp16: cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"),
            supports_int64: true,
            supports_int16: true,
            supports_int8: true,
            supports_async: false, // Synchronous execution
            unified_memory: true,  // Host memory
            shared_memory: false,  // No shared memory on CPU
        }
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn name(&self) -> &'static str {
        "CPU"
    }
}
