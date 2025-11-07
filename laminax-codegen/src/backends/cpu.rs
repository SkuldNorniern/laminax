//! CPU backend using Lamina IR compiler.

use crate::backends::{Backend, BackendCapabilities};
use crate::CodegenError;

/// CPU backend implementation
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile Lamina IR to textual assembly for the current host CPU.
    pub fn compile_assembly(&self, ir: &str) -> Result<Vec<u8>, CodegenError> {
        let mut out = Vec::new();
        lamina::compile_lamina_ir_to_assembly(ir, &mut out)?;
        Ok(out)
    }

    /// Compile from LCIR to host assembly.
    pub fn compile_from_lcir(&self, kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>, CodegenError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use laminax::lcir::{KernelBuilder, MemoryScope, access, index};
    use laminax::{F32, Shape};

    #[test]
    fn test_cpu_backend_compilation() {
        let mut builder = KernelBuilder::new("test_add");

        // Add tensors - use I32 instead of F32 since Lamina may not support F32 loads yet
        let a_id = builder.add_tensor("A", Shape::from([4, 4]), laminax::I32, MemoryScope::Global);
        let b_id = builder.add_tensor("B", Shape::from([4, 4]), laminax::I32, MemoryScope::Global);
        let c_id = builder.add_tensor("C", Shape::from([4, 4]), laminax::I32, MemoryScope::Global);

        // Add loops
        let i_loop = builder.add_loop("i", 0, 4, 1);
        let j_loop = builder.add_loop("j", 0, 4, 1);

        // Add operation: C[i,j] = A[i,j] + B[i,j]
        let a_access = access::global(a_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let b_access = access::global(b_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let c_access = access::global(c_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);

        builder.add_binary_op(c_access.clone(), a_access, laminax::lcir::BinaryOp::Add, b_access);

        let kernel = builder.build();

        // Test compilation through CPU backend
        let backend = CpuBackend::new();
        let assembly = backend.compile_from_lcir(&kernel).unwrap();

        // Verify we got assembly output
        assert!(!assembly.is_empty());
        println!("Generated assembly (first 500 chars):\n{}", 
            String::from_utf8_lossy(&assembly[..assembly.len().min(500)]));
    }
}
