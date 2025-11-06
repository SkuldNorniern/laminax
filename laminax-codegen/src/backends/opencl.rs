//! OpenCL backend for heterogeneous compute.

use crate::backends::{Backend, BackendCapabilities};
use crate::CodegenError;

/// OpenCL backend implementation
pub struct OpenClBackend;

impl OpenClBackend {
    pub fn new() -> Self {
        Self
    }

    /// Compile OpenCL kernel source
    pub fn compile_kernel(&self, _source: &str) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        Err(CodegenError::NotImplemented(
            "OpenCL kernel compilation not yet implemented",
        ))
    }

    /// Compile from LCIR to OpenCL kernel
    pub fn compile_from_lcir(&self, _kernel: &laminax::lcir::Kernel) -> std::result::Result<Vec<u8>, crate::CodegenError> {
        let source = crate::lowering::spirv::lower_lcir_to_opencl(_kernel)?;
        self.compile_kernel(&source)
    }
}

impl Backend for OpenClBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_fp64: false, // Optional in OpenCL
            supports_fp16: true,
            supports_int64: true,
            supports_int16: true,
            supports_int8: true,
            supports_async: true,  // OpenCL command queues
            unified_memory: false, // Separate host/device memory
            shared_memory: true,   // OpenCL local memory
        }
    }

    fn is_available(&self) -> bool {
        // TODO: Check for OpenCL runtime availability
        false // Placeholder
    }

    fn name(&self) -> &'static str {
        "OpenCL"
    }
}

