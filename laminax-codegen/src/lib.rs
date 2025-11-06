//! Laminax Codegen: Modular backend system for heterogeneous compute.
//!
//! Provides a comprehensive, extensible code generation framework supporting:
//! - CPU execution via Lamina IR
//! - GPU compute via CUDA/HIP, Metal, Vulkan
//! - Cross-platform compute via OpenCL, WebGPU
//! - Specialized acceleration via CoreML/ANE
//!
//! ## Architecture
//!
//! The codegen system is organized into four main layers:
//!
//! 1. **Backends**: Platform-specific implementations (CPU, GPU, etc.)
//! 2. **Lowering**: LCIR → target format conversion (Lamina IR, CUDA, Metal, SPIR-V)
//! 3. **Compilation**: Source → binary artifact compilation
//! 4. **Common**: Shared utilities and type mappings
//!
//! ## Usage
//!
//! ```rust
//! use laminax_codegen::{Backend, compile_from_lcir};
//!
//! // Compile LCIR kernel to CPU assembly
//! let cpu_code = compile_from_lcir(&kernel, Backend::Cpu)?;
//!
//! // Compile LCIR kernel to Metal shader
//! let metal_code = compile_from_lcir(&kernel, Backend::Metal)?;
//! ```

#![forbid(unsafe_code)]

// Core modules
pub mod backends;
pub mod common;
pub mod compilation;
pub mod lowering;

// Re-export key traits
pub use compilation::Compiler;

/// Minimal error type for codegen.
#[derive(Debug)]
pub enum CodegenError {
    Lamina(String),
    UnsupportedTarget(&'static str),
    NotImplemented(&'static str),
    InvalidIr(&'static str),
    UnsupportedType {
        backend: &'static str,
        dtype: laminax::DType,
        reason: &'static str,
    },
}

impl From<lamina::LaminaError> for CodegenError {
    fn from(e: lamina::LaminaError) -> Self {
        CodegenError::Lamina(e.to_string())
    }
}


/// Target backends supported by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    AppleNpu,
}

/// Trait that upstream callers can implement to produce Lamina IR.
pub trait ToLaminaIr {
    fn to_lamina_ir(&self) -> std::result::Result<String, CodegenError>;
}

impl ToLaminaIr for laminax::lcir::Kernel {
    fn to_lamina_ir(&self) -> std::result::Result<String, CodegenError> {
        lowering::lamina::lower_lcir_to_lamina(self)
    }
}

/// Compile a Lamina IR program to a textual assembly for the current host CPU.
pub fn compile_lamina_ir_for_host_cpu(ir: &str) -> std::result::Result<String, CodegenError> {
    let compiler = compilation::cpu::CpuCompiler::new();
    let asm_bytes = compiler.compile(ir)?;
    // Bytes are ASCII textual assembly. If conversion fails, treat as invalid IR.
    match String::from_utf8(asm_bytes) {
        Ok(s) => Ok(s),
        Err(_) => Err(CodegenError::InvalidIr("assembly is not valid UTF-8")),
    }
}

/// Compile a Lamina IR program to a backend-specific artifact.
pub fn compile_lamina_ir(ir: &str, backend: Backend) -> std::result::Result<Vec<u8>, CodegenError> {
    match backend {
        Backend::Cpu => {
            let compiler = compilation::cpu::CpuCompiler::new();
            compiler.compile(ir)
        }
        Backend::Metal => {
            // This is a placeholder - real implementation would parse IR
            Err(CodegenError::NotImplemented(
                "Metal compilation from Lamina IR not yet implemented",
            ))
        }
        Backend::AppleNpu => Err(CodegenError::NotImplemented(
            "Apple NPU compilation from Lamina IR not yet implemented",
        )),
    }
}

/// Convenience: lower a `ToLaminaIr` into Lamina IR then compile for a specific backend.
pub fn lower_and_compile<T: ToLaminaIr>(lowerable: &T, backend: Backend) -> std::result::Result<Vec<u8>, CodegenError> {
    let ir = lowerable.to_lamina_ir()?;
    compile_lamina_ir(&ir, backend)
}

/// Compile directly from LCIR kernel to backend-specific binary.
pub fn compile_from_lcir(kernel: &laminax::lcir::Kernel, backend: Backend) -> std::result::Result<Vec<u8>, CodegenError> {
    match backend {
        Backend::Cpu => {
            let backend = backends::cpu::CpuBackend::new();
            backend.compile_from_lcir(kernel)
        }
        Backend::Metal => {
            let backend = backends::metal::MetalBackend::new();
            backend.compile_from_lcir(kernel)
        }
        Backend::AppleNpu => {
            let backend = backends::apple::AppleBackend::new();
            backend.compile_from_lcir(kernel)
        }
    }
}
