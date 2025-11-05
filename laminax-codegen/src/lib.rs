//! Laminax Codegen: Lowering and compilation utilities targeting CPU/Metal/Apple NPU.
//!
//! Design goals:
//! - Keep this crate independent of the `laminax` crate to avoid cyclic deps.
//! - Accept Lamina IR as text for now; provide a trait so upstream can implement
//!   LCIR → Lamina IR lowering in their own crate.
//! - CPU backend implemented via `lamina` crate; Metal/Apple NPU exposed as stubs.

#![forbid(unsafe_code)]

mod cpu;
mod metal;
mod npu;
mod lowering;

/// Minimal error type for codegen.
#[derive(Debug)]
pub enum CodegenError {
    Lamina(String),
    UnsupportedTarget(&'static str),
    NotImplemented(&'static str),
    InvalidIr(&'static str),
}

impl From<lamina::LaminaError> for CodegenError {
    fn from(e: lamina::LaminaError) -> Self {
        CodegenError::Lamina(e.to_string())
    }
}

/// Result alias for codegen.
pub type Result<T> = std::result::Result<T, CodegenError>;

/// Target backends supported by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    AppleNpu,
}

/// Trait that upstream callers can implement to produce Lamina IR.
pub trait ToLaminaIr {
    fn to_lamina_ir(&self) -> Result<String>;
}

impl ToLaminaIr for laminax::lcir::Kernel {
    fn to_lamina_ir(&self) -> Result<String> {
        lowering::lower_lcir_to_lamina(self)
    }
}

/// Compile a Lamina IR program to a textual assembly for the current host CPU.
pub fn compile_lamina_ir_for_host_cpu(ir: &str) -> Result<String> {
    let asm_bytes = cpu::compile_host_assembly(ir)?;
    // Bytes are ASCII textual assembly. If conversion fails, treat as invalid IR.
    match String::from_utf8(asm_bytes) {
        Ok(s) => Ok(s),
        Err(_) => Err(CodegenError::InvalidIr("assembly is not valid UTF-8")),
    }
}

/// Compile a Lamina IR program to a backend-specific artifact (currently textual assembly on CPU).
pub fn compile_lamina_ir(ir: &str, backend: Backend) -> Result<Vec<u8>> {
    match backend {
        Backend::Cpu => cpu::compile_host_assembly(ir),
        Backend::Metal => metal::compile_metal(ir),
        Backend::AppleNpu => npu::compile_apple_npu(ir),
    }
}

/// Convenience: lower a `ToLaminaIr` into Lamina IR then compile for a specific backend.
pub fn lower_and_compile<T: ToLaminaIr>(lowerable: &T, backend: Backend) -> Result<Vec<u8>> {
    let ir = lowerable.to_lamina_ir()?;
    compile_lamina_ir(&ir, backend)
}

/// Compile directly from LCIR kernel.
pub fn compile_from_lcir(kernel: &laminax::lcir::Kernel, backend: Backend) -> Result<Vec<u8>> {
    let ir = lowering::lower_lcir_to_lamina(kernel)?;
    compile_lamina_ir(&ir, backend)
}


