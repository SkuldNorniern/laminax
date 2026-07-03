//! AOT kernel execution via lamina IR → machine code → fn ptr.
//!
//! Pipeline:
//!   LCIR-Kernel
//!     → laminax_codegen: lower to Lamina IR text
//!     → lamina::parser: IR text → Module
//!     → lamina::mir::codegen: Module → MIR Module
//!     → lamina::runtime::compile_to_runtime: MIR → ExecutableMemory + fn ptr
//!     → call fn ptr with raw pointer args

use super::Result;
use super::RuntimeError;
use lamina::runtime::{compile_to_runtime, execute_jit_function, RuntimeResult};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use laminax_lcir::Kernel;

/// A compiled AOT kernel ready for repeated execution.
pub struct CompiledKernel {
    /// Executable memory (keeps code alive — must not be dropped before calls).
    _memory: RuntimeResult,
    /// Raw function pointer into the executable memory.
    pub function_ptr: *const u8,
    /// Number of pointer-sized arguments the kernel expects.
    pub arg_count: usize,
}

// SAFETY: The ExecutableMemory is only accessed through the stable function_ptr.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

/// Detect host platform for lamina compilation.
fn host_target() -> (TargetArchitecture, TargetOperatingSystem) {
    #[cfg(target_arch = "aarch64")]
    let arch = TargetArchitecture::AArch64;
    #[cfg(target_arch = "x86_64")]
    let arch = TargetArchitecture::X86_64;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    compile_error!("Unsupported architecture for lamina AOT");

    #[cfg(target_os = "macos")]
    let os = TargetOperatingSystem::MacOS;
    #[cfg(target_os = "linux")]
    let os = TargetOperatingSystem::Linux;
    #[cfg(target_os = "windows")]
    let os = TargetOperatingSystem::Windows;
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    compile_error!("Unsupported OS for lamina AOT");

    (arch, os)
}

/// Compile an LCIR-Kernel to native code via the lamina AOT pipeline.
///
/// Returns a [`CompiledKernel`] whose `function_ptr` can be called with
/// `arg_count` pointer-width integer arguments (pointer values cast to i64).
pub fn compile_kernel(kernel: &Kernel) -> Result<CompiledKernel> {
    // 1. Lower LCIR-Kernel → Lamina IR text
    let ir_text = laminax_codegen::lowering::lamina::lower_lcir_to_lamina(kernel)
        .map_err(|e| RuntimeError::Compilation(format!("LCIR → lamina IR failed: {:?}", e)))?;

    // 2. Parse Lamina IR text → Module
    let ir_module = lamina::parser::parse_module(&ir_text)
        .map_err(|e| RuntimeError::Compilation(format!("lamina parse failed: {:?}", e)))?;

    // 3. Lower IR → MIR
    let mir_module =
        lamina::mir::codegen::from_ir(&ir_module, "laminax_kernel").map_err(|e| {
            RuntimeError::Compilation(format!("lamina IR → MIR failed: {:?}", e))
        })?;

    // Count parameters from the first function in the kernel
    let arg_count = mir_module
        .functions
        .values()
        .next()
        .map(|f| f.sig.params.len())
        .unwrap_or(0);

    // 4. Compile MIR → executable memory + fn ptr
    let (arch, os) = host_target();
    let runtime_result = compile_to_runtime(&mir_module, arch, os, None).map_err(|e| {
        RuntimeError::Compilation(format!("lamina compile_to_runtime failed: {:?}", e))
    })?;

    let function_ptr = runtime_result.function_ptr;

    Ok(CompiledKernel {
        _memory: runtime_result,
        function_ptr,
        arg_count,
    })
}

/// Call a compiled kernel with pointer arguments (tensors passed as *mut f32 cast to i64).
///
/// # Safety
/// - `compiled.function_ptr` must point to valid executable code with the right ABI.
/// - Each `args[i]` must be a valid *mut f32 pointer cast to i64, valid for the kernel's lifetime.
pub unsafe fn call_kernel(compiled: &CompiledKernel, args: &[i64]) -> Result<()> {
    if args.len() != compiled.arg_count {
        return Err(RuntimeError::Execution(format!(
            "kernel expects {} args, got {}",
            compiled.arg_count,
            args.len()
        )));
    }

    // We need a Signature to drive execute_jit_function. Build a minimal one.
    // All params are i64 (pointer-sized), void return.
    let sig = make_i64_sig(compiled.arg_count);

    unsafe {
        execute_jit_function(&sig, compiled.function_ptr, Some(args), false, None).map_err(
            |e| RuntimeError::Execution(format!("kernel execution failed: {:?}", e)),
        )?;
    }
    Ok(())
}

/// Build a minimal MIR Signature with `n` i64 params and void return.
fn make_i64_sig(n: usize) -> lamina::mir::Signature {
    use lamina::mir::{MirType, Parameter, Register, ScalarType, Signature};
    Signature {
        name: "kernel".to_string(),
        params: (0..n)
            .map(|i| Parameter {
                reg: Register(format!("%arg{}", i)),
                ty: MirType::Scalar(ScalarType::I64),
            })
            .collect(),
        ret_ty: None, // void
    }
}
