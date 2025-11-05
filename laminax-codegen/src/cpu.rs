use crate::Result;

/// Compile Lamina IR to textual assembly for the current host CPU.
pub fn compile_host_assembly(ir: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    lamina::compile_lamina_ir_to_assembly(ir, &mut out)?;
    Ok(out)
}

/// Compile from LCIR to host assembly.
pub fn compile_host_from_lcir(kernel: &laminax::lcir::Kernel) -> Result<Vec<u8>> {
    let ir = crate::lowering::lower_lcir_to_lamina(kernel)?;
    compile_host_assembly(&ir)
}


