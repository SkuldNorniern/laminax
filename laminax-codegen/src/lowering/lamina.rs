//! LCIR → Lamina IR lowering using the Lamina IR builder API.

use crate::lowering::LowerToTarget;
use crate::CodegenError;

/// Lamina IR lowering implementation
pub struct LaminaLowerer;

impl LaminaLowerer {
    pub fn new() -> Self {
        Self
    }
}

impl LowerToTarget for LaminaLowerer {
    fn lower_lcir(&self, kernel: &laminax::lcir::Kernel) -> std::result::Result<String, crate::CodegenError> {
        lower_lcir_to_lamina(kernel)
    }

    fn target_name(&self) -> &'static str {
        "Lamina IR"
    }
}

use std::collections::HashMap;

use lamina::ir::builder::{i64 as lit_i64, var};
use lamina::ir::{
    BinaryOp as LaminaBinOp, CmpOp, FunctionParameter, IRBuilder, PrimitiveType, Type,
};
use laminax::lcir::{BinaryOp as LcBinaryOp, Kernel, MemoryScope, Operation, TensorAccess};
use laminax::{DType, Shape};

/// Lower an LCIR kernel into textual Lamina IR.
pub fn lower_lcir_to_lamina(kernel: &Kernel) -> std::result::Result<String, crate::CodegenError> {
    let mut ctx = ModuleCtx::new(kernel)?;
    ctx.build_module()?;
    Ok(format!("{}", ctx.builder.build()))
}

struct ModuleCtx<'a> {
    kernel: &'a Kernel,
    builder: IRBuilder<'static>,
    names: NamePool,
    tensors: HashMap<laminax::lcir::TensorId, TensorLowerInfo>,
    param_order: Vec<laminax::lcir::TensorId>,
}

struct TensorLowerInfo {
    param_name: &'static str,
    elem_ty: PrimitiveType,
}

impl<'a> ModuleCtx<'a> {
    fn new(kernel: &'a Kernel) -> std::result::Result<Self, crate::CodegenError> {
        if kernel.tensors.is_empty() {
            return Err(CodegenError::InvalidIr("kernel has no tensors"));
        }

        let mut names = NamePool::default();
        let mut tensors = HashMap::new();
        let mut param_order = Vec::new();

        let mut tensor_entries: Vec<_> = kernel.tensors.iter().collect();
        tensor_entries.sort_by_key(|(id, _)| id.0);

        for (tid, info) in tensor_entries {
            if info.scope != MemoryScope::Global {
                return Err(CodegenError::NotImplemented(
                    "only global tensors are supported in Lamina lowering",
                ));
            }
            let param_name = names.fresh(&format!("arg{}", tid.0));
            let elem_ty = primitive_from_dtype(info.dtype)?;
            tensors.insert(
                *tid,
                TensorLowerInfo {
                    param_name,
                    elem_ty,
                },
            );
            param_order.push(*tid);
        }

        Ok(Self {
            kernel,
            builder: IRBuilder::new(),
            names,
            tensors,
            param_order,
        })
    }

    fn build_module(&mut self) -> std::result::Result<(), crate::CodegenError> {
        let func_name = if self.kernel.name.is_empty() {
            self.names.fresh("kernel")
        } else {
            self.names.intern(sanitize(&self.kernel.name))
        };

        let params: Vec<FunctionParameter<'static>> = self
            .param_order
            .iter()
            .map(|tid| FunctionParameter {
                name: self.tensors[tid].param_name,
                ty: Type::Primitive(PrimitiveType::Ptr),
            })
            .collect();

        self.builder
            .function_with_params(func_name, params, Type::Void);

        // Create loop variables for each loop in the kernel
        let mut loop_vars = Vec::new();
        for loop_info in &self.kernel.loops {
            let loop_var = self.names.fresh(&format!("loop_{}", loop_info.name));
            loop_vars.push(loop_var);
        }

        // Generate nested loop structure according to loop_nest
        self.emit_nested_loops(&self.kernel.loop_nest, &loop_vars)?;

        self.builder.ret_void();

        Ok(())
    }

    fn emit_nested_loops(&mut self, loop_nest: &[laminax::lcir::LoopId], loop_vars: &[&'static str]) -> std::result::Result<(), crate::CodegenError> {
        if loop_nest.is_empty() {
            // No loops, just emit operations in the innermost scope
            self.emit_operations(loop_vars)?;
            return Ok(());
        }

        self.emit_nested_loops_recursive(loop_nest, loop_vars, 0)
    }

    fn emit_nested_loops_recursive(&mut self, loop_nest: &[laminax::lcir::LoopId], loop_vars: &[&'static str], depth: usize) -> std::result::Result<(), crate::CodegenError> {
        if depth >= loop_nest.len() {
            // Innermost loop body - emit operations
            self.emit_operations(loop_vars)?;
            return Ok(());
        }

        let loop_id = loop_nest[depth];
        let loop_info = &self.kernel.loops.iter().find(|l| l.id == loop_id)
            .ok_or_else(|| CodegenError::InvalidIr("loop_nest references unknown loop"))?;
        let loop_var = loop_vars[loop_id.0];

        // Initialize loop variable
        self.builder
            .alloc_stack(loop_var, Type::Primitive(PrimitiveType::I64))
            .store(
                Type::Primitive(PrimitiveType::I64),
                var(loop_var),
                lit_i64(loop_info.start),
            );

        let loop_check = self.names.fresh(&format!("{}_check", loop_info.name));
        let loop_body = self.names.fresh(&format!("{}_body", loop_info.name));
        let loop_inc = self.names.fresh(&format!("{}_inc", loop_info.name));
        let after_loop = self.names.fresh(&format!("{}_after", loop_info.name));

        self.builder.jump(loop_check);

        // loop_check block
        self.builder.block(loop_check).load(
            self.names.fresh(&format!("{}_val", loop_info.name)),
            Type::Primitive(PrimitiveType::I64),
            var(loop_var),
        );
        let loop_val_check = self.last_value_name();
        let cond_name = self.names.fresh(&format!("{}_cond", loop_info.name));
        self.builder.cmp(
            CmpOp::Lt,
            cond_name,
            PrimitiveType::I64,
            var(loop_val_check),
            lit_i64(loop_info.end),
        );
        self.builder.branch(var(cond_name), loop_body, after_loop);

        // loop_body block - emit nested loops or operations
        self.builder.block(loop_body);
        self.emit_nested_loops_recursive(loop_nest, loop_vars, depth + 1)?;
        self.builder.jump(loop_inc);

        // loop_inc block
        self.builder.block(loop_inc).load(
            self.names.fresh(&format!("{}_inc_val", loop_info.name)),
            Type::Primitive(PrimitiveType::I64),
            var(loop_var),
        );
        let loop_val_inc = self.last_value_name();
        let next_val = self.names.fresh(&format!("{}_next", loop_info.name));
        self.builder.binary(
            LaminaBinOp::Add,
            next_val,
            PrimitiveType::I64,
            var(loop_val_inc),
            lit_i64(loop_info.step),
        );
        self.builder
            .store(
                Type::Primitive(PrimitiveType::I64),
                var(loop_var),
                var(next_val),
            )
            .jump(loop_check);

        // after_loop block
        self.builder.block(after_loop);

        Ok(())
    }

    fn emit_operations(&mut self, loop_vars: &[&'static str]) -> std::result::Result<(), crate::CodegenError> {
        for op in &self.kernel.operations {
            match op {
                Operation::Binary {
                    result,
                    lhs,
                    op,
                    rhs,
                } => {
                    self.emit_binary_op(loop_vars, result, lhs, rhs, op.clone())?;
                }
                Operation::Unary { result, op, input } => {
                    self.emit_unary_op(loop_vars, result, op.clone(), input)?;
                }
                Operation::Load { result, source } => {
                    self.emit_load_op(loop_vars, result, source)?;
                }
                Operation::Store { dest, value } => {
                    self.emit_store_op(loop_vars, dest, value)?;
                }
                Operation::Barrier => {
                    // No-op placeholder until Lamina exposes barriers.
                }
            }
        }
        Ok(())
    }

    fn emit_binary_op(
        &mut self,
        loop_vars: &[&'static str],
        result: &TensorAccess,
        lhs: &TensorAccess,
        rhs: &TensorAccess,
        op: LcBinaryOp,
    ) -> std::result::Result<(), crate::CodegenError> {
        let res_elem_ty = self.tensor_info(result.tensor_id)?.elem_ty;
        let lhs_elem_ty = self.tensor_info(lhs.tensor_id)?.elem_ty;
        let rhs_elem_ty = self.tensor_info(rhs.tensor_id)?.elem_ty;

        if lhs_elem_ty != rhs_elem_ty || lhs_elem_ty != res_elem_ty {
            return Err(CodegenError::InvalidIr(
                "binary op expects matching tensor element types",
            ));
        }

        let res_ptr = self.emit_tensor_access(result, loop_vars)?;
        let lhs_ptr = self.emit_tensor_access(lhs, loop_vars)?;
        let rhs_ptr = self.emit_tensor_access(rhs, loop_vars)?;

        let lhs_val = self.emit_load_value(lhs_elem_ty, lhs_ptr, "lhs_val");
        let rhs_val = self.emit_load_value(rhs_elem_ty, rhs_ptr, "rhs_val");

        let lamina_op = match op {
            LcBinaryOp::Add => LaminaBinOp::Add,
            LcBinaryOp::Sub => LaminaBinOp::Sub,
            LcBinaryOp::Mul => LaminaBinOp::Mul,
            LcBinaryOp::Div => LaminaBinOp::Div,
            LcBinaryOp::Min | LcBinaryOp::Max => {
                return Err(CodegenError::NotImplemented(
                    "min/max lowering not implemented",
                ));
            }
        };

        let tmp = self.names.fresh("tmp");
        self.builder
            .binary(lamina_op, tmp, res_elem_ty, var(lhs_val), var(rhs_val));
        self.emit_store_value(res_elem_ty, res_ptr, tmp);
        Ok(())
    }

    fn emit_unary_op(
        &mut self,
        loop_vars: &[&'static str],
        result: &TensorAccess,
        op: laminax::lcir::UnaryOp,
        input: &TensorAccess,
    ) -> std::result::Result<(), crate::CodegenError> {
        let res_elem_ty = self.tensor_info(result.tensor_id)?.elem_ty;
        let input_elem_ty = self.tensor_info(input.tensor_id)?.elem_ty;

        if input_elem_ty != res_elem_ty {
            return Err(CodegenError::InvalidIr(
                "unary op expects matching input and result element types",
            ));
        }

        let res_ptr = self.emit_tensor_access(result, loop_vars)?;
        let input_ptr = self.emit_tensor_access(input, loop_vars)?;

        let input_val = self.emit_load_value(input_elem_ty, input_ptr, "input_val");

        // For now, just copy input to output as placeholder for all unary operations
        // TODO: Implement actual unary operations in Lamina IR
        match op {
            laminax::lcir::UnaryOp::Neg => {
                return Err(CodegenError::NotImplemented("Neg not implemented"));
            }
            _ => {
                // Placeholder: just copy input to output
                self.emit_store_value(res_elem_ty, res_ptr, input_val);
            }
        }
        Ok(())
    }

    fn emit_load_op(
        &mut self,
        loop_vars: &[&'static str],
        result: &TensorAccess,
        source: &TensorAccess,
    ) -> std::result::Result<(), crate::CodegenError> {
        let res_elem_ty = self.tensor_info(result.tensor_id)?.elem_ty;
        let src_elem_ty = self.tensor_info(source.tensor_id)?.elem_ty;

        if src_elem_ty != res_elem_ty {
            return Err(CodegenError::InvalidIr(
                "load op expects matching source and result element types",
            ));
        }

        let res_ptr = self.emit_tensor_access(result, loop_vars)?;
        let src_ptr = self.emit_tensor_access(source, loop_vars)?;

        let src_val = self.emit_load_value(src_elem_ty, src_ptr, "load_val");
        self.emit_store_value(res_elem_ty, res_ptr, src_val);
        Ok(())
    }

    fn emit_store_op(
        &mut self,
        loop_vars: &[&'static str],
        dest: &TensorAccess,
        value: &TensorAccess,
    ) -> std::result::Result<(), crate::CodegenError> {
        let dest_elem_ty = self.tensor_info(dest.tensor_id)?.elem_ty;
        let val_elem_ty = self.tensor_info(value.tensor_id)?.elem_ty;

        if val_elem_ty != dest_elem_ty {
            return Err(CodegenError::InvalidIr(
                "store op expects matching value and destination element types",
            ));
        }

        let dest_ptr = self.emit_tensor_access(dest, loop_vars)?;
        let val_ptr = self.emit_tensor_access(value, loop_vars)?;

        let val = self.emit_load_value(val_elem_ty, val_ptr, "store_val");
        self.emit_store_value(dest_elem_ty, dest_ptr, val);
        Ok(())
    }

    fn emit_tensor_access(
        &mut self,
        access: &TensorAccess,
        loop_vars: &[&'static str],
    ) -> std::result::Result<&'static str, crate::CodegenError> {
        let info = self.tensor_info(access.tensor_id)?;
        let param_name = info.param_name;
        let elem_ty = info.elem_ty;

        if access.indices.is_empty() {
            // Scalar access (shouldn't happen in practice)
            return Err(CodegenError::InvalidIr("tensor access requires indices"));
        }

        // Compute the flat index from multi-dimensional indices
        let flat_index = self.emit_index_expr(&access.indices[0], loop_vars)?;

        // For multi-dimensional tensors, we need to compute: flat_index = indices[0] * stride[0] + indices[1] * stride[1] + ...
        // For simplicity, assume row-major (C-style) layout for now
        let mut current_index = flat_index;
        if access.indices.len() > 1 {
            let tensor_info = self.kernel.tensors.get(&access.tensor_id)
                .ok_or_else(|| CodegenError::InvalidIr("tensor access references unknown tensor"))?;

            for (i, index_expr) in access.indices.iter().enumerate().skip(1) {
                let index_val = self.emit_index_expr(index_expr, loop_vars)?;

                // Compute stride for this dimension
                let stride: i64 = tensor_info.shape.dims().iter().skip(i).map(|&x| x as i64).product();

                let stride_val = self.names.fresh("stride");
                self.builder
                    .alloc_stack(stride_val, Type::Primitive(PrimitiveType::I64))
                    .store(
                        Type::Primitive(PrimitiveType::I64),
                        var(stride_val),
                        lit_i64(stride),
                    );

                let contrib = self.names.fresh("index_contrib");
                self.builder.binary(
                    LaminaBinOp::Mul,
                    contrib,
                    PrimitiveType::I64,
                    var(index_val),
                    var(stride_val),
                );

                let new_index = self.names.fresh("flat_index");
                self.builder.binary(
                    LaminaBinOp::Add,
                    new_index,
                    PrimitiveType::I64,
                    var(current_index),
                    var(contrib),
                );
                current_index = new_index;
            }
        }

        // Now get the element pointer using the computed flat index
        let ptr_name = self.names.fresh("elem_ptr");
        self.builder
            .getelementptr(ptr_name, var(param_name), var(current_index), elem_ty);
        Ok(ptr_name)
    }

    fn emit_index_expr(
        &mut self,
        expr: &laminax::lcir::IndexExpr,
        loop_vars: &[&'static str],
    ) -> std::result::Result<&'static str, crate::CodegenError> {
        match expr {
            laminax::lcir::IndexExpr::Const(val) => {
                let name = self.names.fresh("const_idx");
                self.builder
                    .alloc_stack(name, Type::Primitive(PrimitiveType::I64))
                    .store(
                        Type::Primitive(PrimitiveType::I64),
                        var(name),
                        lit_i64(*val),
                    );
                Ok(name)
            }
            laminax::lcir::IndexExpr::LoopVar(loop_id) => {
                Ok(loop_vars[loop_id.0])
            }
            laminax::lcir::IndexExpr::Add(lhs, rhs) => {
                let lhs_val = self.emit_index_expr(lhs, loop_vars)?;
                let rhs_val = self.emit_index_expr(rhs, loop_vars)?;
                let result = self.names.fresh("add_result");
                self.builder.binary(
                    LaminaBinOp::Add,
                    result,
                    PrimitiveType::I64,
                    var(lhs_val),
                    var(rhs_val),
                );
                Ok(result)
            }
            laminax::lcir::IndexExpr::Sub(lhs, rhs) => {
                let lhs_val = self.emit_index_expr(lhs, loop_vars)?;
                let rhs_val = self.emit_index_expr(rhs, loop_vars)?;
                let result = self.names.fresh("sub_result");
                self.builder.binary(
                    LaminaBinOp::Sub,
                    result,
                    PrimitiveType::I64,
                    var(lhs_val),
                    var(rhs_val),
                );
                Ok(result)
            }
            laminax::lcir::IndexExpr::Mul(lhs, rhs) => {
                let lhs_val = self.emit_index_expr(lhs, loop_vars)?;
                let rhs_val = self.emit_index_expr(rhs, loop_vars)?;
                let result = self.names.fresh("mul_result");
                self.builder.binary(
                    LaminaBinOp::Mul,
                    result,
                    PrimitiveType::I64,
                    var(lhs_val),
                    var(rhs_val),
                );
                Ok(result)
            }
            laminax::lcir::IndexExpr::Div(lhs, rhs) => {
                let lhs_val = self.emit_index_expr(lhs, loop_vars)?;
                let rhs_val = self.emit_index_expr(rhs, loop_vars)?;
                let result = self.names.fresh("div_result");
                self.builder.binary(
                    LaminaBinOp::Div,
                    result,
                    PrimitiveType::I64,
                    var(lhs_val),
                    var(rhs_val),
                );
                Ok(result)
            }
        }
    }

    fn emit_load_value(
        &mut self,
        elem_ty: PrimitiveType,
        ptr_var: &'static str,
        hint: &str,
    ) -> &'static str {
        let load_name = self.names.fresh(hint);
        self.builder
            .load(load_name, Type::Primitive(elem_ty), var(ptr_var));
        load_name
    }

    fn emit_store_value(
        &mut self,
        elem_ty: PrimitiveType,
        ptr_var: &'static str,
        value_var: &'static str,
    ) {
        self.builder
            .store(Type::Primitive(elem_ty), var(ptr_var), var(value_var));
    }

    fn tensor_info(&self, tensor_id: laminax::lcir::TensorId) -> std::result::Result<&TensorLowerInfo, crate::CodegenError> {
        self.tensors
            .get(&tensor_id)
            .ok_or_else(|| CodegenError::InvalidIr("tensor access references unknown tensor"))
    }

    fn last_value_name(&self) -> &'static str {
        self.names.last().expect("at least one temporary emitted")
    }
}

fn total_elements(shape: &Shape) -> std::result::Result<i64, crate::CodegenError> {
    let mut total: i128 = 1;
    for &dim in shape.dims() {
        total = total
            .checked_mul(dim as i128)
            .ok_or_else(|| CodegenError::InvalidIr("shape product overflow"))?;
    }
    i64::try_from(total).map_err(|_| CodegenError::InvalidIr("shape exceeds i64 range"))
}

fn primitive_from_dtype(dtype: DType) -> std::result::Result<PrimitiveType, crate::CodegenError> {
    match dtype {
        DType::F32 => Ok(PrimitiveType::F32),
        DType::F64 => Ok(PrimitiveType::F64),
        DType::I8 => Ok(PrimitiveType::I8),
        DType::I16 => Ok(PrimitiveType::I16),
        DType::I32 => Ok(PrimitiveType::I32),
        DType::I64 => Ok(PrimitiveType::I64),
        DType::U8 => Ok(PrimitiveType::U8),
        DType::U16 => Ok(PrimitiveType::U16),
        DType::U32 => Ok(PrimitiveType::U32),
        DType::U64 => Ok(PrimitiveType::U64),
        DType::Bool => Ok(PrimitiveType::Bool),
        _ => Err(CodegenError::NotImplemented("unsupported tensor dtype")),
    }
}

fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "kernel".to_string()
    } else {
        out
    }
}

#[derive(Default)]
struct NamePool {
    counter: usize,
    leaked: Vec<&'static str>,
}

impl NamePool {
    fn fresh(&mut self, prefix: &str) -> &'static str {
        let name = format!("{}_{}", prefix, self.counter);
        self.counter += 1;
        self.intern(name)
    }

    fn intern<S: Into<String>>(&mut self, s: S) -> &'static str {
        let boxed = s.into().into_boxed_str();
        let ptr = Box::leak(boxed);
        self.leaked.push(ptr);
        ptr
    }

    fn last(&self) -> Option<&'static str> {
        self.leaked.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laminax::lcir::{KernelBuilder, MemoryScope, access, index};
    use laminax::{F32, Shape};

    #[test]
    fn test_simple_elementwise_add() {
        let mut builder = KernelBuilder::new("test_add");

        // Add tensors
        let a_id = builder.add_tensor("A", Shape::from([4, 4]), F32, MemoryScope::Global);
        let b_id = builder.add_tensor("B", Shape::from([4, 4]), F32, MemoryScope::Global);
        let c_id = builder.add_tensor("C", Shape::from([4, 4]), F32, MemoryScope::Global);

        // Add loops
        let i_loop = builder.add_loop("i", 0, 4, 1);
        let j_loop = builder.add_loop("j", 0, 4, 1);

        // Add operation: C[i,j] = A[i,j] + B[i,j]
        let a_access = access::global(a_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let b_access = access::global(b_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let c_access = access::global(c_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);

        builder.add_binary_op(c_access.clone(), a_access, laminax::lcir::BinaryOp::Add, b_access);

        let kernel = builder.build();

        // Lower to Lamina IR
        let lamina_ir = lower_lcir_to_lamina(&kernel).unwrap();

        // Print the generated IR for debugging
        println!("Generated Lamina IR:\n{}", lamina_ir);

        // Check that it contains expected elements
        assert!(lamina_ir.contains("function") || lamina_ir.contains("fn"));
        assert!(lamina_ir.contains("loop_i") || lamina_ir.contains("i"));
        assert!(lamina_ir.contains("loop_j") || lamina_ir.contains("j"));
        assert!(lamina_ir.contains("add") || lamina_ir.contains("+"));
    }
}
