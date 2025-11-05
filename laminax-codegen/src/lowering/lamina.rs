//! LCIR → Lamina IR lowering using the Lamina IR builder API.

use crate::lowering::LowerToTarget;
use crate::{CodegenError, Result};

/// Lamina IR lowering implementation
pub struct LaminaLowerer;

impl LaminaLowerer {
    pub fn new() -> Self {
        Self
    }
}

impl LowerToTarget for LaminaLowerer {
    fn lower_lcir(&self, kernel: &laminax::lcir::Kernel) -> Result<String> {
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
pub fn lower_lcir_to_lamina(kernel: &Kernel) -> Result<String> {
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
    total_len: i64,
}

struct TensorLowerInfo {
    param_name: &'static str,
    elem_ty: PrimitiveType,
}

impl<'a> ModuleCtx<'a> {
    fn new(kernel: &'a Kernel) -> Result<Self> {
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

        let shape = kernel
            .tensors
            .values()
            .next()
            .map(|info| info.shape.clone())
            .unwrap_or_else(|| Shape::from([1usize]));
        let total_len = total_elements(&shape)?;

        Ok(Self {
            kernel,
            builder: IRBuilder::new(),
            names,
            tensors,
            param_order,
            total_len,
        })
    }

    fn build_module(&mut self) -> Result<()> {
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

        let idx_ptr = self.names.fresh("idx_ptr");
        self.builder
            .alloc_stack(idx_ptr, Type::Primitive(PrimitiveType::I64))
            .store(
                Type::Primitive(PrimitiveType::I64),
                var(idx_ptr),
                lit_i64(0),
            );

        let loop_check = self.names.fresh("loop_check");
        let loop_body = self.names.fresh("loop_body");
        let loop_inc = self.names.fresh("loop_inc");
        let after_loop = self.names.fresh("after_loop");

        self.builder.jump(loop_check);

        // loop_check block
        self.builder.block(loop_check).load(
            self.names.fresh("idx"),
            Type::Primitive(PrimitiveType::I64),
            var(idx_ptr),
        );
        let idx_check = self.last_value_name();
        let cond_name = self.names.fresh("cond");
        self.builder.cmp(
            CmpOp::Lt,
            cond_name,
            PrimitiveType::I64,
            var(idx_check),
            lit_i64(self.total_len),
        );
        self.builder.branch(var(cond_name), loop_body, after_loop);

        // loop_body block
        self.builder.block(loop_body).load(
            self.names.fresh("idx"),
            Type::Primitive(PrimitiveType::I64),
            var(idx_ptr),
        );
        let idx_body = self.last_value_name();
        self.emit_operations(idx_body)?;
        self.builder.jump(loop_inc);

        // loop_inc block
        self.builder.block(loop_inc).load(
            self.names.fresh("idx"),
            Type::Primitive(PrimitiveType::I64),
            var(idx_ptr),
        );
        let idx_inc = self.last_value_name();
        let next_idx = self.names.fresh("idx_next");
        self.builder.binary(
            LaminaBinOp::Add,
            next_idx,
            PrimitiveType::I64,
            var(idx_inc),
            lit_i64(1),
        );
        self.builder
            .store(
                Type::Primitive(PrimitiveType::I64),
                var(idx_ptr),
                var(next_idx),
            )
            .jump(loop_check);

        // after_loop block
        self.builder.block(after_loop).ret_void();

        Ok(())
    }

    fn emit_operations(&mut self, idx_var: &'static str) -> Result<()> {
        for op in &self.kernel.operations {
            match op {
                Operation::Binary {
                    result,
                    lhs,
                    op,
                    rhs,
                } => {
                    self.emit_binary_op(idx_var, result, lhs, rhs, op.clone())?;
                }
                Operation::Unary { result, op, input } => {
                    self.emit_unary_op(idx_var, result, op.clone(), input)?;
                }
                Operation::Barrier => {
                    // No-op placeholder until Lamina exposes barriers.
                }
                _ => {
                    return Err(CodegenError::NotImplemented(
                        "unsupported LCIR operation in Lamina lowering",
                    ));
                }
            }
        }
        Ok(())
    }

    fn emit_binary_op(
        &mut self,
        idx_var: &'static str,
        result: &TensorAccess,
        lhs: &TensorAccess,
        rhs: &TensorAccess,
        op: LcBinaryOp,
    ) -> Result<()> {
        let res_elem_ty = self.tensor_info(result.tensor_id)?.elem_ty;
        let lhs_elem_ty = self.tensor_info(lhs.tensor_id)?.elem_ty;
        let rhs_elem_ty = self.tensor_info(rhs.tensor_id)?.elem_ty;

        if lhs_elem_ty != rhs_elem_ty || lhs_elem_ty != res_elem_ty {
            return Err(CodegenError::InvalidIr(
                "binary op expects matching tensor element types",
            ));
        }

        let res_ptr = self.emit_element_ptr(result.tensor_id, idx_var)?;
        let lhs_ptr = self.emit_element_ptr(lhs.tensor_id, idx_var)?;
        let rhs_ptr = self.emit_element_ptr(rhs.tensor_id, idx_var)?;

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
        idx_var: &'static str,
        result: &TensorAccess,
        op: laminax::lcir::UnaryOp,
        input: &TensorAccess,
    ) -> Result<()> {
        let res_elem_ty = self.tensor_info(result.tensor_id)?.elem_ty;
        let input_elem_ty = self.tensor_info(input.tensor_id)?.elem_ty;

        if input_elem_ty != res_elem_ty {
            return Err(CodegenError::InvalidIr(
                "unary op expects matching input and result element types",
            ));
        }

        let res_ptr = self.emit_element_ptr(result.tensor_id, idx_var)?;
        let input_ptr = self.emit_element_ptr(input.tensor_id, idx_var)?;

        let input_val = self.emit_load_value(input_elem_ty, input_ptr, "input_val");

        let lamina_op = match op {
            laminax::lcir::UnaryOp::Neg => {
                return Err(CodegenError::NotImplemented("Neg not implemented"));
            }
            laminax::lcir::UnaryOp::Exp => LaminaBinOp::Add, // Placeholder, need unary exp
            laminax::lcir::UnaryOp::Log => LaminaBinOp::Add, // Placeholder
            laminax::lcir::UnaryOp::Sqrt => LaminaBinOp::Add, // Placeholder
            laminax::lcir::UnaryOp::Sin => LaminaBinOp::Add, // Placeholder
            laminax::lcir::UnaryOp::Cos => LaminaBinOp::Add, // Placeholder
            laminax::lcir::UnaryOp::Tanh => LaminaBinOp::Add, // Placeholder
        };

        // For now, just copy input to output as placeholder
        let tmp = input_val; // self.names.fresh("tmp");
        // self.builder.unary(lamina_op, tmp, res_elem_ty, var(input_val));
        self.emit_store_value(res_elem_ty, res_ptr, tmp);
        Ok(())
    }

    fn emit_element_ptr(
        &mut self,
        tensor_id: laminax::lcir::TensorId,
        idx_var: &'static str,
    ) -> Result<&'static str> {
        let info = self.tensor_info(tensor_id)?;
        let param_name = info.param_name;
        let elem_ty = info.elem_ty;
        let ptr_name = self.names.fresh("elem_ptr");
        self.builder
            .getelementptr(ptr_name, var(param_name), var(idx_var), elem_ty);
        Ok(ptr_name)
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

    fn tensor_info(&self, tensor_id: laminax::lcir::TensorId) -> Result<&TensorLowerInfo> {
        self.tensors
            .get(&tensor_id)
            .ok_or_else(|| CodegenError::InvalidIr("tensor access references unknown tensor"))
    }

    fn last_value_name(&self) -> &'static str {
        self.names.last().expect("at least one temporary emitted")
    }
}

fn total_elements(shape: &Shape) -> Result<i64> {
    let mut total: i128 = 1;
    for &dim in shape.dims() {
        total = total
            .checked_mul(dim as i128)
            .ok_or_else(|| CodegenError::InvalidIr("shape product overflow"))?;
    }
    i64::try_from(total).map_err(|_| CodegenError::InvalidIr("shape exceeds i64 range"))
}

fn primitive_from_dtype(dtype: DType) -> Result<PrimitiveType> {
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
