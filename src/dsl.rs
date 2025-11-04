//! Laminax DSL - Domain-Specific Language for expressing computations
//!
//! The DSL provides ergonomic Rust syntax for tensor/array computations
//! that can be lowered to optimized LCIR kernels.

use crate::lcir::{BinaryOp, KernelBuilder, MemoryScope, TensorId, UnaryOp, access, index};
use crate::{DType, LaminaxError, Result, Shape, Tensor};

/// Core trait for DSL expressions that can be evaluated
pub trait DSLExpr {
    /// Get the shape of this expression's result
    fn shape(&self) -> &Shape;

    /// Get the data type of this expression's result
    fn dtype(&self) -> DType;

    /// Lower this expression to LCIR operations
    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()>;

    /// Evaluate this expression directly (for testing/debugging)
    fn eval(&self) -> Result<Tensor>;
}

/// Represents a scheduled computation ready for execution
pub struct Computation {
    expr: Box<dyn DSLExpr>,
    schedule: Schedule,
}

impl std::fmt::Debug for Computation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Computation")
            .field("schedule", &self.schedule)
            .field("expr", &"<DSL expression>")
            .finish()
    }
}

impl Computation {
    /// Create a new computation from an expression
    pub fn new(expr: Box<dyn DSLExpr>) -> Self {
        Self {
            expr,
            schedule: Schedule::default(),
        }
    }

    /// Apply scheduling transformations
    pub fn schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Tile a loop axis
    pub fn tile(mut self, axis: usize, size: usize) -> Self {
        self.schedule.tiles.push((axis, size));
        self
    }

    /// Parallelize a loop axis
    pub fn parallelize(mut self, axis: usize) -> Self {
        self.schedule.parallel_axes.push(axis);
        self
    }

    /// Vectorize a loop axis
    pub fn vectorize(mut self, axis: usize) -> Self {
        self.schedule.vectorized_axes.push(axis);
        self
    }

    /// Execute the computation via LCIR lowering and execution
    pub fn run(self) -> Result<Tensor> {
        // First try direct evaluation for simple cases
        if let Ok(result) = self.expr.eval() {
            return Ok(result);
        }

        // Fall back to LCIR-based execution
        self.run_via_lcir()
    }

    /// Execute via LCIR lowering (future implementation)
    fn run_via_lcir(self) -> Result<Tensor> {
        // TODO: Implement full LCIR → Lamina IR → execution pipeline
        Err(LaminaxError::InvalidOperation(
            "LCIR execution not yet implemented".to_string(),
        ))
    }
}

// ============================================================================
// EXPRESSION TYPES
// ============================================================================

/// Binary operation expression
#[derive(Debug)]
pub struct BinaryExpr {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub op: BinaryOpType,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operation expression
#[derive(Debug)]
pub struct UnaryExpr {
    pub input: Tensor,
    pub op: UnaryOpType,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOpType {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
}

/// Matrix multiplication expression
#[derive(Debug)]
pub struct MatMulExpr {
    pub lhs: Tensor,
    pub rhs: Tensor,
    result_shape: Shape,
}

// ============================================================================
// DSL TRAIT IMPLEMENTATIONS
// ============================================================================

impl DSLExpr for Tensor {
    fn shape(&self) -> &Shape {
        self.shape()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        // For tensor inputs, we just load them into LCIR
        let _input_id = builder.add_tensor(
            "input",
            self.shape().clone(),
            self.dtype(),
            MemoryScope::Global,
        );
        // TODO: Add load operation
        Ok(())
    }

    fn eval(&self) -> Result<Tensor> {
        // A tensor evaluates to itself
        Ok(self.clone_tensor())
    }
}

impl DSLExpr for BinaryExpr {
    fn shape(&self) -> &Shape {
        // For element-wise operations, result shape must match both inputs
        assert_eq!(
            self.lhs.shape(),
            self.rhs.shape(),
            "Binary operation requires matching shapes"
        );
        self.lhs.shape()
    }

    fn dtype(&self) -> DType {
        // Assume both operands have the same dtype
        assert_eq!(
            self.lhs.dtype(),
            self.rhs.dtype(),
            "Binary operation requires matching dtypes"
        );
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        let lhs_id = builder.add_tensor(
            "lhs",
            self.lhs.shape().clone(),
            self.lhs.dtype(),
            MemoryScope::Global,
        );
        let rhs_id = builder.add_tensor(
            "rhs",
            self.rhs.shape().clone(),
            self.rhs.dtype(),
            MemoryScope::Global,
        );

        // Add loops for all dimensions
        let mut loop_ids = Vec::new();
        for (i, &dim) in self.lhs.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        // Create tensor accesses
        let lhs_access = access::tensor(lhs_id, indices.clone(), MemoryScope::Global);
        let rhs_access = access::tensor(rhs_id, indices.clone(), MemoryScope::Global);
        let result_access = access::tensor(result_tensor, indices, MemoryScope::Global);

        // Add binary operation
        let op = match self.op {
            BinaryOpType::Add => BinaryOp::Add,
            BinaryOpType::Sub => BinaryOp::Sub,
            BinaryOpType::Mul => BinaryOp::Mul,
            BinaryOpType::Div => BinaryOp::Div,
        };

        builder.add_binary_op(result_access, lhs_access, op, rhs_access);
        Ok(())
    }

    fn eval(&self) -> Result<Tensor> {
        // Direct evaluation using numina operations
        match self.op {
            BinaryOpType::Add => self
                .lhs
                .add(&self.rhs)
                .map_err(LaminaxError::InvalidOperation),
            BinaryOpType::Sub => {
                // TODO: Implement subtraction in Tensor
                Err(LaminaxError::InvalidOperation(
                    "Subtraction not yet implemented".to_string(),
                ))
            }
            BinaryOpType::Mul => self
                .lhs
                .mul(&self.rhs)
                .map_err(LaminaxError::InvalidOperation),
            BinaryOpType::Div => {
                // TODO: Implement division in Tensor
                Err(LaminaxError::InvalidOperation(
                    "Division not yet implemented".to_string(),
                ))
            }
        }
    }
}

impl DSLExpr for UnaryExpr {
    fn shape(&self) -> &Shape {
        self.input.shape()
    }

    fn dtype(&self) -> DType {
        self.input.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        let input_id = builder.add_tensor(
            "input",
            self.input.shape().clone(),
            self.input.dtype(),
            MemoryScope::Global,
        );

        // Add loops for all dimensions
        let mut loop_ids = Vec::new();
        for (i, &dim) in self.input.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        // Create tensor accesses
        let input_access = access::tensor(input_id, indices.clone(), MemoryScope::Global);
        let result_access = access::tensor(result_tensor, indices, MemoryScope::Global);

        // Add unary operation
        let op = match self.op {
            UnaryOpType::Exp => UnaryOp::Exp,
            UnaryOpType::Log => UnaryOp::Log,
            UnaryOpType::Sqrt => UnaryOp::Sqrt,
            UnaryOpType::Sin => UnaryOp::Sin,
            UnaryOpType::Cos => UnaryOp::Cos,
            UnaryOpType::Tanh => UnaryOp::Tanh,
        };

        builder.add_unary_op(result_access, op, input_access);
        Ok(())
    }

    fn eval(&self) -> Result<Tensor> {
        // Direct evaluation using numina operations
        match self.op {
            UnaryOpType::Exp => self
                .input
                .exp()
                .map_err(LaminaxError::InvalidOperation),
            UnaryOpType::Log => self
                .input
                .log()
                .map_err(LaminaxError::InvalidOperation),
            UnaryOpType::Sqrt => self
                .input
                .sqrt()
                .map_err(LaminaxError::InvalidOperation),
            _ => Err(LaminaxError::InvalidOperation(format!(
                "Unary operation {:?} not implemented",
                self.op
            ))),
        }
    }
}

impl DSLExpr for MatMulExpr {
    fn shape(&self) -> &Shape {
        &self.result_shape
    }

    fn dtype(&self) -> DType {
        assert_eq!(
            self.lhs.dtype(),
            self.rhs.dtype(),
            "Matrix multiplication requires matching dtypes"
        );
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        let lhs_id = builder.add_tensor(
            "lhs",
            self.lhs.shape().clone(),
            self.lhs.dtype(),
            MemoryScope::Global,
        );
        let rhs_id = builder.add_tensor(
            "rhs",
            self.rhs.shape().clone(),
            self.rhs.dtype(),
            MemoryScope::Global,
        );

        let lhs_dims = self.lhs.shape().dims();
        let rhs_dims = self.rhs.shape().dims();

        // Add loops: i (rows), j (cols), k (reduction dim)
        let i_loop = builder.add_loop("i", 0, lhs_dims[0] as i64, 1);
        let j_loop = builder.add_loop("j", 0, rhs_dims[1] as i64, 1);
        let k_loop = builder.add_loop("k", 0, lhs_dims[1] as i64, 1);

        // Create tensor accesses
        let lhs_access = access::tensor(
            lhs_id,
            vec![index::loop_var(i_loop), index::loop_var(k_loop)],
            MemoryScope::Global,
        );
        let rhs_access = access::tensor(
            rhs_id,
            vec![index::loop_var(k_loop), index::loop_var(j_loop)],
            MemoryScope::Global,
        );
        let result_access = access::tensor(
            result_tensor,
            vec![index::loop_var(i_loop), index::loop_var(j_loop)],
            MemoryScope::Global,
        );

        // Add multiplication and accumulation
        builder.add_binary_op(result_access.clone(), lhs_access, BinaryOp::Mul, rhs_access);

        // TODO: Add reduction (accumulation) over k dimension
        // This requires more complex LCIR operations

        Ok(())
    }

    fn eval(&self) -> Result<Tensor> {
        // Use laminax matmul function
        crate::matmul(&self.lhs, &self.rhs)
            .map_err(LaminaxError::InvalidOperation)
            .map(Tensor::from_ndarray)
    }
}

// ============================================================================
// DSL TRAIT FOR TENSOR - CLEAN API
// ============================================================================

/// Extension trait providing DSL methods for Tensor
pub trait TensorDSL {
    /// Element-wise addition
    fn dsl_add(self, rhs: Tensor) -> Computation;
    /// Element-wise subtraction
    fn dsl_sub(self, rhs: Tensor) -> Computation;
    /// Element-wise multiplication
    fn dsl_mul(self, rhs: Tensor) -> Computation;
    /// Element-wise division
    fn dsl_div(self, rhs: Tensor) -> Computation;

    /// Matrix multiplication
    fn dsl_matmul(self, rhs: Tensor) -> Computation;

    /// Unary operations
    fn dsl_exp(self) -> Computation;
    fn dsl_log(self) -> Computation;
    fn dsl_sqrt(self) -> Computation;
}

impl TensorDSL for Tensor {
    fn dsl_add(self, rhs: Tensor) -> Computation {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: BinaryOpType::Add,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_sub(self, rhs: Tensor) -> Computation {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: BinaryOpType::Sub,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_mul(self, rhs: Tensor) -> Computation {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: BinaryOpType::Mul,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_div(self, rhs: Tensor) -> Computation {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: BinaryOpType::Div,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_matmul(self, rhs: Tensor) -> Computation {
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let result_shape = Shape::from([lhs_dims[0], rhs_dims[1]]);

        let expr = MatMulExpr {
            lhs: self,
            rhs,
            result_shape,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_exp(self) -> Computation {
        let expr = UnaryExpr {
            input: self,
            op: UnaryOpType::Exp,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_log(self) -> Computation {
        let expr = UnaryExpr {
            input: self,
            op: UnaryOpType::Log,
        };
        Computation::new(Box::new(expr))
    }

    fn dsl_sqrt(self) -> Computation {
        let expr = UnaryExpr {
            input: self,
            op: UnaryOpType::Sqrt,
        };
        Computation::new(Box::new(expr))
    }
}

// ============================================================================
// SCHEDULING TRANSFORMATIONS
// ============================================================================

/// Scheduling transformations for optimization
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct Schedule {
    pub tiles: Vec<(usize, usize)>, // (axis, size)
    pub parallel_axes: Vec<usize>,
    pub vectorized_axes: Vec<usize>,
}


impl Schedule {
    /// Create a new schedule
    pub fn new() -> Self {
        Self::default()
    }

    /// Tile a loop axis
    pub fn tile(mut self, axis: usize, size: usize) -> Self {
        self.tiles.push((axis, size));
        self
    }

    /// Parallelize a loop axis
    pub fn parallelize(mut self, axis: usize) -> Self {
        self.parallel_axes.push(axis);
        self
    }

    /// Vectorize a loop axis
    pub fn vectorize(mut self, axis: usize) -> Self {
        self.vectorized_axes.push(axis);
        self
    }

    /// Apply scheduling to LCIR builder
    fn apply(&self, builder: &mut KernelBuilder) -> Result<()> {
        // Apply parallelization
        for &axis in &self.parallel_axes {
            if let Some(loop_info) = builder.loops_mut().get_mut(axis) {
                let loop_id = loop_info.id;
                builder.set_parallel(loop_id, true);
            }
        }

        // Apply vectorization
        for &axis in &self.vectorized_axes {
            if let Some(loop_info) = builder.loops_mut().get_mut(axis) {
                let loop_id = loop_info.id;
                builder.set_vectorized(loop_id, true);
            }
        }

        // TODO: Implement tiling
        for &(axis, size) in &self.tiles {
            // This would require splitting loops and adjusting bounds
            // For now, just mark as TODO
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn dsl_binary_operations() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]));
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]));

        // Test addition
        let computation = a.clone_tensor().dsl_add(b.clone_tensor());
        assert_eq!(computation.expr.shape().dims(), &[2]);

        // Test that it runs without error
        let result = computation.run().unwrap();
        assert_eq!(result.shape().dims(), &[2]);
    }

    #[test]
    fn dsl_matmul() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2]));
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2]));

        // Test matmul DSL
        let computation = a.dsl_matmul(b);
        assert_eq!(computation.expr.shape().dims(), &[2, 2]);

        // Test that it runs without error
        let result = computation.run().unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn dsl_unary_operations() {
        let a = Tensor::from_slice(&[1.0f32, 4.0], Shape::from([2]));

        // Test exp
        let computation = a.clone_tensor().dsl_exp();
        assert_eq!(computation.expr.shape().dims(), &[2]);

        // Test sqrt
        let computation = a.dsl_sqrt();
        assert_eq!(computation.expr.shape().dims(), &[2]);
    }

    #[test]
    fn dsl_scheduling() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]));
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]));

        // Test scheduling
        let computation = a.dsl_add(b).parallelize(0).vectorize(0);

        assert_eq!(computation.schedule.parallel_axes, vec![0]);
        assert_eq!(computation.schedule.vectorized_axes, vec![0]);
    }
}
