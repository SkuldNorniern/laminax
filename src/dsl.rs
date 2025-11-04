//! Laminax DSL - Domain-Specific Language for expressing computations
//!
//! The DSL provides ergonomic Rust syntax for tensor/array computations
//! that can be lowered to optimized LCIR kernels.

use crate::lcir::{KernelBuilder, MemoryScope, BinaryOp, LoopId, TensorId, index, access};
use crate::{Tensor, Shape, DType, Result};

/// Trait for types that can be used in DSL computations
pub trait DSLExpr {
    /// Get the shape of this expression
    fn shape(&self) -> &Shape;

    /// Get the data type of this expression
    fn dtype(&self) -> DType;

    /// Lower this expression to LCIR operations
    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()>;
}

/// Represents a computation that can be scheduled and executed
#[derive(Debug)]
pub struct Computation<T> {
    pub expr: T,
    pub schedule: Schedule,
}

impl<T: DSLExpr> Computation<T> {
    /// Create a new computation from an expression
    pub fn new(expr: T) -> Self {
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

    /// Execute the computation
    pub fn run(self) -> Result<Tensor> {
        // Build LCIR kernel
        let mut builder = KernelBuilder::new("computation");

        // Add result tensor
        let result_id = builder.add_tensor(
            "result",
            self.expr.shape().clone(),
            self.expr.dtype(),
            MemoryScope::Global
        );

        // Lower expression to LCIR
        self.expr.lower_to_lcir(&mut builder, result_id)?;

        // Apply scheduling transformations
        self.schedule.apply(&mut builder)?;

        // For now, just create a result tensor with zeros
        // TODO: Actually execute the kernel
        Ok(Tensor::zeros(self.expr.dtype(), self.expr.shape().clone()))
    }
}

/// Scheduling transformations for optimization
#[derive(Debug, Clone)]
pub struct Schedule {
    pub tiles: Vec<(usize, usize)>, // (axis, size)
    pub parallel_axes: Vec<usize>,
    pub vectorized_axes: Vec<usize>,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            tiles: Vec::new(),
            parallel_axes: Vec::new(),
            vectorized_axes: Vec::new(),
        }
    }
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

/// Binary operation expression
#[derive(Debug)]
pub struct BinaryExpr<L, R, Op> {
    pub lhs: L,
    pub rhs: R,
    pub op: Op,
}


/// Element-wise operation expression
#[derive(Debug)]
pub struct ElementWiseExpr<T, Op> {
    pub input: T,
    pub op: Op,
}

/// Unary operation marker types
#[derive(Debug)]
pub struct ExpOp;
#[derive(Debug)]
pub struct LogOp;
#[derive(Debug)]
pub struct SqrtOp;

/// Binary operation marker types
#[derive(Debug)]
pub struct AddOp;
#[derive(Debug)]
pub struct SubOp;
#[derive(Debug)]
pub struct MulOp;
#[derive(Debug)]
pub struct DivOp;

// DSL Trait Implementations

impl DSLExpr for Tensor {
    fn shape(&self) -> &Shape {
        self.shape()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        // For a tensor input, we just need to load it
        // Add input tensor to LCIR
        let input_id = builder.add_tensor(
            "input",
            self.shape().clone(),
            self.dtype(),
            MemoryScope::Global
        );

        // Add loops for element-wise access
        let mut loop_ids = Vec::new();
        for (i, &dim) in self.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(&format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        // Create indices for access
        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        // Add load operation
        let input_access = access::global(input_id, indices.clone());
        let result_access = access::global(result_tensor, indices);
        builder.add_load(result_access, input_access);

        Ok(())
    }
}

// DSL extension methods for Tensor
pub trait TensorDSL {
    fn dsl_add(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, AddOp>>;
    fn dsl_sub(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, SubOp>>;
    fn dsl_mul(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, MulOp>>;
}

impl TensorDSL for Tensor {
    fn dsl_add(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, AddOp>> {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: AddOp,
        };
        Computation::new(expr)
    }

    fn dsl_sub(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, SubOp>> {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: SubOp,
        };
        Computation::new(expr)
    }

    fn dsl_mul(self, rhs: Tensor) -> Computation<BinaryExpr<Tensor, Tensor, MulOp>> {
        let expr = BinaryExpr {
            lhs: self,
            rhs,
            op: MulOp,
        };
        Computation::new(expr)
    }
}

// DSL Expression Implementations

impl DSLExpr for BinaryExpr<Tensor, Tensor, AddOp> {
    fn shape(&self) -> &Shape {
        // For element-wise operations, shapes must match
        assert_eq!(self.lhs.shape(), self.rhs.shape(), "Shape mismatch in addition");
        self.lhs.shape()
    }

    fn dtype(&self) -> DType {
        // Assume same dtype for now
        assert_eq!(self.lhs.dtype(), self.rhs.dtype(), "Dtype mismatch in addition");
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        // Add input tensors
        let lhs_id = builder.add_tensor("lhs", self.lhs.shape().clone(), self.lhs.dtype(), MemoryScope::Global);
        let rhs_id = builder.add_tensor("rhs", self.rhs.shape().clone(), self.rhs.dtype(), MemoryScope::Global);

        // Add loops
        let mut loop_ids = Vec::new();
        for (i, &dim) in self.lhs.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(&format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        // Create indices
        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        // Add binary operation
        let lhs_access = access::global(lhs_id, indices.clone());
        let rhs_access = access::global(rhs_id, indices.clone());
        let result_access = access::global(result_tensor, indices);

        builder.add_binary_op(result_access, lhs_access, BinaryOp::Add, rhs_access);

        Ok(())
    }
}

// Similar implementations for Sub, Mul, Div...

impl DSLExpr for BinaryExpr<Tensor, Tensor, SubOp> {
    fn shape(&self) -> &Shape {
        assert_eq!(self.lhs.shape(), self.rhs.shape(), "Shape mismatch in subtraction");
        self.lhs.shape()
    }

    fn dtype(&self) -> DType {
        assert_eq!(self.lhs.dtype(), self.rhs.dtype(), "Dtype mismatch in subtraction");
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        let lhs_id = builder.add_tensor("lhs", self.lhs.shape().clone(), self.lhs.dtype(), MemoryScope::Global);
        let rhs_id = builder.add_tensor("rhs", self.rhs.shape().clone(), self.rhs.dtype(), MemoryScope::Global);

        let mut loop_ids = Vec::new();
        for (i, &dim) in self.lhs.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(&format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        let lhs_access = access::global(lhs_id, indices.clone());
        let rhs_access = access::global(rhs_id, indices.clone());
        let result_access = access::global(result_tensor, indices);

        builder.add_binary_op(result_access, lhs_access, BinaryOp::Sub, rhs_access);

        Ok(())
    }
}

impl DSLExpr for BinaryExpr<Tensor, Tensor, MulOp> {
    fn shape(&self) -> &Shape {
        assert_eq!(self.lhs.shape(), self.rhs.shape(), "Shape mismatch in multiplication");
        self.lhs.shape()
    }

    fn dtype(&self) -> DType {
        assert_eq!(self.lhs.dtype(), self.rhs.dtype(), "Dtype mismatch in multiplication");
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        let lhs_id = builder.add_tensor("lhs", self.lhs.shape().clone(), self.lhs.dtype(), MemoryScope::Global);
        let rhs_id = builder.add_tensor("rhs", self.rhs.shape().clone(), self.rhs.dtype(), MemoryScope::Global);

        let mut loop_ids = Vec::new();
        for (i, &dim) in self.lhs.shape().dims().iter().enumerate() {
            let loop_id = builder.add_loop(&format!("i{}", i), 0, dim as i64, 1);
            loop_ids.push(loop_id);
        }

        let indices: Vec<_> = loop_ids.iter().map(|&id| index::loop_var(id)).collect();

        let lhs_access = access::global(lhs_id, indices.clone());
        let rhs_access = access::global(rhs_id, indices.clone());
        let result_access = access::global(result_tensor, indices);

        builder.add_binary_op(result_access, lhs_access, BinaryOp::Mul, rhs_access);

        Ok(())
    }
}

/// MatMul expression that owns its result shape
#[derive(Debug)]
pub struct MatMulExprOwned {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub result_shape: Shape,
}

impl MatMulExprOwned {
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        let lhs_dims = lhs.shape().dims();
        let rhs_dims = rhs.shape().dims();

        // For now, assume 2D matrices
        assert_eq!(lhs_dims.len(), 2, "LHS must be 2D for matmul");
        assert_eq!(rhs_dims.len(), 2, "RHS must be 2D for matmul");
        assert_eq!(lhs_dims[1], rhs_dims[0], "Inner dimensions must match for matmul");

        // Create result shape
        let result_shape = Shape::from([lhs_dims[0], rhs_dims[1]]);

        Self {
            lhs,
            rhs,
            result_shape,
        }
    }
}

// Matrix multiplication DSL function
pub fn matmul(lhs: Tensor, rhs: Tensor) -> Computation<MatMulExprOwned> {
    let expr = MatMulExprOwned::new(lhs, rhs);
    Computation::new(expr)
}


impl DSLExpr for MatMulExprOwned {
    fn shape(&self) -> &Shape {
        &self.result_shape
    }

    fn dtype(&self) -> DType {
        assert_eq!(self.lhs.dtype(), self.rhs.dtype(), "Dtype mismatch in matmul");
        self.lhs.dtype()
    }

    fn lower_to_lcir(&self, builder: &mut KernelBuilder, result_tensor: TensorId) -> Result<()> {
        // Matrix multiplication LCIR generation
        let lhs_id = builder.add_tensor("lhs", self.lhs.shape().clone(), self.lhs.dtype(), MemoryScope::Global);
        let rhs_id = builder.add_tensor("rhs", self.rhs.shape().clone(), self.rhs.dtype(), MemoryScope::Global);

        let lhs_dims = self.lhs.shape().dims();
        let rhs_dims = self.rhs.shape().dims();

        // Add loops: i, j, k where result[i,j] += lhs[i,k] * rhs[k,j]
        let i_loop = builder.add_loop("i", 0, lhs_dims[0] as i64, 1);
        let j_loop = builder.add_loop("j", 0, rhs_dims[1] as i64, 1);
        let k_loop = builder.add_loop("k", 0, lhs_dims[1] as i64, 1);

        // For each i,j: result[i,j] = sum over k of lhs[i,k] * rhs[k,j]
        let lhs_access = access::global(lhs_id, vec![index::loop_var(i_loop), index::loop_var(k_loop)]);
        let rhs_access = access::global(rhs_id, vec![index::loop_var(k_loop), index::loop_var(j_loop)]);
        let result_access = access::global(result_tensor, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);

        builder.add_binary_op(result_access.clone(), lhs_access, BinaryOp::Mul, rhs_access);
        // TODO: Add reduction over k dimension

        Ok(())
    }
}

// Convenience methods for scheduling
pub trait Schedulable<T> {
    fn tile(self, axis: usize, size: usize) -> Computation<T>;
    fn parallelize(self, axis: usize) -> Computation<T>;
    fn vectorize(self, axis: usize) -> Computation<T>;
}

impl<T> Schedulable<T> for Computation<T> {
    fn tile(mut self, axis: usize, size: usize) -> Computation<T> {
        self.schedule.tiles.push((axis, size));
        self
    }

    fn parallelize(mut self, axis: usize) -> Computation<T> {
        self.schedule.parallel_axes.push(axis);
        self
    }

    fn vectorize(mut self, axis: usize) -> Computation<T> {
        self.schedule.vectorized_axes.push(axis);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32;

    #[test]
    fn dsl_basic_operations() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]));
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]));

        // Test DSL methods
        let computation = a.dsl_add(b);
        assert_eq!(computation.expr.shape(), &Shape::from([2]));
        assert_eq!(computation.expr.dtype(), F32);
    }

    #[test]
    fn dsl_scheduling() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], Shape::from([2]));
        let b = Tensor::from_slice(&[3.0f32, 4.0], Shape::from([2]));

        // Test scheduling
        let computation = a.dsl_add(b)
            .parallelize(0)
            .vectorize(0);

        assert_eq!(computation.schedule.parallel_axes, vec![0]);
        assert_eq!(computation.schedule.vectorized_axes, vec![0]);
    }

    #[test]
    fn dsl_matmul() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2]));
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2]));

        // Test matmul DSL
        let computation = matmul(a, b);
        assert_eq!(computation.expr.shape().dims(), &[2, 2]);
    }
}
