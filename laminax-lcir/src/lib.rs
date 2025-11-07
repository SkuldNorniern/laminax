//! Lamina Compute IR (LCIR) - Target-agnostic kernel representation
//!
//! LCIR provides explicit control over:
//! - Loop structures and indices
//! - Memory scopes (global, shared, local)
//! - Synchronization barriers
//! - Memory access patterns

use laminax_types::{DType, Shape};
use std::collections::HashMap;

/// Unique identifier for tensors in LCIR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

/// Unique identifier for loops in LCIR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopId(pub usize);

/// Memory scope for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Global memory (GPU: device memory, CPU: heap)
    Global,
    /// Shared memory (GPU: block-local scratchpad, CPU: L1/L2 cache)
    Shared,
    /// Local memory (GPU: thread registers, CPU: stack)
    Local,
}

/// Loop structure representing iteration
#[derive(Debug, Clone)]
pub struct Loop {
    pub id: LoopId,
    pub name: String,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub parallel: bool,
    pub vectorized: bool,
    pub unroll_factor: Option<usize>,
}

/// Tensor access with indexing
#[derive(Debug, Clone)]
pub struct TensorAccess {
    pub tensor_id: TensorId,
    pub indices: Vec<IndexExpr>,
    pub scope: MemoryScope,
}

/// Index expression for tensor access
#[derive(Debug, Clone)]
pub enum IndexExpr {
    /// Constant index value
    Const(i64),
    /// Loop variable reference
    LoopVar(LoopId),
    /// Arithmetic expression on indices
    Add(Box<IndexExpr>, Box<IndexExpr>),
    /// Subtraction expression
    Sub(Box<IndexExpr>, Box<IndexExpr>),
    /// Multiplication expression
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    /// Division expression
    Div(Box<IndexExpr>, Box<IndexExpr>),
}

/// Binary operations on tensors
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Unary operations on tensors
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
}

/// LCIR operation types
#[derive(Debug, Clone)]
pub enum Operation {
    /// Binary operation: result = lhs op rhs
    Binary {
        result: TensorAccess,
        lhs: TensorAccess,
        op: BinaryOp,
        rhs: TensorAccess,
    },
    /// Unary operation: result = op input
    Unary {
        result: TensorAccess,
        op: UnaryOp,
        input: TensorAccess,
    },
    /// Load from memory
    Load {
        result: TensorAccess,
        source: TensorAccess,
    },
    /// Store to memory
    Store {
        dest: TensorAccess,
        value: TensorAccess,
    },
    /// Synchronization barrier
    Barrier,
}

/// LCIR kernel representation
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub tensors: HashMap<TensorId, TensorInfo>,
    pub loops: Vec<Loop>,
    pub operations: Vec<Operation>,
    pub loop_nest: Vec<LoopId>, // Defines loop nesting order
}

/// Tensor metadata for LCIR
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Shape,
    pub dtype: DType,
    pub scope: MemoryScope,
    pub name: String,
}

/// Builder for constructing LCIR kernels
#[derive(Debug)]
pub struct KernelBuilder {
    next_tensor_id: usize,
    next_loop_id: usize,
    tensors: HashMap<TensorId, TensorInfo>,
    loops: Vec<Loop>,
    operations: Vec<Operation>,
    loop_nest: Vec<LoopId>,
}

impl KernelBuilder {
    /// Create a new kernel builder
    pub fn new(_name: impl Into<String>) -> Self {
        Self {
            next_tensor_id: 0,
            next_loop_id: 0,
            tensors: HashMap::new(),
            loops: Vec::new(),
            operations: Vec::new(),
            loop_nest: Vec::new(),
        }
    }

    /// Add a tensor to the kernel
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        scope: MemoryScope,
    ) -> TensorId {
        let id = TensorId(self.next_tensor_id);
        self.next_tensor_id += 1;

        let info = TensorInfo {
            shape,
            dtype,
            scope,
            name: name.into(),
        };

        self.tensors.insert(id, info);
        id
    }

    /// Add a loop to the kernel
    pub fn add_loop(&mut self, name: impl Into<String>, start: i64, end: i64, step: i64) -> LoopId {
        let id = LoopId(self.next_loop_id);
        self.next_loop_id += 1;

        let loop_info = Loop {
            id,
            name: name.into(),
            start,
            end,
            step,
            parallel: false,
            vectorized: false,
            unroll_factor: None,
        };

        self.loops.push(loop_info);
        self.loop_nest.push(id);
        id
    }

    /// Mark a loop as parallel
    pub fn set_parallel(&mut self, loop_id: LoopId, parallel: bool) {
        if let Some(loop_info) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_info.parallel = parallel;
        }
    }

    /// Mark a loop as vectorized
    pub fn set_vectorized(&mut self, loop_id: LoopId, vectorized: bool) {
        if let Some(loop_info) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_info.vectorized = vectorized;
        }
    }

    /// Set unroll factor for a loop
    pub fn set_unroll(&mut self, loop_id: LoopId, factor: usize) {
        if let Some(loop_info) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_info.unroll_factor = Some(factor);
        }
    }

    /// Add a binary operation
    pub fn add_binary_op(
        &mut self,
        result: TensorAccess,
        lhs: TensorAccess,
        op: BinaryOp,
        rhs: TensorAccess,
    ) {
        self.operations.push(Operation::Binary {
            result,
            lhs,
            op,
            rhs,
        });
    }

    /// Add a unary operation
    pub fn add_unary_op(&mut self, result: TensorAccess, op: UnaryOp, input: TensorAccess) {
        self.operations.push(Operation::Unary { result, op, input });
    }

    /// Add a load operation
    pub fn add_load(&mut self, result: TensorAccess, source: TensorAccess) {
        self.operations.push(Operation::Load { result, source });
    }

    /// Add a store operation
    pub fn add_store(&mut self, dest: TensorAccess, value: TensorAccess) {
        self.operations.push(Operation::Store { dest, value });
    }

    /// Add a synchronization barrier
    pub fn add_barrier(&mut self) {
        self.operations.push(Operation::Barrier);
    }

    /// Get mutable access to loops for scheduling
    pub fn loops_mut(&mut self) -> &mut Vec<Loop> {
        &mut self.loops
    }

    /// Build the kernel
    pub fn build(self) -> Kernel {
        Kernel {
            name: "kernel".to_string(), // TODO: Pass name to builder
            tensors: self.tensors,
            loops: self.loops,
            operations: self.operations,
            loop_nest: self.loop_nest,
        }
    }
}

/// Helper functions for creating index expressions
pub mod index {
    use super::*;

    pub fn constant(val: i64) -> IndexExpr {
        IndexExpr::Const(val)
    }

    pub fn loop_var(loop_id: LoopId) -> IndexExpr {
        IndexExpr::LoopVar(loop_id)
    }

    pub fn add(lhs: IndexExpr, rhs: IndexExpr) -> IndexExpr {
        IndexExpr::Add(Box::new(lhs), Box::new(rhs))
    }

    pub fn sub(lhs: IndexExpr, rhs: IndexExpr) -> IndexExpr {
        IndexExpr::Sub(Box::new(lhs), Box::new(rhs))
    }

    pub fn mul(lhs: IndexExpr, rhs: IndexExpr) -> IndexExpr {
        IndexExpr::Mul(Box::new(lhs), Box::new(rhs))
    }

    pub fn div(lhs: IndexExpr, rhs: IndexExpr) -> IndexExpr {
        IndexExpr::Div(Box::new(lhs), Box::new(rhs))
    }
}

/// Helper functions for creating tensor accesses
pub mod access {
    use super::*;

    pub fn tensor(
        tensor_id: TensorId,
        indices: Vec<IndexExpr>,
        scope: MemoryScope,
    ) -> TensorAccess {
        TensorAccess {
            tensor_id,
            indices,
            scope,
        }
    }

    pub fn global(tensor_id: TensorId, indices: Vec<IndexExpr>) -> TensorAccess {
        tensor(tensor_id, indices, MemoryScope::Global)
    }

    pub fn shared(tensor_id: TensorId, indices: Vec<IndexExpr>) -> TensorAccess {
        tensor(tensor_id, indices, MemoryScope::Shared)
    }

    pub fn local(tensor_id: TensorId, indices: Vec<IndexExpr>) -> TensorAccess {
        tensor(tensor_id, indices, MemoryScope::Local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_kernel_builder() {
        let mut builder = KernelBuilder::new("test_kernel");

        // Add tensors
        let a_id = builder.add_tensor("A", Shape::from([4, 4]), laminax_types::F32, MemoryScope::Global);
        let b_id = builder.add_tensor("B", Shape::from([4, 4]), laminax_types::F32, MemoryScope::Global);
        let c_id = builder.add_tensor("C", Shape::from([4, 4]), laminax_types::F32, MemoryScope::Global);

        // Add loops
        let i_loop = builder.add_loop("i", 0, 4, 1);
        let j_loop = builder.add_loop("j", 0, 4, 1);

        // Mark loops as parallel
        builder.set_parallel(i_loop, true);

        // Add operations
        let a_access = access::global(a_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let b_access = access::global(b_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);
        let c_access = access::global(c_id, vec![index::loop_var(i_loop), index::loop_var(j_loop)]);

        builder.add_binary_op(c_access.clone(), a_access, BinaryOp::Add, b_access);

        let kernel = builder.build();

        assert_eq!(kernel.tensors.len(), 3);
        assert_eq!(kernel.loops.len(), 2);
        assert_eq!(kernel.operations.len(), 1);
        assert!(kernel.loops[0].parallel); // i loop should be parallel
    }
}
