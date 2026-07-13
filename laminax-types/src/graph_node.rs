//! Tensor graph node types for use with [`laminax_dag::Dag`].
//!
//! Shared by LCIR-Graph and Cetana TensorGraph so the same op set and descriptor shape are used.

use laminax_dag::{NodeLike, Ref};
use numina::DTypeId;

/// Shape and dtype of a tensor in the graph (node outputs and validation).
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub dtype_id: DTypeId,
}

/// Operation kind for a node; parameters (e.g. axes for Sum) are part of the variant.
#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Sum { axes: Vec<usize>, keep_dims: bool },
    Reshape { shape: Vec<usize> },
    Copy,
}

/// A single node in a tensor op DAG: one op, its input refs, and the output descriptor.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<Ref>,
    pub output: TensorDesc,
}

impl NodeLike for Node {
    fn inputs(&self) -> &[Ref] {
        &self.inputs
    }
}
