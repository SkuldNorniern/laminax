//! LCIR-Graph: DAG of ops and tensor dependencies (logical operation order).
//!
//! Built on the generic [`laminax_dag::Dag`]; nodes use shared [`Node`], [`Op`], [`TensorDesc`]
//! from laminax-types so the same graph shape can be consumed by Laminax runtime and Cetana.

use laminax_dag::{Dag, DagError, Ref};
use laminax_types::DTypeId;

pub use laminax_dag::NodeId;
pub use laminax_types::{Node, Op, TensorDesc};

/// Alias for use in graph APIs (input or node output).
pub type TensorRef = Ref;

/// Errors from graph construction or queries.
#[derive(Debug, Clone)]
pub enum GraphError {
    DagCycle,
    DagInvalidRef { node_id: usize, input_index: usize },
}

impl From<DagError> for GraphError {
    fn from(e: DagError) -> Self {
        match e {
            DagError::Cycle => GraphError::DagCycle,
            DagError::InvalidRef { node_id, input_index } => GraphError::DagInvalidRef {
                node_id,
                input_index,
            },
        }
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::DagCycle => write!(f, "graph contains a cycle"),
            GraphError::DagInvalidRef { node_id, input_index } => {
                write!(f, "node {} has invalid input ref at index {}", node_id, input_index)
            }
        }
    }
}

impl std::error::Error for GraphError {}

pub type GraphResult<T> = Result<T, GraphError>;

/// DAG of tensor ops. Built by adding inputs then nodes; refs only point to existing inputs or earlier nodes.
#[derive(Debug, Default, Clone)]
pub struct Graph(Dag<Node>);

impl Graph {
    /// Creates an empty graph (no inputs, no nodes).
    pub fn new() -> Self {
        Self(Dag::new())
    }

    /// Registers a graph input; returns a ref to use as a node input.
    pub fn add_input(&mut self, _shape: Vec<usize>, _dtype_id: DTypeId) -> TensorRef {
        self.0.add_input()
    }

    /// Appends a node. Returns an error if any ref is to a non-existent input or node.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<TensorRef>,
        output: TensorDesc,
    ) -> GraphResult<NodeId> {
        self.0.add_node(Node { op, inputs, output }).map_err(Into::into)
    }

    /// Returns the node for the given id, or `None` if out of range.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.0.node(id)
    }

    /// Node ids in topological order (with current builder this is `0..nodes.len()`).
    pub fn topological_order(&self) -> Vec<NodeId> {
        self.0.topological_order()
    }

    /// Execution waves: each inner `Vec<NodeId>` can run in parallel; waves are ordered by dependency.
    pub fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        self.0.parallel_levels()
    }

    /// Returns true if the graph contains a cycle (invalid with the current builder).
    pub fn has_cycle(&self) -> bool {
        self.0.has_cycle()
    }

    /// Number of graph inputs.
    pub fn input_count(&self) -> usize {
        self.0.input_count
    }

    /// All nodes in the graph.
    pub fn nodes(&self) -> &[Node] {
        &self.0.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_linear_chain() {
        let mut g = Graph::new();
        let a = g.add_input(vec![2, 2], DTypeId::F32);
        let b = g.add_input(vec![2, 2], DTypeId::F32);
        let c = g.add_node(
            Op::Add,
            vec![a, b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        ).unwrap();
        let _d = g.add_node(
            Op::Mul,
            vec![TensorRef::Node(c), b],
            TensorDesc {
                shape: vec![2, 2],
                dtype_id: DTypeId::F32,
            },
        ).unwrap();
        assert_eq!(g.input_count(), 2);
        assert_eq!(g.nodes().len(), 2);
        assert!(!g.has_cycle());
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[1].len(), 1);
    }

    #[test]
    fn graph_invalid_ref_rejected() {
        let mut g = Graph::new();
        let _ = g.add_input(vec![2], DTypeId::F32);
        let err = g.add_node(
            Op::Copy,
            vec![TensorRef::Node(NodeId(0))],
            TensorDesc {
                shape: vec![2],
                dtype_id: DTypeId::F32,
            },
        ).unwrap_err();
        match err {
            GraphError::DagInvalidRef { node_id: 0, input_index: 0 } => {}
            _ => panic!("expected DagInvalidRef"),
        }
    }

    #[test]
    fn graph_parallel_fan_in() {
        let mut g = Graph::new();
        let a = g.add_input(vec![4], DTypeId::F32);
        let b = g.add_input(vec![4], DTypeId::F32);
        let c = g.add_input(vec![4], DTypeId::F32);
        let n0 = g.add_node(Op::Add, vec![a, b], TensorDesc { shape: vec![4], dtype_id: DTypeId::F32 }).unwrap();
        let n1 = g.add_node(Op::Add, vec![b, c], TensorDesc { shape: vec![4], dtype_id: DTypeId::F32 }).unwrap();
        let _n2 = g.add_node(
            Op::Add,
            vec![TensorRef::Node(n0), TensorRef::Node(n1)],
            TensorDesc { shape: vec![4], dtype_id: DTypeId::F32 },
        ).unwrap();
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2);
        assert_eq!(levels[1].len(), 1);
    }
}
