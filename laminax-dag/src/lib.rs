//! Generic directed acyclic graph for Laminax.
//!
//! Used for op-order DAGs: nodes are operations, edges are data dependencies via [`Ref`].
//! Tensor graphs (LCIR-Graph, Cetana TensorGraph) and other consumers instantiate [`Dag`] with
//! their node type and get [`topological_order`](Dag::topological_order), [`parallel_levels`](Dag::parallel_levels), and cycle detection.

use std::{
    collections::VecDeque,
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

/// Identifies a node (index into the nodes vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Reference to a value in the graph: either a graph input or the output of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    /// The `i`-th graph input (zero-based).
    Input(usize),
    /// The single output of the given node.
    Node(NodeId),
}

/// Trait for node types that can be stored in a [`Dag`]. Only the dependency structure is required.
pub trait NodeLike {
    /// Predecessor refs (inputs). Used for topological order and parallel levels.
    fn inputs(&self) -> &[Ref];
}

/// Errors from graph construction or queries.
#[derive(Debug, Clone)]
pub enum DagError {
    /// Graph contains a cycle.
    Cycle,
    /// A node references a non-existent input or node.
    InvalidRef { node_id: usize, input_index: usize },
}

impl Display for DagError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            DagError::Cycle => write!(f, "graph contains a cycle"),
            DagError::InvalidRef {
                node_id,
                input_index,
            } => {
                write!(
                    f,
                    "node {} has invalid input ref at index {}",
                    node_id, input_index
                )
            }
        }
    }
}

impl Error for DagError {}

pub type DagResult<T> = Result<T, DagError>;

/// Generic DAG. Built by [`add_input`](Dag::add_input) and [`add_node`](Dag::add_node); refs may
/// only point to existing inputs or earlier nodes so the graph stays acyclic.
#[derive(Debug, Clone)]
pub struct Dag<N> {
    pub input_count: usize,
    pub nodes: Vec<N>,
}

impl<N> Default for Dag<N> {
    fn default() -> Self {
        Self {
            input_count: 0,
            nodes: Vec::new(),
        }
    }
}

impl<N: NodeLike> Dag<N> {
    /// Creates an empty DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a graph input; returns a ref to use as a node input.
    pub fn add_input(&mut self) -> Ref {
        let i = self.input_count;
        self.input_count += 1;
        Ref::Input(i)
    }

    /// Appends a node. Returns an error if any ref in `node.inputs()` is invalid.
    pub fn add_node(&mut self, node: N) -> DagResult<NodeId> {
        for (input_index, r) in node.inputs().iter().enumerate() {
            match r {
                Ref::Input(i) => {
                    if *i >= self.input_count {
                        return Err(DagError::InvalidRef {
                            node_id: self.nodes.len(),
                            input_index,
                        });
                    }
                }
                Ref::Node(n) => {
                    if n.0 >= self.nodes.len() {
                        return Err(DagError::InvalidRef {
                            node_id: self.nodes.len(),
                            input_index,
                        });
                    }
                }
            }
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        Ok(id)
    }

    /// Returns the node for the given id, or `None` if out of range.
    pub fn node(&self, id: NodeId) -> Option<&N> {
        self.nodes.get(id.0)
    }

    /// Node ids in topological order. With the current builder (refs only backward) this is `0..nodes.len()`.
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Execution waves: each inner `Vec<NodeId>` can run in parallel; waves are ordered by dependency.
    pub fn parallel_levels(&self) -> Vec<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = vec![vec![]; n];

        for (node_id, node) in self.nodes.iter().enumerate() {
            for r in node.inputs() {
                let pred = match r {
                    Ref::Input(_) => continue,
                    Ref::Node(p) => p.0,
                };
                adjacency[pred].push(node_id);
                in_degree[node_id] += 1;
            }
        }

        let mut queue = VecDeque::new();
        for (i, &d) in in_degree.iter().enumerate() {
            if d == 0 {
                queue.push_back(i);
            }
        }

        let mut levels = Vec::new();
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut level = Vec::with_capacity(level_size);
            for _ in 0..level_size {
                let u = queue.pop_front().expect("queue length fixed at loop start");
                level.push(NodeId(u));
                for &v in &adjacency[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push_back(v);
                    }
                }
            }
            levels.push(level);
        }
        levels
    }

    /// Returns true if the graph contains a cycle (invalid with refs-only-backward builder).
    pub fn has_cycle(&self) -> bool {
        for id in self.topological_order() {
            let node = &self.nodes[id.0];
            for r in node.inputs() {
                if let Ref::Node(p) = r {
                    if p.0 >= id.0 {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestNode {
        inputs: Vec<Ref>,
    }

    impl NodeLike for TestNode {
        fn inputs(&self) -> &[Ref] {
            &self.inputs
        }
    }

    #[test]
    fn dag_linear_chain() {
        let mut g: Dag<TestNode> = Dag::new();
        let a = g.add_input();
        let b = g.add_input();
        let c = g.add_node(TestNode { inputs: vec![a, b] }).unwrap();
        let _d = g
            .add_node(TestNode {
                inputs: vec![Ref::Node(c), b],
            })
            .unwrap();
        assert_eq!(g.input_count, 2);
        assert_eq!(g.nodes.len(), 2);
        assert!(!g.has_cycle());
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[1].len(), 1);
    }

    #[test]
    fn dag_invalid_ref_rejected() {
        let mut g: Dag<TestNode> = Dag::new();
        let _ = g.add_input();
        let err = g.add_node(TestNode {
            inputs: vec![Ref::Node(NodeId(5))],
        });
        assert!(err.is_err());
    }

    #[test]
    fn dag_parallel_fan_in() {
        let mut g: Dag<TestNode> = Dag::new();
        let a = g.add_input();
        let b = g.add_input();
        let c = g.add_input();
        let n0 = g.add_node(TestNode { inputs: vec![a, b] }).unwrap();
        let n1 = g.add_node(TestNode { inputs: vec![b, c] }).unwrap();
        let _n2 = g
            .add_node(TestNode {
                inputs: vec![Ref::Node(n0), Ref::Node(n1)],
            })
            .unwrap();
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2);
        assert_eq!(levels[1].len(), 1);
    }
}
