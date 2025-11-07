//! Computational graph representation for execution planning
//!
//! Models the dependencies and execution order of tensor operations.

use super::Result;
use laminax_lcir::{self as lcir, Operation, TensorAccess, TensorId};
use std::collections::{HashMap, VecDeque};

/// Node in the computational graph
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub operation: Operation,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// Edge representing data dependency between operations
#[derive(Debug, Clone)]
pub struct Edge {
    pub from_node: usize,
    pub to_node: usize,
    pub tensor_id: TensorId,
}

/// Execution plan with optimized operation order
#[derive(Debug)]
pub struct ExecutionPlan {
    pub nodes: Vec<Node>,
    pub execution_order: Vec<usize>,
    pub tensor_lifetimes: HashMap<TensorId, (usize, usize)>, // (first_use, last_use)
}

/// Computational graph representing a kernel's operations and dependencies
#[derive(Debug)]
pub struct ComputationGraph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub tensor_info: HashMap<TensorId, lcir::TensorInfo>,
}

impl ComputationGraph {
    /// Build computational graph from LCIR kernel
    pub fn from_lcir(kernel: &lcir::Kernel) -> Result<Self> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut tensor_producers = HashMap::new(); // tensor -> producing node
        let mut tensor_consumers = HashMap::new(); // tensor -> consuming nodes

        // Process each operation
        for (op_index, operation) in kernel.operations.iter().enumerate() {
            let node_id = op_index;

            // Extract inputs and outputs from operation
            let (inputs, outputs) = match operation {
                Operation::Binary {
                    lhs, rhs, result, ..
                } => {
                    let inputs = extract_tensor_ids(&[lhs.clone(), rhs.clone()]);
                    let outputs = extract_tensor_ids(&[result.clone()]);
                    (inputs, outputs)
                }
                Operation::Unary { input, result, .. } => {
                    let inputs = extract_tensor_ids(&[input.clone()]);
                    let outputs = extract_tensor_ids(&[result.clone()]);
                    (inputs, outputs)
                }
                Operation::Load { result, source } => {
                    let inputs = extract_tensor_ids(&[source.clone()]);
                    let outputs = extract_tensor_ids(&[result.clone()]);
                    (inputs, outputs)
                }
                Operation::Store { dest, value } => {
                    let inputs = extract_tensor_ids(&[value.clone()]);
                    let outputs = extract_tensor_ids(&[dest.clone()]);
                    (inputs, outputs)
                }
                Operation::Barrier => (vec![], vec![]),
            };

            // Create node
            let node = Node {
                id: node_id,
                operation: operation.clone(),
                inputs: inputs.clone(),
                outputs: outputs.clone(),
            };
            nodes.push(node);

            // Track producers and consumers
            for &tensor_id in &outputs {
                tensor_producers.insert(tensor_id, node_id);
            }

            for &tensor_id in &inputs {
                tensor_consumers
                    .entry(tensor_id)
                    .or_insert_with(Vec::new)
                    .push(node_id);
            }
        }

        // Create edges based on data dependencies
        for (&tensor_id, &producer) in &tensor_producers {
            if let Some(consumers) = tensor_consumers.get(&tensor_id) {
                for &consumer in consumers {
                    edges.push(Edge {
                        from_node: producer,
                        to_node: consumer,
                        tensor_id,
                    });
                }
            }
        }

        Ok(Self {
            name: kernel.name.clone(),
            nodes,
            edges,
            tensor_info: kernel.tensors.clone(),
        })
    }

    /// Get topological execution order
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; self.nodes.len()];
        let mut adjacency = vec![vec![]; self.nodes.len()];

        // Build adjacency list and in-degrees
        for edge in &self.edges {
            adjacency[edge.from_node].push(edge.to_node);
            in_degree[edge.to_node] += 1;
        }

        // Kahn's algorithm
        let mut queue = VecDeque::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node);

            for &neighbor in &adjacency[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(super::RuntimeError::Graph(
                "Cycle detected in computational graph".to_string(),
            ));
        }

        Ok(result)
    }

    /// Analyze tensor lifetimes for memory optimization
    pub fn analyze_tensor_lifetimes(
        &self,
        execution_order: &[usize],
    ) -> HashMap<TensorId, (usize, usize)> {
        let mut lifetimes = HashMap::new();
        let mut last_use = HashMap::new();

        // Forward pass: find first use
        for (step, &node_id) in execution_order.iter().enumerate() {
            let node = &self.nodes[node_id];
            for &tensor_id in &node.inputs {
                lifetimes.entry(tensor_id).or_insert((step, step));
            }
            for &tensor_id in &node.outputs {
                lifetimes.entry(tensor_id).or_insert((step, step));
            }
        }

        // Backward pass: find last use
        for (step, &node_id) in execution_order.iter().enumerate().rev() {
            let node = &self.nodes[node_id];
            for &tensor_id in &node.inputs {
                if let Some((first, _)) = lifetimes.get_mut(&tensor_id) {
                    *last_use.entry(tensor_id).or_insert(step) =
                        step.max(*last_use.get(&tensor_id).unwrap_or(&0));
                }
            }
        }

        // Update lifetimes with last use information
        for (tensor_id, lifetime) in lifetimes.iter_mut() {
            if let Some(&last) = last_use.get(tensor_id) {
                lifetime.1 = last;
            }
        }

        lifetimes
    }
}

impl ExecutionPlan {
    /// Create execution plan from computational graph
    pub fn from_graph(graph: &ComputationGraph) -> Result<Self> {
        let execution_order = graph.topological_sort()?;
        let tensor_lifetimes = graph.analyze_tensor_lifetimes(&execution_order);

        Ok(Self {
            nodes: graph.nodes.clone(),
            execution_order,
            tensor_lifetimes,
        })
    }
}

/// Extract tensor IDs from tensor accesses
fn extract_tensor_ids(accesses: &[TensorAccess]) -> Vec<TensorId> {
    accesses.iter().map(|access| access.tensor_id).collect()
}
