//! Execution of an LCIR-Graph (op DAG) on CPU with F32 buffers.
//!
//! API aligned with the Cetana–Laminax contract: run a graph with given input
//! data and shapes; return one buffer per node in node order. Supports Add, Sub,
//! Mul, Div, MatMul, Copy. Level parallelism: nodes in the same parallel level
//! are independent; execution is currently sequential per level.

use super::Result;
use laminax_lcir::{Graph, Node, Op, TensorRef};
use laminax_types::DTypeId;

fn buffer_index(input_count: usize, r: TensorRef) -> usize {
    match r {
        TensorRef::Input(i) => i,
        TensorRef::Node(n) => input_count + n.0,
    }
}

fn shape_num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn run_one_node(
    node: &Node,
    input_buffers: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> Result<Vec<f32>> {
    let result = match &node.op {
        Op::Add => {
            if node.inputs.len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "Add expects 2 inputs".to_string(),
                ));
            }
            let a = &input_buffers[0];
            let b = &input_buffers[1];
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        }
        Op::Sub => {
            if node.inputs.len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "Sub expects 2 inputs".to_string(),
                ));
            }
            let a = &input_buffers[0];
            let b = &input_buffers[1];
            a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
        }
        Op::Mul => {
            if node.inputs.len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "Mul expects 2 inputs".to_string(),
                ));
            }
            let a = &input_buffers[0];
            let b = &input_buffers[1];
            a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
        }
        Op::Div => {
            if node.inputs.len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "Div expects 2 inputs".to_string(),
                ));
            }
            let a = &input_buffers[0];
            let b = &input_buffers[1];
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if *y == 0.0 { 0.0 } else { x / y })
                .collect()
        }
        Op::MatMul => {
            if node.inputs.len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "MatMul expects 2 inputs".to_string(),
                ));
            }
            if input_shapes[0].len() != 2 || input_shapes[1].len() != 2 {
                return Err(super::RuntimeError::Execution(
                    "MatMul expects 2D inputs".to_string(),
                ));
            }
            let m = input_shapes[0][0];
            let n = input_shapes[0][1];
            let k = input_shapes[1][1];
            if input_shapes[1][0] != n {
                return Err(super::RuntimeError::Execution(format!(
                    "MatMul shape mismatch: lhs {:?} rhs {:?}",
                    input_shapes[0], input_shapes[1]
                )));
            }
            let a = &input_buffers[0];
            let b = &input_buffers[1];
            let mut out = vec![0.0_f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    let mut sum = 0.0_f32;
                    for q in 0..n {
                        sum += a[i * n + q] * b[q * k + j];
                    }
                    out[i * k + j] = sum;
                }
            }
            out
        }
        Op::Copy => {
            if node.inputs.len() != 1 {
                return Err(super::RuntimeError::Execution(
                    "Copy expects 1 input".to_string(),
                ));
            }
            input_buffers[0].clone()
        }
        Op::Sum { .. } | Op::Reshape { .. } => {
            return Err(super::RuntimeError::Execution(format!(
                "op {:?} not yet supported in graph execution",
                node.op
            )));
        }
    };
    Ok(result)
}

/// Executes an LCIR-Graph on CPU with the given inputs (F32 only).
///
/// `input_data` and `input_shapes` must have length `graph.input_count`; each
/// `input_data[i]` must have length equal to the product of `input_shapes[i]`.
/// Returns one buffer per graph node (in node order). Nodes in the same
/// parallel level are independent; execution runs levels in order, nodes within
/// a level sequentially (parallel per level can be added later).
pub fn execute_graph(
    graph: &Graph,
    input_data: &[Vec<f32>],
    input_shapes: &[Vec<usize>],
) -> Result<Vec<Vec<f32>>> {
    if graph.input_count != input_data.len() || graph.input_count != input_shapes.len() {
        return Err(super::RuntimeError::Graph(format!(
            "graph has {} inputs but got {} data and {} shape buffers",
            graph.input_count,
            input_data.len(),
            input_shapes.len()
        )));
    }

    for (i, (data, shape)) in input_data.iter().zip(input_shapes.iter()).enumerate() {
        let expected = shape_num_elements(shape);
        if data.len() != expected {
            return Err(super::RuntimeError::Graph(format!(
                "input {} length {} does not match shape {:?} ({} elements)",
                i,
                data.len(),
                shape,
                expected
            )));
        }
    }

    let input_count = graph.input_count;
    let mut buffers: Vec<Vec<f32>> = input_data.to_vec();
    let mut shapes: Vec<Vec<usize>> = input_shapes.to_vec();

    for node in &graph.nodes {
        if node.output.dtype_id != DTypeId::F32 {
            return Err(super::RuntimeError::Execution(
                "only F32 dtype is supported for graph execution".to_string(),
            ));
        }
        let out_len = shape_num_elements(&node.output.shape);
        shapes.push(node.output.shape.clone());
        buffers.push(vec![0.0; out_len]);
    }

    for level in graph.parallel_levels() {
        for node_id in level {
            let node = graph
                .node(node_id)
                .ok_or_else(|| super::RuntimeError::Graph("node id out of range".to_string()))?;
            let input_buffers: Vec<Vec<f32>> = node
                .inputs
                .iter()
                .map(|r| buffers[buffer_index(input_count, *r)].clone())
                .collect();
            let input_shapes_for_node: Vec<Vec<usize>> = node
                .inputs
                .iter()
                .map(|r| shapes[buffer_index(input_count, *r)].clone())
                .collect();
            let result = run_one_node(node, &input_buffers, &input_shapes_for_node)?;
            let out_idx = input_count + node_id.0;
            buffers[out_idx] = result;
        }
    }

    let node_outputs = buffers.split_off(input_count);
    Ok(node_outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use laminax_lcir::{Graph, TensorDesc};
    use laminax_types::DTypeId;

    #[test]
    fn execute_graph_add_mul() {
        let mut graph = Graph::new();
        let a = graph.add_input(vec![2, 2], DTypeId::F32);
        let b = graph.add_input(vec![2, 2], DTypeId::F32);
        let c = graph
            .add_node(
                Op::Add,
                vec![a, b],
                TensorDesc {
                    shape: vec![2, 2],
                    dtype_id: DTypeId::F32,
                },
            )
            .unwrap();
        let _d = graph
            .add_node(
                Op::Mul,
                vec![TensorRef::Node(c), b],
                TensorDesc {
                    shape: vec![2, 2],
                    dtype_id: DTypeId::F32,
                },
            )
            .unwrap();

        let input_data: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
        ];
        let input_shapes: Vec<Vec<usize>> = vec![vec![2, 2], vec![2, 2]];

        let outputs = execute_graph(&graph, &input_data, &input_shapes).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(outputs[1], &[110.0, 440.0, 990.0, 1760.0]);
    }
}
