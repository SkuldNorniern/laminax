//! Execution of an LCIR-Graph (op DAG) on CPU with F32 buffers.
//!
//! API aligned with the Cetana–Laminax contract: run a graph with given input
//! data and shapes; return one buffer per node in node order. Supports Add, Sub,
//! Mul, Div, MatMul, Copy, Sum, Reshape. Level parallelism: nodes in the same parallel level
//! are independent; execution is currently sequential per level.

use super::Result;
use laminax_lcir::{Graph, Node, NodeId, Op, TensorRef};
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
        Op::Sum { axes, keep_dims } => run_sum(
            &input_buffers[0],
            &input_shapes[0],
            axes,
            *keep_dims,
        )?,
        Op::Reshape { shape } => {
            if node.inputs.len() != 1 {
                return Err(super::RuntimeError::Execution(
                    "Reshape expects 1 input".to_string(),
                ));
            }
            let n: usize = shape.iter().product();
            if n != input_buffers[0].len() {
                return Err(super::RuntimeError::Execution(format!(
                    "Reshape output shape {:?} has {} elements, input has {}",
                    shape,
                    n,
                    input_buffers[0].len()
                )));
            }
            input_buffers[0].clone()
        }
    };
    Ok(result)
}

fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![0; ndim];
    if ndim > 0 {
        strides[ndim - 1] = 1;
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
    }
    strides
}

fn run_sum(
    data: &[f32],
    in_shape: &[usize],
    axes: &[usize],
    keep_dims: bool,
) -> Result<Vec<f32>> {
    let ndim = in_shape.len();
    for &ax in axes {
        if ax >= ndim {
            return Err(super::RuntimeError::Execution(format!(
                "Sum axis {} out of range for shape {:?}",
                ax, in_shape
            )));
        }
    }
    let in_strides = strides_for_shape(in_shape);
    let out_shape: Vec<usize> = if keep_dims {
        in_shape
            .iter()
            .enumerate()
            .map(|(d, &s)| if axes.contains(&d) { 1 } else { s })
            .collect()
    } else {
        in_shape
            .iter()
            .enumerate()
            .filter(|(d, _)| !axes.contains(d))
            .map(|(_, &s)| s)
            .collect()
    };
    let out_size: usize = out_shape.iter().product();
    let out_strides = strides_for_shape(&out_shape);
    let non_reduced: Vec<usize> = (0..ndim).filter(|d| !axes.contains(d)).collect();

    let mut out = vec![0.0_f32; out_size];
    for out_flat in 0..out_size {
        let mut out_multi = vec![0; out_shape.len()];
        let mut rem = out_flat;
        for (d, &stride) in out_strides.iter().enumerate().rev() {
            out_multi[d] = rem / stride;
            rem %= stride;
        }
        let mut sum = 0.0_f32;
        let reduced_size: usize = axes.iter().map(|&ax| in_shape[ax]).product();
        for linear in 0..reduced_size {
            let mut reduced_indices = vec![0; axes.len()];
            let mut r = linear;
            for (idx, _) in axes.iter().enumerate() {
                let step: usize = axes[idx + 1..].iter().map(|&a| in_shape[a]).product();
                reduced_indices[idx] = r / step;
                r %= step;
            }
            let in_flat = out_multi_to_in_flat(
                &out_multi,
                &reduced_indices,
                in_shape,
                &in_strides,
                &non_reduced,
                axes,
                keep_dims,
            );
            sum += data[in_flat];
        }
        out[out_flat] = sum;
    }
    Ok(out)
}

fn out_multi_to_in_flat(
    out_multi: &[usize],
    reduced_indices: &[usize],
    in_shape: &[usize],
    in_strides: &[usize],
    non_reduced: &[usize],
    axes: &[usize],
    keep_dims: bool,
) -> usize {
    let ndim = in_shape.len();
    let mut in_multi = vec![0; ndim];
    if keep_dims {
        for d in 0..ndim {
            in_multi[d] = if axes.contains(&d) {
                reduced_indices[axes.iter().position(|&a| a == d).unwrap()]
            } else {
                out_multi[d]
            };
        }
    } else {
        for (out_d, &in_d) in non_reduced.iter().enumerate() {
            in_multi[in_d] = out_multi[out_d];
        }
        for (idx, &ax) in axes.iter().enumerate() {
            in_multi[ax] = reduced_indices[idx];
        }
    }
    in_multi
        .iter()
        .zip(in_strides.iter())
        .map(|(m, s)| m * s)
        .sum()
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

/// Like [`execute_graph`] but runs nodes in the same parallel level concurrently using threads.
/// Same inputs/outputs contract; only the execution order within each level is parallel.
pub fn execute_graph_parallel(
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
        let level_results: Result<Vec<(NodeId, Vec<f32>)>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(level.len());
            for &node_id in level.iter() {
                let node = graph
                    .node(node_id)
                    .ok_or_else(|| super::RuntimeError::Graph("node id out of range".to_string()))?
                    .clone();
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
                let handle = scope.spawn(move || run_one_node(&node, &input_buffers, &input_shapes_for_node).map(|r| (node_id, r)));
                handles.push(handle);
            }
            let mut out = Vec::with_capacity(handles.len());
            for h in handles {
                let joined = h.join().expect("graph execution thread panicked");
                out.push(joined?);
            }
            Ok(out)
        });
        for (node_id, result) in level_results? {
            buffers[input_count + node_id.0] = result;
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

    #[test]
    fn execute_graph_sum_reduce_last_axis() {
        let mut graph = Graph::new();
        let a = graph
            .add_input(vec![2, 3], DTypeId::F32);
        let _sum = graph
            .add_node(
                Op::Sum {
                    axes: vec![1],
                    keep_dims: false,
                },
                vec![a],
                TensorDesc {
                    shape: vec![2],
                    dtype_id: DTypeId::F32,
                },
            )
            .unwrap();
        let input_data: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let input_shapes: Vec<Vec<usize>> = vec![vec![2, 3]];
        let outputs = execute_graph(&graph, &input_data, &input_shapes).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], &[6.0, 15.0]);
    }

    #[test]
    fn execute_graph_parallel_same_result_as_sequential() {
        let mut graph = Graph::new();
        let a = graph.add_input(vec![2, 2], DTypeId::F32);
        let b = graph.add_input(vec![2, 2], DTypeId::F32);
        let c = graph
            .add_node(Op::Add, vec![a, b], TensorDesc { shape: vec![2, 2], dtype_id: DTypeId::F32 })
            .unwrap();
        let _d = graph
            .add_node(
                Op::Mul,
                vec![TensorRef::Node(c), b],
                TensorDesc { shape: vec![2, 2], dtype_id: DTypeId::F32 },
            )
            .unwrap();
        let input_data: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let input_shapes: Vec<Vec<usize>> = vec![vec![2, 2], vec![2, 2]];
        let seq = execute_graph(&graph, &input_data, &input_shapes).unwrap();
        let par = execute_graph_parallel(&graph, &input_data, &input_shapes).unwrap();
        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s, p);
        }
    }
}
