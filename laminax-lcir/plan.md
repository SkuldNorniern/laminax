# laminax-lcir plan

**Purpose:** This crate holds the Lamina Compute IR (LCIR) in two layers: **LCIR-Graph** (DAG of ops and buffer/tensor refs) and **LCIR-Kernel** (loop/SSA-level IR for a single kernel). It is the generic compute IR for Laminax; nodes in the graph map to **operation chunks** (elementwise, linalg, reduction, copy, layout), not a monolithic "tensor-only" surface. Tensor execution is one consumer of this IR; other clients (library dispatch, custom graphs) use the same primitives.

**Parent plan:** See `../plan.md` for Laminax roadmap, Phase 1 (Graph IR), and the "generic layer + tensor on top" architecture.

---

## 1. Role in Laminax

- **LCIR-Graph** is the graph layer: a DAG where each node is one op from a small, coherent set (a "chunk"). The runtime and codegen consume this graph; tensor graphs from Cetana (or elsewhere) lower to or align with it.
- **LCIR-Kernel** is the kernel layer: loop nests, tensor accesses, scalar ops (Binary, Unary, Load, Store, Barrier). Used for codegen and for single-kernel execution paths.
- **Alignment:** Graph concepts (Node, TensorRef, TensorDesc, parallel_levels) align with Cetana `TensorGraph` so the same logical graph can be passed to Laminax (adapter or shared shape).

---

## 2. Operation chunks (LCIR-Graph)

Graph nodes use the `Op` enum. Variants are grouped by chunk:

| Chunk        | Op variants              | Notes                          |
|-------------|--------------------------|--------------------------------|
| Elementwise | Add, Sub, Mul, Div       | Same shape in/out; broadcast TBD |
| Linalg      | MatMul                   | 2D matmul; batch TBD            |
| Reduction   | Sum { axes, keep_dims }  | Axes and keep_dims              |
| Copy        | Copy                     | Single input, same shape/type   |
| Layout      | Reshape { shape }       | Shape change only              |

Adding a new op usually means adding a variant to `Op` and implementing it in the runtime/codegen; adding a new **chunk** (e.g. conv) means a new coherent set of ops and possibly new lowering rules.

---

## 3. File layout

| File        | Contents |
|------------|----------|
| `src/lib.rs`   | LCIR-Kernel: `Kernel`, `KernelBuilder`, `TensorInfo`, `Operation`, `Loop`, `IndexExpr`, `TensorId`, `LoopId`, `MemoryScope`, `BinaryOp`, `UnaryOp`. Helpers: `index::`, `access::`. Re-exports from `graph`. |
| `src/graph.rs` | LCIR-Graph: `Graph`, `Node`, `NodeId`, `TensorRef`, `TensorDesc`, `Op`; `add_input`, `add_node`, `topological_order`, `parallel_levels`, `has_cycle`. Uses `laminax_types::DTypeId` for tensor metadata. |

---

## 4. Types and API (summary)

- **Graph:** `add_input(shape, dtype_id) -> TensorRef`, `add_node(op, inputs, output: TensorDesc) -> Result<NodeId>`. Refs are `TensorRef::Input(i)` or `TensorRef::Node(NodeId)`.
- **Queries:** `topological_order()`, `parallel_levels()` (execution waves), `has_cycle()`, `node(id)`.
- **TensorDesc:** `shape: Vec<usize>`, `dtype_id: DTypeId`. Strides/layout can be added later for layout-sensitive backends.

---

## 5. Tests

- `graph.rs`: `graph_linear_chain`, `graph_invalid_ref_rejected`, `graph_parallel_fan_in`.
- `lib.rs`: `basic_kernel_builder` (LCIR-Kernel).

Keep tests in-file per project convention.

---

## 6. Maintenance

- When adding a new `Op` variant: add to the chunk table above, implement in laminax-runtime `op_graph::run_one_node` (and codegen if applicable), and add a test if needed.
- Keep this plan in sync with `../plan.md` section 10.2 (Laminax repos) and Phase 1 acceptance criteria.
