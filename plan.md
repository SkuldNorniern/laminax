# Laminax plan (detailed)

**Purpose:** Laminax is the target backend stack for Cetana. It will **replace** `cetana::backend` (CPU, CUDA, Vulkan, MPS) with a target-agnostic IR (LCIR), codegen, and runtime. This document is the single source of truth for the Laminax roadmap and its alignment with Cetana; it should be **updated whenever** Laminax or Cetana’s graph/backend integration changes.

**Role: BLAS / OpenCL replacement.** Laminax is a **generic compute layer**, not a tensor-only library. It must handle more than tensors: buffers, dtypes, and **modular chunks of operations** (elementwise, linalg, reduction, copy, etc.) that can be composed. **Tensor operations are one use case** that runs on top of this layer (tensor graphs lower to or dispatch into these operation chunks), rather than Laminax being one huge "tensor op only" blob. Design for modularity and reuse: small, coherent operation sets and clear boundaries so that tensor execution is a consumer of the same primitives that other clients (e.g. custom kernels, lib dispatch) can use.

**Repository layout:** `laminax/` contains the workspace: `laminax-lcir/`, `laminax-runtime/`, `laminax-types/`, `laminax-codegen/`, and top-level `src/` (DSL, lib). Cetana lives in the parent repo (`../src/`, `../plan.md`).

---

## 0. Glossary

| Term | Meaning |
|------|---------|
| **LCIR** | Lamina Compute IR — Laminax's target-agnostic representation (graph + kernel layers). |
| **LCIR-Graph** | DAG of ops and tensor dependencies (logical op order). Implemented in `laminax-lcir/src/graph.rs`. Nodes map to operation chunks; aligned with Cetana `TensorGraph`. |
| **LCIR-Kernel** | Loop/SSA-level IR for a single kernel: tensors, loops, ops (Binary, Unary, Load, Store, Barrier). In `laminax-lcir/src/lib.rs`. |
| **TensorDesc** | Tensor metadata: dtype (or DTypeId), shape, strides, layout. Used in graphs and at kernel boundaries. |
| **Grappler** | TensorFlow's graph optimizer. Laminax will provide analogous passes (fusion, memory, etc.). |
| **TorchInductor** | PyTorch's default torch.compile backend. Same role: consume graph, optimize, codegen. |

**Dependencies:** Laminax workspace depends on **numina** (dtypes, shapes, arrays; DTypeId/DTypeInfo for Phase 0) and **lamina** (CPU IR/codegen in laminax-codegen). Cetana is the upstream consumer; it may depend on Laminax for execution or Laminax may implement a trait/API that Cetana’s backend currently implements.

---

## 1. Relationship to Cetana

- **Cetana** provides:
  - **Eager tensors** (`cetana::tensor::Tensor<T>`): creation, ops, autodiff, serialization.
  - **Tensor op DAG** (`cetana::tensor::TensorGraph`): built from `add_input` / `add_node`; `parallel_levels()` gives execution waves for scheduling.
  - **Current backend** (`cetana::backend`): `Backend` trait (add, multiply, matmul, etc.), `Device`/`DeviceManager`, and **graph execution** (`execute_graph`, `execute_graph_parallel`) that run a `TensorGraph` on a `Backend`.
- **Laminax** will:
  - Replace `cetana::backend` as the execution layer: device discovery, memory, and running the op graph.
  - Consume a graph that is compatible with LCIR-Graph (same idea as Cetana’s DAG: nodes = ops, edges = tensor deps).
  - Provide scheduling (parallel levels, buffer reuse, fusion), lowering (LCIR-Graph → LCIR-Kernel → backend codegen or library calls), and runtime (allocation, copy insertion, dispatch).

**Alignment:** Cetana’s `TensorGraph` (ops, `TensorRef`, `TensorDesc`, `parallel_levels`) is the graph abstraction that Laminax runtime will schedule and execute. The plan is to map this into LCIR-Graph (or a thin adapter) so that Laminax takes over `execute_graph` / `execute_graph_parallel` without changing Cetana’s public graph API.

**Responsibility split (see repository root `plan.md`):** TensorGraph (Cetana) defines **operation order** only — the DAG and dependency waves. Laminax owns **job order, latency, and parallel management** — when/where to run nodes, buffer reuse, fusion, streams. The two graphs stay separate: Laminax consumes the op-order DAG and adds execution and scheduling. The same split appears in TensorFlow (tf.Graph vs Grappler/runtime) and PyTorch (captured graph vs torch.compile backends); see `plan.md` for the comparison.

---

## 2. Goals

- **Generic, modular compute layer** (BLAS/OpenCL replacement): expose **chunks of operations** (elementwise, linalg, reduction, copy, etc.) over buffers and dtypes, not a single monolithic "tensor op" surface. Tensor execution uses these chunks; other clients (library dispatch, custom graphs) use the same primitives.
- Define a **target-agnostic compute IR** (LCIR) that lowers to CUDA/HIP/ROCm, Metal, Vulkan/WGSL, OpenCL, and CPU (Lamina IR).
- Provide a **CUBLAS/OpenCL-style** path: unified IR that can either **codegen kernels** or **dispatch vendor libraries**.
- Make **dtype, layout, and memory semantics** explicit and stable (numina `DTypeId`, explicit strides/layout).
- Support **graph-level scheduling** and **kernel-level codegen** without baking backend-specific logic into the IR.

---

## 3. Principles

- **Modular ops over generic buffers:** Laminax is organized by **operation chunks** (e.g. elementwise, linalg, reduction, copy), not by "tensor-only" APIs. Each chunk is a coherent set of primitives; tensor graphs and other callers compose them.
- LCIR is the canonical compute IR for Laminax; backend codegen is a lowering step.
- DType identity uses numina `DTypeId` with stable serialization.
- Layout and memory are explicit: shape, strides, storage order, alignment, address spaces, scopes, barriers, atomics.

---

## 4. Architecture: generic layer + tensor on top

- **Core:** Buffers, dtypes (DTypeId), and **modular operation sets** (elementwise, linalg, reduction, copy, layout). These are the BLAS/OpenCL-style building blocks. IR and runtime are defined in terms of these chunks, not "tensor ops" as a single block.
- **Tensor use case:** Tensor graphs run by dispatching nodes into the core operation chunks. Tensor execution is a **consumer** of the generic layer.
- **Modularity:** New backends or library dispatch plug into the same operation chunks; adding a new op set (e.g. conv) extends a chunk rather than the whole stack.

---

## 5. IR layers

### 5.1 Graph IR (LCIR-Graph)

- **DAG** of ops and dependencies; can represent tensor graphs or other logical graphs that lower to the same core ops.
- **Nodes** map to **operation chunks**: elementwise, reduction, matmul, conv, copy, layout. Nodes reference buffers and descriptors.
- **Descriptors:** `TensorDesc { dtype_id, shape, strides, layout, address_space }` (or generic buffer descriptors where appropriate).
- **Scheduling:** topological order and parallel levels.

### 5.2 Kernel IR (LCIR-Kernel)

- Loop/SSA-level IR: scalar values, buffer/tensor accesses, explicit memory scope.
- Control flow: loops, if/else, predication.
- Ops: arithmetic, reductions, loads/stores, atomics, barriers (chunked by kind where useful).
- Launch config: grid/workgroup sizes, vector width, unroll hints.

### 5.3 Runtime plan

- Kernel selection, buffer allocation, copies, execution order. Tensor graph execution is one scheduling path; the runtime plan is generic over graphs that use the same core ops.

---

## 6. Library dispatch model

- LCIR-Graph nodes can lower to:
  - **Generated kernels** (LCIR-Kernel → backend codegen), or
  - **Library calls** (cuBLAS/rocBLAS/MKL/MPS/oneDNN/etc.) by runtime policy.
- Graph nodes may carry optional `library_hint` and `accum_dtype`.
- Library lowering must respect LCIR tensor layout/stride semantics (no implicit transpose).

---

## 7. Kernel ABI (target-independent)

- Kernel signature: `(buffers, scalars, launch_config, metadata)`.
- `BufferBinding`: `address_space`, `mutability`, `alias_group`, `byte_offset`.
- At kernel boundary, `TensorDesc`: `dtype_id`, `shape`, `strides`, `layout`, `storage_bits`, `align`.

---

## 8. Data model

- **DTypeId / DTypeInfo** from numina.
- **TensorDesc:** `dtype_id`, `shape`, `strides`, `layout`, `storage_bits`, `align`, `byte_offset`.
- **BufferBinding:** `address_space`, `scope`, `mutability`, `alias_group`.
- **IndexExpr:** affine and symbolic; non-affine is an error.

---

## 9. Backend targets

| Target | Lowering |
|--------|----------|
| CUDA/HIP/ROCm | LCIR-Kernel → PTX or LLVM IR |
| Metal | MSL (existing metal backend) |
| Vulkan/WebGPU | SPIR-V / WGSL |
| OpenCL | OpenCL C |
| CPU | Lamina IR or direct CPU kernel path |

### 9.1 Future backends (reference)

Existing work in the repo and in Laminax to align with when implementing each backend. Laminax will replace Cetana backends; reuse or port kernels, shaders, and codegen where possible.

| Backend | Repo / Cetana | Laminax-codegen | Notes |
|---------|----------------|-----------------|-------|
| **CUDA** | `cuda/` (repo root): `kernels.cu`, `kernels_wrapper.cu`, nvcc build. Cetana `src/backend/cuda/`: backend, compute, launch, stream. | `lowering/cuda.rs`, `compilation/cuda.rs`, `backends/cuda.rs`. LCIR → CUDA source (stub). | PTX/LLVM path for HIP/ROCm reuse. |
| **Vulkan** | Cetana `src/backend/vulkan/`: backend, buffer, compute, pipeline, descriptor, memory. `shaders/vulkan/`: `binary_ops.comp`, `matmul.comp`, `reduction.comp` (compute shaders). | `backends/vulkan.rs`. LCIR → SPIR-V/WGSL via `lowering/spirv.rs` (stubs). | SPIR-V or WGSL; wgpu can consume WGSL. |
| **Metal / MPS** | Cetana `src/backend/mps/`: backend, compute. `shaders/metal/`: `binary_ops.metal`, `matrix_ops.metal`, `operations.metal`, `reduction.metal`, `shaders.metallib`. | `lowering/metal.rs`, `compilation/metal.rs`, `backends/metal.rs`, `backends/apple.rs`. LCIR → MSL. | Apple CPU/GPU; MPS for acceleration. |
| **ROCm** | Cetana `src/backend/rocm/`: module stub. Laminax-types `laminax-types/src/array/rocm.rs`: `RocmArray`, `RocmDevice`; `array/gpu.rs` detects ROCm libs. | Same lowering as CUDA (HIP); backend target HIP/ROCm. `backends/` can add ROCm-specific build/runtime. | HIP = CUDA-like API; share LCIR→GPU codegen with CUDA where possible. |
| **Lamina (CPU)** | — | `lowering/lamina.rs`: LCIR → Lamina IR; `compilation/cpu.rs` compiles Lamina IR. Depends on `lamina` 0.0.6. | CPU codegen path; used by Laminax DSL today. |
| **wgpu / WebGPU** | — | `backends/webgpu.rs`. WGSL from `lowering/spirv.rs` (stub). | Cross-platform GPU via wgpu; WGSL as shader language. |
| **OpenCL** | — | `backends/opencl.rs`. `lowering/spirv.rs`: `lower_lcir_to_opencl` (stub). `compilation/shader.rs`: `compile_opencl`. | OpenCL C kernels; SPIR-V for OpenCL 2.1+ optional. |
| **Coral / Edge TPU** | — | Laminax-types `array/coral.rs`: `CoralDevice`, `CoralArray`; `from_compiled_model` stub. | Google Edge TPU; inference runtime; add codegen/lowering when targeting. |
| **TPU (cloud)** | — | Laminax-types `array/tpu.rs`: device shape (cores, HBM). | Cloud TPU; placeholder; XLA/tensorflow integration path if needed. |

When adding a backend: (1) add or extend lowering in `laminax-codegen/src/lowering/` and compilation in `compilation/`; (2) wire backend in `backends/` and runtime device/memory in `laminax-runtime` / `laminax-types`; (3) consider library dispatch (cuBLAS, rocBLAS, etc.) for linalg ops alongside or instead of codegen.

---

## 10. Current state (keep updated)

### 10.1 Cetana (upstream)

- **Tensor:** `cetana/src/tensor/` — eager `Tensor<T>`, backend per tensor, autodiff via `grad`/`grad_fn`.
- **TensorGraph:** `cetana/src/tensor/dag.rs` — `Graph` (alias `TensorGraph`), `Node`, `NodeId`, `Op`, `TensorRef`, `TensorDesc`; `add_input`, `add_node`, `topological_order`, `parallel_levels`, `has_cycle`. `CompiledGraph` + `compile_for_execution()`; `ExecutableGraph` trait used by backend.
- **Backend:** `cetana/src/backend/` — `Backend` trait (add, sub, multiply, div, matmul, exp, log, pow, sqrt, sum, mean); `Device`/`DeviceManager`; **graph execution** in `backend/graph.rs`: `execute_graph(backend, graph, input_data, input_shapes)`, `execute_graph_parallel(Arc<Backend>, graph, ...)` — takes `impl ExecutableGraph`, runs by levels, parallel within a level on CPU.
- **Scheduling plan:** `cetana/src/backend/scheduling_plan.md` — Phase 1 (level parallel) done; Phase 2 buffer reuse, Phase 3 fusion/cost, Phase 4 streams, Phase 5 graph capture.

### 10.2 Laminax repos (file locations)

| Crate | Path | Key types / entrypoints |
|-------|------|--------------------------|
| **laminax-lcir** | `laminax-lcir/src/lib.rs`, `laminax-lcir/src/graph.rs` | **LCIR-Kernel:** `Kernel`, `KernelBuilder`, `TensorInfo`, `Operation`, `Loop`, `IndexExpr`, `TensorId`, `LoopId`, `MemoryScope`. **LCIR-Graph (Phase 1):** `Graph`, `Node`, `NodeId`, `TensorRef`, `TensorDesc`, `Op` (by chunk: elementwise, linalg, reduction, copy, layout); `add_input`, `add_node`, `topological_order`, `parallel_levels`, `has_cycle`. Uses `laminax_types::DTypeId` in graph. |
| **laminax-runtime** | `laminax-runtime/src/` | `Runtime`, `Executor`, `ComputationGraph` (`graph.rs`: from one LCIR-Kernel via `from_lcir`), `ExecutionPlan`, `Executor::execute_plan` (`execution.rs`). **LCIR-Graph execution (Phase 5):** `op_graph.rs` — `execute_graph(graph, input_data, input_shapes)` runs LCIR-Graph on CPU (F32); uses `graph.parallel_levels()`, nodes within a level sequential (parallel-within-level optional later). `run_one_node` for Add, Sub, Mul, Div, MatMul, Copy. `device.rs`, `memory.rs`. |
| **laminax-types** | `laminax-types/src/` | Re-exports numina `DType`, `DTypeId`, `DTypeInfo`, `Shape`, `Strides`, `NdArray`, etc. (Phase 0 done.) `tensor.rs`, `array/`, `device/`. Uses numina via `path = "../../numina"`. |
| **laminax-codegen** | `laminax-codegen/src/` | `Backend` enum, `compile_from_lcir`, `Compiler`; `lowering/` (lamina, cuda, metal, spirv), `backends/`, `compilation/`. LCIR-Kernel to target source/binary. |
| **laminax (top)** | `src/lib.rs`, `src/dsl.rs` | DSL (`Computation`, `Schedule`), `lower_to_lcir`, `run_via_lcir`; re-exports lcir, types. Depends on lamina 0.0.6, numina 0.0.1. |

### 10.3 DType vs DTypeId

- **Plan (Phase 0):** LCIR and runtime use **numina `DTypeId`** for serialization and stable dispatch; `DTypeInfo` for layout.
- **Current:** Phase 0 done. laminax-types re-exports `DTypeId`, `DTypeInfo` from numina. LCIR-Graph (`laminax-lcir/graph.rs`) uses `DTypeId` in `TensorDesc`. Runtime `execute_graph` (`op_graph.rs`) uses `DTypeId` (F32 only for now). LCIR-Kernel `TensorInfo` and codegen may still use `DType` elsewhere; migration as needed.

### 9.3 Gap vs “replace cetana backend”

- **Graph IR:** LCIR-Graph implemented in laminax-lcir (DAG of ops/tensors); aligned with Cetana’s `TensorGraph` so the same logical graph can be passed to Laminax (adapter or shared shape).
- **Runtime:** Laminax-runtime accepts LCIR-Graph and runs execute_graph(graph, input_data, input_shapes) by parallel levels (sequential within level; parallel-within-level and buffer reuse later). Replacing Cetana execute_graph / execute_graph_parallel fully requires Cetana to call Laminax and/or an adapter from ExecutableGraph to LCIR-Graph.
- **Cetana integration:** Either Cetana depends on Laminax and calls Laminax runtime for graph execution, or Laminax provides a drop-in that implements the same execution contract. See "Cetana–Laminax API contract" below.

---

## 11. Phases (detailed)

### Phase 0 — DType + tensor metadata alignment

- Re-export numina `DType`, `DTypeId`, `DTypeInfo` in laminax-types.
- LCIR tensor metadata: `DTypeId` + explicit layout (shape, strides, etc.).
- LCIR serialization uses `DTypeId` only.
- Optional: `HostTensor` in laminax-types `(bytes, dtype_id, shape, strides)`.

### Phase 1 — Graph IR (LCIR-Graph) in laminax-lcir

- Define **Graph**, **Node**, **NodeId**, **TensorRef**, **TensorDesc**, and op set (elementwise, reduction, matmul, copy, layout) in laminax-lcir. See §15 (Needed before Phase 1) for concrete types and builder API.
- Shape and dtype validation in the graph builder.
- **Alignment:** Same concepts as Cetana `TensorGraph` (nodes = ops, tensor refs, output descriptor); ensure wire format or in-memory form can be produced from or consumed by Cetana’s graph.
- LCIR-Graph serialization and versioning.
- **Done when:** §14 Phase 1 acceptance criteria met.

### Phase 2 — Kernel IR expansion

- SSA scalar values and types.
- Address spaces, barriers, atomics in the IR.
- Kernel launch config and vectorization hints.

### Phase 2.5 — Library op interface

- `LibraryCall` node: op name, backend family, parameter schema.
- Legality checks vs `TensorDesc` (layout/stride compatibility).

### Phase 3 — Lowering passes

- Graph → Kernel lowering (tiling, fusion, reduction lowering).
- Layout legalization (strides, contiguous transforms).
- Canonicalization and dead code elimination.

### Phase 4 — Backend codegen integration

- Map LCIR-Kernel to existing backend emitters in laminax-codegen.
- Per-backend validation (vector width, shared memory, atomics).

### Phase 4.5 — Library integration

- Map LCIR-Graph nodes to backend library calls when available.
- Fallback to generated kernels when a library op is missing.

### Phase 5 — Runtime integration (replacing cetana backend)

- **Input:** LCIR-Graph (or adapter from Cetana `TensorGraph`).
- **Runtime:** Build execution plan from graph (topological + parallel levels, optional buffer reuse from lifetime analysis).
- **Execution:** Buffer allocation, copy insertion, dispatch (sequential or parallel per level; later streams/async).
- **API:** Provide an API that Cetana can call for “run this graph with these inputs” (replacing `execute_graph` / `execute_graph_parallel`).
- Cross-device scheduling policies (multi-device later).

### Phase 6 — Scheduling (align with cetana scheduling plan)

- **Level parallelism:** Run nodes in the same parallel level concurrently (already in Cetana; replicate in Laminax runtime). Maps to Cetana `scheduling_plan.md` Phase 1.
- **Memory planning:** Lifetime analysis, buffer reuse (see Cetana `scheduling_plan.md` Phase 2). Runtime already has `analyze_tensor_lifetimes` in `graph.rs`; use it for buffer reuse.
- **Op-aware:** Cost model, fusion hints (Cetana Phase 3).
- **Streams/async:** Backend streams for GPUs (Cetana Phase 4).

---

## 12. Tests

- DType ID stability and serialization round-trips.
- Shape/stride validation for graph ops.
- Kernel lowering correctness (golden IR).
- Cross-backend codegen smoke tests.
- **Integration:** Run a graph (LCIR-Graph or TensorGraph-shaped) through Laminax runtime and compare outputs to Cetana’s `execute_graph`.

---

## 13. Open questions

- Use Lamina IR as CPU lowering target or keep a dedicated LCIR CPU backend?
- How to represent quantized ops and mixed-precision in LCIR-Kernel?
- How to encode backend-specific capabilities without leaking into LCIR?
- Library call nodes in LCIR-Graph vs runtime-only decision?
- Exact Cetana ↔ Laminax boundary: adapter in Cetana, or Laminax implementing a trait/interface that Cetana’s backend module currently implements?

---

## 14. Cetana–Laminax API contract (draft)

To replace `cetana::backend::execute_graph` / `execute_graph_parallel`, Laminax must support the same contract:

- **Inputs:** A graph implementing `ExecutableGraph` (or an LCIR-Graph produced from it): `input_count()`, `nodes()`, `parallel_levels()`, `node(id)`. Plus `input_data: &[Vec<f32>]` and `input_shapes: &[Vec<usize>]` with length `input_count()`; each `input_data[i]` has length equal to the product of `input_shapes[i]`.
- **Output:** `Result<Vec<Vec<f32>>>` — one buffer per graph node (in node order), each buffer length = product of that node’s output shape. Currently F32 only.
- **Semantics:** Execute nodes in topological order; nodes in the same `parallel_levels()` wave may run concurrently. No reordering that changes semantics.

**Options:** (1) Laminax crate exposes `execute_graph(backend, graph, input_data, input_shapes)` that accepts something implementing a trait compatible with `ExecutableGraph` (adapter from LCIR-Graph or Cetana’s `CompiledGraph`). (2) Cetana adds a Laminax backend that forwards to Laminax runtime. (3) Cetana depends on Laminax and calls Laminax directly; `cetana::backend` becomes a thin wrapper or is removed.

---

## 15. Phase acceptance criteria

| Phase | Done when |
|-------|-----------|
| **0** | laminax-types re-exports `DTypeId`, `DTypeInfo`; LCIR can carry DTypeId in tensor metadata; serialization uses DTypeId. |
| **1** | laminax-lcir has `Graph`, `Node`, `NodeId`, `TensorRef`, `TensorDesc`, op set (Add, Sub, Mul, Div, MatMul, Copy, etc.) and `add_input`/`add_node`, `topological_order`, `parallel_levels`, `has_cycle`; shape/dtype validation; can build a graph equivalent to Cetana’s for the same ops. |
| **2** | Kernel IR has SSA scalars, address spaces, barriers, launch config, vectorization hints. |
| **3** | Graph → Kernel lowering (at least one path: e.g. one node → one kernel); layout legalization; optional canonicalization. |
| **4** | LCIR-Kernel lowers to all target backends (CPU, CUDA, Metal, Vulkan, etc.) via existing codegen. |
| **5** | Runtime accepts LCIR-Graph (or adapter); builds execution plan with parallel levels; allocates buffers, runs by level (sequential or parallel per level); API compatible with "run this graph with these inputs" (see §13). |
| **6** | Level parallelism in runtime; optional buffer reuse from lifetime analysis; optional streams/async. |

---

## 16. Needed before Phase 1

- **laminax-lcir:** New module or file for LCIR-Graph (separate from existing kernel IR in `lib.rs`): introduce `Graph`, `Node`, `NodeId`, `TensorRef` (Input(index) | Node(NodeId)), `TensorDesc` (shape, dtype_id or DType), `Op` enum (Add, Sub, Mul, Div, MatMul, Sum, Reshape, Copy, etc. — align with `cetana::tensor::Op`). Builder: `add_input(shape, dtype) -> TensorRef`, `add_node(op, inputs, output_desc) -> Result<NodeId>`. Queries: `topological_order()`, `parallel_levels()`, `has_cycle()`. Keep existing Kernel/KernelBuilder for LCIR-Kernel.
- **Alignment checklist with Cetana:** Same op names and semantics for elementwise and matmul; same notion of parallel levels (nodes with no dependency between them in the same wave); TensorDesc shape/dtype; input ordering (input 0, 1, … match `input_data[0]`, `input_data[1]`, …).
- **Optional:** Wire format or in-memory conversion from Cetana `CompiledGraph`/`Graph` to LCIR-Graph so Cetana can pass a graph without building it twice.

---

## 17. Document maintenance

- **When to update this plan:** Any change to Laminax phases, to Cetana’s tensor graph or backend execution API, or to the intended replacement of `cetana::backend` by Laminax.
- **Sections to keep in sync:** §9 (current state), §10 (phases), §13 (API contract), §14 (acceptance criteria), §15 (needed before Phase 1).
- **Cross-doc:** The graph responsibility split (op order vs job/latency/parallel) is stated in the repository root **`plan.md`**; keep the “Responsibility split” paragraph in §1 aligned with that.
