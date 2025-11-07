//! Kernel execution and dispatch
//!
//! Handles the actual running of compiled kernels on devices.

use super::Result;
use super::graph::ExecutionPlan;
use super::memory::{Buffer, MemoryManager};
use laminax::{DType, Shape};
use laminax_types::Device;
use std::collections::HashMap;
use std::sync::Arc;

/// Compiled kernel instance ready for execution
pub struct KernelInstance {
    pub name: String,
    pub device: Arc<dyn Device>,
    // In real implementation, this would hold compiled code
}

/// Executor manages kernel execution on a specific device
pub struct Executor {
    device: Arc<dyn Device>,
    memory_manager: Arc<MemoryManager>,
    buffers: HashMap<usize, Buffer>, // buffer_id -> buffer
}

impl Executor {
    pub fn new(device: Arc<dyn Device>, memory_manager: Arc<MemoryManager>) -> Result<Self> {
        Ok(Self {
            device,
            memory_manager,
            buffers: HashMap::new(),
        })
    }

    /// Allocate a buffer for tensor data
    pub fn allocate_buffer(&mut self, shape: Shape, dtype: DType) -> Result<Buffer> {
        let buffer = self.memory_manager.allocate(shape, dtype, &self.device)?;
        self.buffers.insert(buffer.id, buffer.clone());
        Ok(buffer)
    }

    /// Allocate buffer and initialize with data
    pub fn allocate_buffer_with_data(
        &mut self,
        shape: Shape,
        dtype: DType,
        data: Vec<u8>,
    ) -> Result<Buffer> {
        let buffer = self.allocate_buffer(shape, dtype)?;
        // Copy data to buffer
        {
            let mut buf_data = buffer.data.lock().unwrap();
            buf_data.copy_from_slice(&data);
        }
        Ok(buffer)
    }

    /// Read data from buffer back to host
    pub fn read_buffer(&self, buffer: &Buffer) -> Result<Vec<u8>> {
        // For CPU, just return a copy of the buffer data
        let data = buffer.data.lock().unwrap();
        Ok(data.clone())
    }

    /// Execute an execution plan
    pub fn execute_plan(
        &mut self,
        plan: &ExecutionPlan,
        buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
    ) -> Result<()> {
        println!(
            "Executing plan with {} operations on {}",
            plan.nodes.len(),
            self.device.name()
        );

        // For now, execute the operations directly as an interpreter
        for &node_id in &plan.execution_order {
            let node = &plan.nodes[node_id];
            self.execute_operation(node, buffers)?;
        }

        Ok(())
    }

    /// Execute a single operation
    fn execute_operation(
        &mut self,
        node: &super::graph::Node,
        buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
    ) -> Result<()> {
        use laminax::lcir::Operation;

        match &node.operation {
            Operation::Binary { result, lhs, op, rhs } => {
                self.execute_binary_op(buffers, result, lhs, rhs, op)?;
            }
            Operation::Unary { result, op, input } => {
                self.execute_unary_op(buffers, result, op, input)?;
            }
            Operation::Load { result, source } => {
                self.execute_load_op(buffers, result, source)?;
            }
            Operation::Store { dest, value } => {
                self.execute_store_op(buffers, dest, value)?;
            }
            Operation::Barrier => {
                // No-op for CPU
            }
        }

        Ok(())
    }

    fn execute_binary_op(
        &mut self,
        buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
        result: &laminax::lcir::TensorAccess,
        lhs: &laminax::lcir::TensorAccess,
        rhs: &laminax::lcir::TensorAccess,
        op: &laminax::lcir::BinaryOp,
    ) -> Result<()> {
        let result_buf = buffers.get(&result.tensor_id).unwrap();
        let lhs_buf = buffers.get(&lhs.tensor_id).unwrap();
        let rhs_buf = buffers.get(&rhs.tensor_id).unwrap();

        // For simplicity, execute element-wise operation on all elements
        // In a real implementation, we'd handle the loop structure properly
        let len = result_buf.shape.len();
        for i in 0..len {
            match result_buf.dtype {
                laminax::DType::I32 => {
                    let lhs_val = self.read_i32(lhs_buf, i);
                    let rhs_val = self.read_i32(rhs_buf, i);
                    let result_val = match op {
                        laminax::lcir::BinaryOp::Add => lhs_val + rhs_val,
                        laminax::lcir::BinaryOp::Sub => lhs_val - rhs_val,
                        laminax::lcir::BinaryOp::Mul => lhs_val * rhs_val,
                        laminax::lcir::BinaryOp::Div => lhs_val / rhs_val,
                        _ => return Err(super::RuntimeError::Execution("Unsupported binary op".to_string())),
                    };
                    self.write_i32(result_buf, i, result_val);
                }
                _ => return Err(super::RuntimeError::Execution("Unsupported dtype".to_string())),
            }
        }

        Ok(())
    }

    fn execute_unary_op(
        &mut self,
        _buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
        _result: &laminax::lcir::TensorAccess,
        _op: &laminax::lcir::UnaryOp,
        _input: &laminax::lcir::TensorAccess,
    ) -> Result<()> {
        // Placeholder - implement unary operations
        Ok(())
    }

    fn execute_load_op(
        &mut self,
        buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
        result: &laminax::lcir::TensorAccess,
        source: &laminax::lcir::TensorAccess,
    ) -> Result<()> {
        // For now, just copy data (simplified)
        let result_buf = buffers.get(&result.tensor_id).unwrap();
        let source_buf = buffers.get(&source.tensor_id).unwrap();

        // Copy all data (simplified - should handle indexing properly)
        {
            let source_data = source_buf.data.lock().unwrap();
            let mut result_data = result_buf.data.lock().unwrap();
            result_data.copy_from_slice(&source_data);
        }
        Ok(())
    }

    fn execute_store_op(
        &mut self,
        buffers: &HashMap<laminax::lcir::TensorId, Buffer>,
        dest: &laminax::lcir::TensorAccess,
        value: &laminax::lcir::TensorAccess,
    ) -> Result<()> {
        // For now, just copy data (simplified)
        let dest_buf = buffers.get(&dest.tensor_id).unwrap();
        let value_buf = buffers.get(&value.tensor_id).unwrap();

        // Copy all data (simplified - should handle indexing properly)
        {
            let value_data = value_buf.data.lock().unwrap();
            let mut dest_data = dest_buf.data.lock().unwrap();
            dest_data.copy_from_slice(&value_data);
        }
        Ok(())
    }

    fn compute_flat_index(
        &self,
        _access: &laminax::lcir::TensorAccess,
        _loop_vars: &[i64],
    ) -> Result<usize> {
        // Simplified - return 0 for now
        // Real implementation would compute: indices[0] * stride[0] + indices[1] * stride[1] + ...
        Ok(0)
    }

    fn read_i32(&self, buffer: &Buffer, index: usize) -> i32 {
        let data = buffer.data.lock().unwrap();
        let offset = index * 4; // i32 is 4 bytes
        i32::from_le_bytes(data[offset..offset+4].try_into().unwrap())
    }

    fn write_i32(&self, buffer: &Buffer, index: usize, value: i32) {
        let mut data = buffer.data.lock().unwrap();
        let offset = index * 4; // i32 is 4 bytes
        data[offset..offset+4].copy_from_slice(&value.to_le_bytes());
    }

    /// Compile a kernel for this device (placeholder)
    pub fn compile_kernel(&self, _kernel_name: &str, _code: &[u8]) -> Result<KernelInstance> {
        Ok(KernelInstance {
            name: _kernel_name.to_string(),
            device: self.device.clone(),
        })
    }

    /// Launch a compiled kernel (placeholder)
    pub fn launch_kernel(&self, _kernel: &KernelInstance, _args: &[&Buffer]) -> Result<()> {
        // Placeholder for kernel launch
        Ok(())
    }
}
