//! Laminax Runtime - Execution Engine and Computational Graph Management
//!
//! This crate provides the runtime execution capabilities for Laminax,
//! including computational graph representation, device abstraction, memory
//! management, and kernel execution.

use laminax::lcir::{self, MemoryScope};
use std::collections::HashMap;
use std::sync::Arc;

pub mod device;
pub mod execution;
pub mod graph;
pub mod memory;

pub use device::{Device, DeviceCapabilities, DeviceType};
pub use execution::{Executor, KernelInstance};
pub use graph::{ComputationGraph, Edge, ExecutionPlan, Node};
pub use memory::{Buffer, MemoryManager};

/// Runtime error types
#[derive(Debug)]
pub enum RuntimeError {
    Device(String),
    Memory(String),
    Graph(String),
    Execution(String),
    Compilation(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::Device(msg) => write!(f, "Device error: {}", msg),
            RuntimeError::Memory(msg) => write!(f, "Memory error: {}", msg),
            RuntimeError::Graph(msg) => write!(f, "Graph error: {}", msg),
            RuntimeError::Execution(msg) => write!(f, "Execution error: {}", msg),
            RuntimeError::Compilation(msg) => write!(f, "Compilation error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Main runtime context managing devices, memory, and execution
pub struct Runtime {
    devices: Vec<Arc<dyn Device>>,
    memory_manager: Arc<MemoryManager>,
}

impl Runtime {
    /// Create a new runtime with available devices
    pub fn new() -> Result<Self> {
        let devices = device::enumerate_devices()?;
        let memory_manager = Arc::new(MemoryManager::new(devices.clone())?);

        Ok(Self {
            devices,
            memory_manager,
        })
    }

    /// Get all available devices
    pub fn devices(&self) -> &[Arc<dyn Device>] {
        &self.devices
    }

    /// Get the default CPU device
    pub fn cpu_device(&self) -> Option<&Arc<dyn Device>> {
        self.devices
            .iter()
            .find(|d| d.device_type() == DeviceType::Cpu)
    }

    /// Create an executor for running computations
    pub fn executor(&self, device: Arc<dyn Device>) -> Result<Executor> {
        Executor::new(device, Arc::clone(&self.memory_manager))
    }

    /// Execute a kernel directly (convenience method)
    pub fn execute_kernel(
        &self,
        kernel: &lcir::Kernel,
        inputs: HashMap<String, Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let graph = ComputationGraph::from_lcir(kernel)?;
        let device = self
            .cpu_device()
            .ok_or_else(|| RuntimeError::Device("No CPU device available".to_string()))?
            .clone();

        let mut executor = self.executor(device)?;
        let plan = ExecutionPlan::from_graph(&graph)?;

        // Allocate buffers and transfer input data
        let mut buffers = HashMap::new();
        for (tensor_id, tensor_info) in &kernel.tensors {
            let buffer = if let Some(input_data) = inputs.get(&tensor_info.name) {
                executor.allocate_buffer_with_data(
                    tensor_info.shape.clone(),
                    tensor_info.dtype,
                    input_data.clone(),
                )?
            } else {
                executor.allocate_buffer(tensor_info.shape.clone(), tensor_info.dtype)?
            };
            buffers.insert(*tensor_id, buffer);
        }

        // Execute the plan
        executor.execute_plan(&plan, &buffers)?;

        // Extract output data
        let mut outputs = HashMap::new();
        for (tensor_id, tensor_info) in &kernel.tensors {
            if tensor_info.scope == MemoryScope::Global {
                // Assume outputs are tensors that aren't in inputs
                if !inputs.contains_key(&tensor_info.name) {
                    let buffer = buffers.get(tensor_id).unwrap();
                    let data = executor.read_buffer(buffer)?;
                    outputs.insert(tensor_info.name.clone(), data);
                }
            }
        }

        Ok(outputs)
    }
}

/// Convenience function to run a simple kernel (for examples/demos)
pub fn execute_simple_kernel(
    kernel: &lcir::Kernel,
    inputs: HashMap<String, Vec<u8>>,
) -> Result<HashMap<String, Vec<u8>>> {
    let runtime = Runtime::new()?;
    runtime.execute_kernel(kernel, inputs)
}
