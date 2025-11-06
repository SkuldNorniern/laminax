//! Device abstraction layer for heterogeneous computing.
//!
//! Provides unified interfaces for different compute devices (CPU, GPU, etc.)
//! across all Laminax crates.

/// Types of compute devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Hip,
    Metal,
    Vulkan,
    Rocm,
    Tpu,
    Coral,
    Custom,
}

/// Device capabilities and properties
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub device_type: DeviceType,
    pub name: String,
    pub compute_units: usize,
    pub max_work_group_size: usize,
    pub local_memory_size: usize,
    pub global_memory_size: usize,
    pub supports_fp64: bool,
    pub supports_fp16: bool,
    pub supports_async: bool,
    pub unified_memory: bool,
    pub shared_memory: bool,
}

/// Abstract device interface
pub trait Device: Send + Sync {
    /// Get device type
    fn device_type(&self) -> DeviceType;

    /// Get device capabilities
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Check if device is available for use
    fn is_available(&self) -> bool;

    /// Get device name
    fn name(&self) -> &str {
        &self.capabilities().name
    }
}

