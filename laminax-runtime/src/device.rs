//! Device abstraction layer for heterogeneous computing
//!
//! Provides unified interfaces for different compute devices (CPU, GPU, etc.)
//! Uses device types from laminax-types.

use super::Result;
use std::sync::Arc;
use laminax_types::{Device, DeviceType, DeviceCapabilities};

/// CPU device implementation
pub struct CpuDevice {
    capabilities: DeviceCapabilities,
}

impl CpuDevice {
    pub fn new() -> Self {
        let capabilities = DeviceCapabilities {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1), // Fallback to 1 if unavailable
            max_work_group_size: 1024,           // Arbitrary limit for CPU
            local_memory_size: 32 * 1024 * 1024, // 32MB L1/L2 cache estimate
            global_memory_size: get_system_memory(),
            supports_fp64: true,
            supports_fp16: cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"),
            supports_async: false, // CPU operations are typically synchronous
            unified_memory: true,  // CPU memory is unified
            shared_memory: false,  // No special shared memory on CPU
        };

        Self { capabilities }
    }
}

/// Get system memory using std::fs and /proc/meminfo (Linux) or other platform-specific methods
fn get_system_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Try reading /proc/meminfo on Linux
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Try using sysctl on macOS
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(mem_str) = std::str::from_utf8(&output.stdout) {
                if let Ok(mem) = mem_str.trim().parse::<usize>() {
                    return mem;
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Try using systeminfo on Windows
        use std::process::Command;
        if let Ok(output) = Command::new("wmic")
            .args(["ComputerSystem", "get", "TotalPhysicalMemory"])
            .output()
        {
            if let Ok(mem_str) = std::str::from_utf8(&output.stdout) {
                // Parse the output (skip header line)
                for line in mem_str.lines().skip(1) {
                    if let Ok(mem) = line.trim().parse::<usize>() {
                        return mem;
                    }
                }
            }
        }
    }

    // Fallback: estimate 8GB
    8 * 1024 * 1024 * 1024
}

impl Device for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }
}

/// Enumerate all available devices
pub fn enumerate_devices() -> Result<Vec<Arc<dyn Device>>> {
    let mut devices = Vec::new();

    // Add CPU device
    devices.push(Arc::new(CpuDevice::new()) as Arc<dyn Device>);

    // TODO: Add GPU devices when available
    // devices.extend(enumerate_gpu_devices()?);

    Ok(devices)
}
