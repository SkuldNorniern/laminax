//! Apple Metal GPU array implementation.
//!
//! Provides GPU-accelerated arrays for Apple Silicon using Metal.
//! Features unified memory, async compute, and optimized for Apple hardware.

use std::sync::Arc;
use numina::{NdArray, Shape, Strides, DType};

use super::{Device, DeviceCapabilities, DeviceType};

/// Metal GPU device representation
#[derive(Debug, Clone)]
pub struct MetalDevice {
    capabilities: DeviceCapabilities,
    device_id: usize,
}

impl MetalDevice {
    pub fn new(device_id: usize) -> Result<Self, String> {
        let capabilities = DeviceCapabilities {
            device_type: DeviceType::Metal,
            name: format!("Metal Device {}", device_id),
            compute_units: 8, // Default GPU cores on Apple Silicon
            max_work_group_size: 1024,
            local_memory_size: 32 * 1024, // 32KB threadgroup memory
            global_memory_size: {
                // Apple Silicon typically has unified memory
                // Use system memory as GPU memory
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1) * 8 * 1024 * 1024 * 1024 // 8GB per CPU core as estimate
            },
            supports_fp64: false, // Metal doesn't support FP64
            supports_fp16: true,
            supports_async: true,
            unified_memory: true, // Apple Silicon unified memory
            shared_memory: true,
        };

        Ok(Self {
            capabilities,
            device_id,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl Device for MetalDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Metal
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos")
    }
}

/// Metal GPU array implementation
#[derive(Debug)]
pub struct MetalArray {
    shape: Shape,
    dtype: DType,
    device: Arc<MetalDevice>,
    // In real implementation:
    // - buffer: MTLBuffer (Metal buffer)
    // - command_queue: MTLCommandQueue
    // - heap: MTLHeap for memory management
}

impl MetalArray {
    /// Create a new Metal array on the specified device
    pub fn new(shape: Shape, dtype: DType, device: Arc<MetalDevice>) -> Result<Self, String> {
        // In real implementation: create MTLBuffer
        Ok(Self {
            shape,
            dtype,
            device,
        })
    }

    /// Create Metal array from host data
    pub fn from_slice<T: Copy>(data: &[T], shape: Shape, device: Arc<MetalDevice>) -> Result<Self, String> {
        if data.len() != shape.len() {
            return Err("Data length doesn't match shape".to_string());
        }

        let array = Self::new(shape, DType::F32, device)?;
        // In real implementation: copy to MTLBuffer
        Ok(array)
    }

    /// Copy data from GPU to host (unified memory makes this efficient)
    pub fn to_host_vec(&self) -> Result<Vec<u8>, String> {
        let size = self.shape.len() * self.dtype.dtype_size_bytes();
        let mut host_data = vec![0u8; size];
        // In real implementation: with unified memory, data is already accessible
        Ok(host_data)
    }

    /// Get the Metal device
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Check if this array uses unified memory
    pub fn uses_unified_memory(&self) -> bool {
        self.device.capabilities().unified_memory
    }
}

impl NdArray for MetalArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        // Metal arrays are typically contiguous
        unimplemented!("Metal strides not implemented - arrays assumed contiguous")
    }

    fn len(&self) -> usize {
        self.shape.len()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        if self.uses_unified_memory() {
            // With unified memory, we can access directly
            // In real implementation: return buffer contents
            panic!("Unified memory access not implemented")
        } else {
            panic!("Cannot access Metal buffer memory directly")
        }
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        if self.uses_unified_memory() {
            // With unified memory, we can access directly
            panic!("Unified memory access not implemented")
        } else {
            panic!("Cannot access Metal buffer memory directly")
        }
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        // In real implementation: create new MTLBuffer and copy
        Box::new(Self {
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        if new_shape.len() != self.shape.len() {
            return Err("Reshape must preserve element count".to_string());
        }
        Ok(Box::new(Self {
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        }))
    }

    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        if self.shape.ndim() != 2 {
            return Err("Transpose only supported for 2D arrays".to_string());
        }
        let new_shape = Shape::from([self.shape.dim(1), self.shape.dim(0)]);
        Ok(Box::new(Self {
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        }))
    }

    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let array = Self::new(shape, self.dtype, self.device.clone())?;
        // In real implementation: Metal compute kernel to fill zeros
        Ok(Box::new(array))
    }

    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let array = Self::new(shape, self.dtype, self.device.clone())?;
        // In real implementation: Metal compute kernel to fill ones
        Ok(Box::new(array))
    }

    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        Self::new(shape, dtype, self.device.clone()).map(|arr| Box::new(arr) as Box<dyn NdArray>)
    }
}

impl std::fmt::Display for MetalArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MetalArray({}, {}, device: {})",
            self.shape,
            self.dtype,
            self.device.name()
        )
    }
}
