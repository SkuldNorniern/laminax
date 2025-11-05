//! Generic GPU array implementation.
//!
//! Provides a unified interface for GPU arrays across different backends
//! (CUDA, Metal, Vulkan, ROCm). Automatically selects the best available backend.

use std::sync::Arc;
use numina::{NdArray, Shape, Strides, DType};

use super::{Device, DeviceCapabilities, DeviceType};

/// Generic GPU device representation
#[derive(Debug, Clone)]
pub struct GpuDevice {
    capabilities: DeviceCapabilities,
    backend: GpuBackendType,
    device_id: i32,
}

/// Available GPU backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

impl GpuDevice {
    pub fn new(device_id: i32) -> Result<Self, String> {
        // Auto-detect the best available GPU backend
        let backend = Self::detect_backend()?;
        let capabilities = Self::get_capabilities_for_backend(backend, device_id)?;

        Ok(Self {
            capabilities,
            backend,
            device_id,
        })
    }

    pub fn with_backend(device_id: i32, backend: GpuBackendType) -> Result<Self, String> {
        let capabilities = Self::get_capabilities_for_backend(backend, device_id)?;
        Ok(Self {
            capabilities,
            backend,
            device_id,
        })
    }

    fn detect_backend() -> Result<GpuBackendType, String> {
        // Priority: CUDA > Metal > Vulkan > ROCm
        #[cfg(target_os = "linux")]
        {
            // Check for CUDA first
            if std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists() ||
               std::path::Path::new("/usr/local/cuda/lib64/libcudart.so").exists() {
                return Ok(GpuBackendType::Cuda);
            }
            // Check for ROCm
            if std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists() {
                return Ok(GpuBackendType::Rocm);
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Metal is always available on macOS
            return Ok(GpuBackendType::Metal);
        }

        #[cfg(target_os = "windows")]
        {
            // Check for CUDA on Windows
            if std::path::Path::new("C:\\Windows\\System32\\nvcuda.dll").exists() {
                return Ok(GpuBackendType::Cuda);
            }
        }

        // Fallback to Vulkan (cross-platform)
        Ok(GpuBackendType::Vulkan)
    }

    fn get_capabilities_for_backend(backend: GpuBackendType, device_id: i32) -> Result<DeviceCapabilities, String> {
        let base_capabilities = match backend {
            GpuBackendType::Cuda => DeviceCapabilities {
                device_type: DeviceType::Cuda,
                name: format!("GPU Device {} (CUDA)", device_id),
                compute_units: 80, // Default SM count
                max_work_group_size: 1024,
                local_memory_size: 48 * 1024, // 48KB shared memory
                global_memory_size: 8 * 1024 * 1024 * 1024, // 8GB default
                supports_fp64: true,
                supports_fp16: true,
                supports_async: true,
                unified_memory: false,
                shared_memory: true,
            },
            GpuBackendType::Metal => DeviceCapabilities {
                device_type: DeviceType::Metal,
                name: format!("GPU Device {} (Metal)", device_id),
                compute_units: 8, // Default GPU cores
                max_work_group_size: 1024,
                local_memory_size: 32 * 1024, // 32KB threadgroup memory
                global_memory_size: {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1) * 8 * 1024 * 1024 * 1024
                },
                supports_fp64: false,
                supports_fp16: true,
                supports_async: true,
                unified_memory: true, // Apple Silicon
                shared_memory: true,
            },
            GpuBackendType::Vulkan => DeviceCapabilities {
                device_type: DeviceType::Vulkan,
                name: format!("GPU Device {} (Vulkan)", device_id),
                compute_units: 60, // Default CU count
                max_work_group_size: 1024,
                local_memory_size: 64 * 1024, // 64KB workgroup memory
                global_memory_size: 8 * 1024 * 1024 * 1024, // 8GB default
                supports_fp64: true,
                supports_fp16: true,
                supports_async: true,
                unified_memory: false,
                shared_memory: true,
            },
            GpuBackendType::Rocm => DeviceCapabilities {
                device_type: DeviceType::Rocm,
                name: format!("GPU Device {} (ROCm)", device_id),
                compute_units: 60, // Default CU count
                max_work_group_size: 1024,
                local_memory_size: 64 * 1024, // 64KB LDS
                global_memory_size: 16 * 1024 * 1024 * 1024, // 16GB default
                supports_fp64: true,
                supports_fp16: true,
                supports_async: true,
                unified_memory: false,
                shared_memory: true,
            },
        };

        Ok(base_capabilities)
    }

    pub fn backend(&self) -> GpuBackendType {
        self.backend
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Device for GpuDevice {
    fn device_type(&self) -> DeviceType {
        match self.backend {
            GpuBackendType::Cuda => DeviceType::Cuda,
            GpuBackendType::Metal => DeviceType::Metal,
            GpuBackendType::Vulkan => DeviceType::Vulkan,
            GpuBackendType::Rocm => DeviceType::Rocm,
        }
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        match self.backend {
            GpuBackendType::Cuda => cfg!(target_os = "linux") || cfg!(target_os = "windows"),
            GpuBackendType::Metal => cfg!(target_os = "macos"),
            GpuBackendType::Vulkan => true, // Vulkan is cross-platform
            GpuBackendType::Rocm => cfg!(target_os = "linux"),
        }
    }
}

/// Generic GPU array that automatically selects the best backend
#[derive(Debug)]
pub struct GpuArray {
    shape: Shape,
    dtype: DType,
    device: Arc<GpuDevice>,
    // In real implementation:
    // This would contain backend-specific data structures
    // - For CUDA: CUdeviceptr
    // - For Metal: MTLBuffer
    // - For Vulkan: VkBuffer
    // - For ROCm: hipDeviceptr_t
}

impl GpuArray {
    /// Create a new GPU array using auto-detected backend
    pub fn new(shape: Shape, dtype: DType) -> Result<Self, String> {
        let device = Arc::new(GpuDevice::new(0)?);
        Self::new_with_device(shape, dtype, device)
    }

    /// Create a new GPU array with specific device
    pub fn new_with_device(shape: Shape, dtype: DType, device: Arc<GpuDevice>) -> Result<Self, String> {
        // In real implementation: allocate GPU memory based on backend
        Ok(Self {
            shape,
            dtype,
            device,
        })
    }

    /// Create GPU array from host data
    pub fn from_slice<T: Copy>(data: &[T], shape: Shape) -> Result<Self, String> {
        if data.len() != shape.len() {
            return Err("Data length doesn't match shape".to_string());
        }
        let array = Self::new(shape, DType::F32)?;
        // In real implementation: copy to GPU memory
        Ok(array)
    }

    /// Copy data from GPU to host
    pub fn to_host_vec(&self) -> Result<Vec<u8>, String> {
        let size = self.shape.len() * self.dtype.dtype_size_bytes();
        let mut host_data = vec![0u8; size];
        // In real implementation: copy from GPU memory
        Ok(host_data)
    }

    /// Get the GPU device
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackendType {
        self.device.backend()
    }
}

impl NdArray for GpuArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        unimplemented!("GPU strides not implemented - arrays assumed contiguous")
    }

    fn len(&self) -> usize {
        self.shape.len()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        panic!("Cannot access GPU memory directly - use to_host_vec")
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        panic!("Cannot access GPU memory directly - use from_slice")
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        // In real implementation: allocate new GPU memory and copy
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
        let array = Self::new_with_device(shape, self.dtype, self.device.clone())?;
        // In real implementation: GPU memset/kernel
        Ok(Box::new(array))
    }

    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let array = Self::new_with_device(shape, self.dtype, self.device.clone())?;
        // In real implementation: GPU fill kernel
        Ok(Box::new(array))
    }

    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        Self::new_with_device(shape, dtype, self.device.clone()).map(|arr| Box::new(arr) as Box<dyn NdArray>)
    }
}

impl std::fmt::Display for GpuArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuArray({}, {}, backend: {:?}, device: {})",
            self.shape,
            self.dtype,
            self.backend(),
            self.device.name()
        )
    }
}
