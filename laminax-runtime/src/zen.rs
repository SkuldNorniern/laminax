#![cfg(any(feature = "hip", feature = "vulkan", feature = "cuda"))]

use std::sync::Arc;

use zengpu::{
    Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, GpuDevice, Instance, MemoryUsage,
    Scalar, ShaderDesc,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::RuntimeError;

pub type Result<T> = std::result::Result<T, RuntimeError>;

fn err(s: impl Into<String>) -> RuntimeError {
    RuntimeError::Execution(s.into())
}

fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn cast_u8(v: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
}

const SGEMM: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32 }
    @workgroup_size(16, 16)
    kernel sgemm(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let row = id.y
        let col = id.x
        if row < p.m && col < p.n {
            let sum: f32 = 0.0
            for i in 0..p.k {
                sum = sum + a[row * p.k + i] * b[i * p.n + col]
            }
            c[row * p.n + col] = sum
        }
    }
);

const ADD: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel add(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            c[i] = a[i] + b[i]
        }
    }
);

const SUB: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel sub(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            c[i] = a[i] + b[i] * -1.0
        }
    }
);

const MUL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel mul(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            c[i] = a[i] * b[i]
        }
    }
);

const DIV: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel div(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            c[i] = a[i] / b[i]
        }
    }
);

const EXP: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel exp_k(
        id: global_id,
        a: device buffer<f32>,
        b: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            b[i] = exp(a[i])
        }
    }
);

const LOG: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel log_k(
        id: global_id,
        a: device buffer<f32>,
        b: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            b[i] = log(a[i])
        }
    }
);

const SQRT: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel sqrt_k(
        id: global_id,
        a: device buffer<f32>,
        b: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            b[i] = sqrt(a[i])
        }
    }
);

const POW: ZslShader = zsl!(
    push P { n: u32, power: f32 }
    @workgroup_size(256)
    kernel pow_k(
        id: global_id,
        a: device buffer<f32>,
        b: device mut buffer<f32>,
        p: P,
    ) {
        let i = id.x
        if i < p.n {
            b[i] = pow(a[i], p.power)
        }
    }
);

fn pick<'a>(s: &'a ZslShader) -> (ShaderDesc<'a>, &'static str) {
    #[cfg(feature = "hip")]
    { return (ShaderDesc::hip(s.hip), "zsl_kernel"); }
    #[cfg(all(not(feature = "hip"), feature = "cuda"))]
    { return (ShaderDesc::cuda_src(s.cuda), "zsl_kernel"); }
    #[cfg(all(not(feature = "hip"), not(feature = "cuda")))]
    { (s.spirv_desc(), "main") }
}

pub struct ZenEngine {
    device: Arc<dyn GpuDevice>,
    #[allow(dead_code)]
    instance: Instance,
}

impl ZenEngine {
    pub fn new() -> Result<Self> {
        let instance = build_instance().map_err(|e| err(format!("ZenGPU init: {e}")))?;
        let adapters = instance.enumerate_adapters();
        if adapters.is_empty() {
            return Err(err("no ZenGPU adapters found"));
        }
        let device: Arc<dyn GpuDevice> =
            Arc::from(adapters[0].open(zengpu::DeviceRequest::default())
                .map_err(|e| err(format!("open device: {e}")))?);
        Ok(Self { device, instance })
    }

    pub fn device_name(&self) -> String {
        self.instance.enumerate_adapters()
            .first()
            .map(|a| a.info().name.clone())
            .unwrap_or_else(|| "unknown".into())
    }

    fn alloc(&self, n: usize) -> zengpu::Result<zengpu::BufferHandle> {
        self.device.create_buffer(BufferDesc {
            size: (n * 4) as u64,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        })
    }

    fn upload(&self, data: &[f32]) -> zengpu::Result<zengpu::BufferHandle> {
        let buf = self.alloc(data.len())?;
        self.device.write_buffer(buf, 0, cast_f32(data))?;
        Ok(buf)
    }

    fn download(&self, buf: zengpu::BufferHandle, n: usize) -> zengpu::Result<Vec<f32>> {
        let raw = self.device.read_buffer(buf, 0, (n * 4) as u64)?;
        Ok(cast_u8(&raw).to_vec())
    }

    fn run_binary(
        &self,
        a: &[f32],
        b: &[f32],
        shader: &ZslShader,
        extra_scalars: &[Scalar],
    ) -> Result<Vec<f32>> {
        let n = a.len();
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.upload(b).map_err(|e| err(e.to_string()))?;
        let bc = self.alloc(n).map_err(|e| err(e.to_string()))?;

        let (desc, entry) = pick(shader);
        let sh = self.device.create_shader(desc).map_err(|e| err(e.to_string()))?;
        let pipeline = self.device.create_compute_pipeline(ComputePipelineDesc {
            shader: sh, entry, block: [256, 1, 1],
        }).map_err(|e| err(e.to_string()))?;

        let mut scalars = vec![Scalar::U32(n as u32)];
        scalars.extend_from_slice(extra_scalars);

        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &scalars,
        };
        let grid = [(n as u32 + 255) / 256, 1, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        let out = self.download(bc, n).map_err(|e| err(e.to_string()))?;

        self.device.destroy_pipeline(pipeline);
        self.device.destroy_shader(sh);
        self.device.destroy_buffer(ba);
        self.device.destroy_buffer(bb);
        self.device.destroy_buffer(bc);
        Ok(out)
    }

    fn run_unary(
        &self,
        a: &[f32],
        shader: &ZslShader,
        extra_scalars: &[Scalar],
    ) -> Result<Vec<f32>> {
        let n = a.len();
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.alloc(n).map_err(|e| err(e.to_string()))?;

        let (desc, entry) = pick(shader);
        let sh = self.device.create_shader(desc).map_err(|e| err(e.to_string()))?;
        let pipeline = self.device.create_compute_pipeline(ComputePipelineDesc {
            shader: sh, entry, block: [256, 1, 1],
        }).map_err(|e| err(e.to_string()))?;

        let mut scalars = vec![Scalar::U32(n as u32)];
        scalars.extend_from_slice(extra_scalars);

        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index()],
            textures: &[],
            scalars:  &scalars,
        };
        let grid = [(n as u32 + 255) / 256, 1, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        let out = self.download(bb, n).map_err(|e| err(e.to_string()))?;

        self.device.destroy_pipeline(pipeline);
        self.device.destroy_shader(sh);
        self.device.destroy_buffer(ba);
        self.device.destroy_buffer(bb);
        Ok(out)
    }

    pub fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &ADD, &[])
    }

    pub fn sub(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &SUB, &[])
    }

    pub fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &MUL, &[])
    }

    pub fn div(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &DIV, &[])
    }

    pub fn exp(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &EXP, &[])
    }

    pub fn log(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &LOG, &[])
    }

    pub fn sqrt(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &SQRT, &[])
    }

    pub fn pow(&self, a: &[f32], power: f32) -> Result<Vec<f32>> {
        self.run_unary(a, &POW, &[Scalar::F32(power)])
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.upload(b).map_err(|e| err(e.to_string()))?;
        let bc = self.alloc(m * n).map_err(|e| err(e.to_string()))?;

        let (desc, entry) = pick(&SGEMM);
        let sh = self.device.create_shader(desc).map_err(|e| err(e.to_string()))?;
        let pipeline = self.device.create_compute_pipeline(ComputePipelineDesc {
            shader: sh, entry, block: [16, 16, 1],
        }).map_err(|e| err(e.to_string()))?;

        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        let out = self.download(bc, m * n).map_err(|e| err(e.to_string()))?;

        self.device.destroy_pipeline(pipeline);
        self.device.destroy_shader(sh);
        self.device.destroy_buffer(ba);
        self.device.destroy_buffer(bb);
        self.device.destroy_buffer(bc);
        Ok(out)
    }

    pub fn sum(&self, a: &[f32]) -> Result<f32> {
        Ok(a.iter().sum())
    }

    pub fn mean(&self, a: &[f32]) -> Result<f32> {
        if a.is_empty() { return Ok(0.0); }
        Ok(a.iter().sum::<f32>() / a.len() as f32)
    }
}

fn build_instance() -> zengpu::Result<Instance> {
    let b = Instance::builder();

    #[cfg(feature = "hip")]
    let b = match b.try_hip() {
        Ok(b) | Err(b) => b,
    };

    #[cfg(feature = "vulkan")]
    let b = match b.try_vulkan() {
        Ok(b) | Err(b) => b,
    };

    #[cfg(feature = "cuda")]
    let b = b.cuda();

    Ok(b.build())
}
