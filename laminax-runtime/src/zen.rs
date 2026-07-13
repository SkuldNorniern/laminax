#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zengpu::{
    BackendPreference, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, GpuDevice,
    Instance, MemoryUsage, Scalar, ShaderDesc,
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

const BGEMM: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32 }
    @workgroup_size(16, 16)
    kernel bgemm(
        id: global_id,
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        let row = id.y
        let col = id.x
        let batch = id.z
        if row < p.m && col < p.n {
            let ao = batch * p.m * p.k
            let bo = batch * p.k * p.n
            let sum: f32 = 0.0
            for i in 0..p.k {
                sum = sum + a[ao + row * p.k + i] * b[bo + i * p.n + col]
            }
            c[batch * p.m * p.n + row * p.n + col] = sum
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

/// A compiled kernel kept alive for the engine's lifetime.
struct CachedPipeline {
    shader: zengpu::ShaderHandle,
    pipeline: zengpu::PipelineHandle,
}

/// Max recycled buffers kept per size bucket.
const POOL_BUCKET_CAP: usize = 32;

pub struct ZenEngine {
    device: Arc<dyn GpuDevice>,
    #[allow(dead_code)]
    instance: Instance,
    backend: BackendPreference,
    device_name: String,
    /// Kernel cache keyed by a caller-supplied kernel name. The backend entry-point
    /// name cannot be the key: ZSL emits every HIP kernel as `zsl_kernel`, so keying
    /// on it would silently hand one kernel's pipeline to another. Compiling a shader
    /// goes through hiprtc/naga per call otherwise — orders of magnitude slower than
    /// the kernel itself.
    pipelines: Mutex<HashMap<&'static str, CachedPipeline>>,
    /// Free-list of device buffers keyed by element count; avoids alloc/free per op.
    pool: Mutex<HashMap<usize, Vec<zengpu::BufferHandle>>>,
}

impl ZenEngine {
    pub fn new() -> Result<Self> {
        Self::with_adapter(0)
    }

    /// Open the `index`-th GPU adapter. Each engine owns one device; for multi-GPU,
    /// create one engine per adapter and drive them from separate threads.
    pub fn with_adapter(index: usize) -> Result<Self> {
        let (instance, backend) = build_instance()
            .map_err(|e| err(format!("ZenGPU init: {e}")))?;
        let adapters = instance.enumerate_adapters();
        if adapters.is_empty() {
            return Err(err("no ZenGPU adapters found"));
        }
        let adapter = adapters
            .get(index)
            .ok_or_else(|| err(format!("adapter {index} out of range ({} found)", adapters.len())))?;
        let device_name = adapter.info().name.clone();
        let device: Arc<dyn GpuDevice> =
            Arc::from(adapter.open(zengpu::DeviceRequest::default())
                .map_err(|e| err(format!("open device: {e}")))?);
        Ok(Self {
            device,
            instance,
            backend,
            device_name,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(HashMap::new()),
        })
    }

    /// Number of GPU adapters visible to the preferred backend.
    pub fn adapter_count() -> usize {
        build_instance()
            .map(|(instance, _)| instance.enumerate_adapters().len())
            .unwrap_or(0)
    }

    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }

    pub fn backend(&self) -> BackendPreference {
        self.backend
    }

    fn pick<'a>(&self, shader: &'a ZslShader) -> (ShaderDesc<'a>, &'static str) {
        shader.for_backend(self.backend)
    }

    /// Take a pooled `n`-element buffer or create one. Callers must return it
    /// with [`Self::recycle`] instead of destroying it.
    fn alloc(&self, n: usize) -> zengpu::Result<zengpu::BufferHandle> {
        if let Some(buf) = self.pool.lock().unwrap().get_mut(&n).and_then(Vec::pop) {
            return Ok(buf);
        }
        self.device.create_buffer(BufferDesc {
            size: (n * 4) as u64,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        })
    }

    fn recycle(&self, buf: zengpu::BufferHandle, n: usize) {
        let mut pool = self.pool.lock().unwrap();
        let bucket = pool.entry(n).or_default();
        if bucket.len() < POOL_BUCKET_CAP {
            bucket.push(buf);
        } else {
            drop(pool);
            self.device.destroy_buffer(buf);
        }
    }

    /// Compiled pipeline for `shader`, building and caching it under `name` on first use.
    fn pipeline_for(
        &self,
        shader: &ZslShader,
        name: &'static str,
        block: [u32; 3],
    ) -> Result<zengpu::PipelineHandle> {
        if let Some(cached) = self.pipelines.lock().unwrap().get(name) {
            return Ok(cached.pipeline);
        }
        let (desc, entry) = self.pick(shader);
        let sh = self.device.create_shader(desc).map_err(|e| err(e.to_string()))?;
        let pipeline = self
            .device
            .create_compute_pipeline(ComputePipelineDesc { shader: sh, entry, block })
            .map_err(|e| err(e.to_string()))?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(name, CachedPipeline { shader: sh, pipeline });
        Ok(pipeline)
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
        name: &'static str,
        extra_scalars: &[Scalar],
    ) -> Result<Vec<f32>> {
        let n = a.len();
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.upload(b).map_err(|e| err(e.to_string()))?;
        let bc = self.alloc(n).map_err(|e| err(e.to_string()))?;

        let pipeline = self.pipeline_for(shader, name, [256, 1, 1])?;

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

        self.recycle(ba, n);
        self.recycle(bb, n);
        self.recycle(bc, n);
        Ok(out)
    }

    fn run_unary(
        &self,
        a: &[f32],
        shader: &ZslShader,
        name: &'static str,
        extra_scalars: &[Scalar],
    ) -> Result<Vec<f32>> {
        let n = a.len();
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.alloc(n).map_err(|e| err(e.to_string()))?;

        let pipeline = self.pipeline_for(shader, name, [256, 1, 1])?;

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

        self.recycle(ba, n);
        self.recycle(bb, n);
        Ok(out)
    }

    pub fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &ADD, "add", &[])
    }

    pub fn sub(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &SUB, "sub", &[])
    }

    pub fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &MUL, "mul", &[])
    }

    pub fn div(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.run_binary(a, b, &DIV, "div", &[])
    }

    pub fn exp(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &EXP, "exp", &[])
    }

    pub fn log(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &LOG, "log", &[])
    }

    pub fn sqrt(&self, a: &[f32]) -> Result<Vec<f32>> {
        self.run_unary(a, &SQRT, "sqrt", &[])
    }

    pub fn pow(&self, a: &[f32], power: f32) -> Result<Vec<f32>> {
        self.run_unary(a, &POW, "pow", &[Scalar::F32(power)])
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.upload(b).map_err(|e| err(e.to_string()))?;
        let bc = self.alloc(m * n).map_err(|e| err(e.to_string()))?;

        let pipeline = self.pipeline_for(&SGEMM, "sgemm", [16, 16, 1])?;

        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        let out = self.download(bc, m * n).map_err(|e| err(e.to_string()))?;

        self.recycle(ba, a.len());
        self.recycle(bb, b.len());
        self.recycle(bc, m * n);
        Ok(out)
    }

    /// Batched GEMM: `batch` back-to-back `[m,k] @ [k,n]` products in one dispatch
    /// (grid z = batch index). `a` is `batch*m*k` elements, `b` is `batch*k*n`.
    pub fn matmul_batched(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let ba = self.upload(a).map_err(|e| err(e.to_string()))?;
        let bb = self.upload(b).map_err(|e| err(e.to_string()))?;
        let bc = self.alloc(batch * m * n).map_err(|e| err(e.to_string()))?;

        let pipeline = self.pipeline_for(&BGEMM, "bgemm", [16, 16, 1])?;

        let bindings = Bindings {
            buffers:  &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        let out = self.download(bc, batch * m * n).map_err(|e| err(e.to_string()))?;

        self.recycle(ba, a.len());
        self.recycle(bb, b.len());
        self.recycle(bc, batch * m * n);
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

impl Drop for ZenEngine {
    fn drop(&mut self) {
        for bucket in self.pool.lock().unwrap().drain() {
            for buf in bucket.1 {
                self.device.destroy_buffer(buf);
            }
        }
        for (_, cached) in self.pipelines.lock().unwrap().drain() {
            self.device.destroy_pipeline(cached.pipeline);
            self.device.destroy_shader(cached.shader);
        }
    }
}

fn build_instance() -> zengpu::Result<(Instance, BackendPreference)> {
    let b = Instance::builder();

    if let Ok(b) = b.try_hip() {
        return Ok((b.build(), BackendPreference::Hip));
    }

    let b = Instance::builder();
    if let Ok(b) = b.try_vulkan() {
        return Ok((b.build(), BackendPreference::Vulkan));
    }

    let b = Instance::builder();
    Ok((b.build(), BackendPreference::Auto))
}
