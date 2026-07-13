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

const TILED_SGEMM_HIP: &str = r#"
#define TILE 16
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ A, const float* __restrict__ B,
                float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = blockIdx.y * TILE + ty;
    unsigned int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    unsigned int tiles = (K + TILE - 1) / TILE;
    for (unsigned int t = 0; t < tiles; ++t) {
        unsigned int aCol = t * TILE + tx;
        unsigned int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < TILE; ++i) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

const TILED_BGEMM_HIP: &str = r#"
#define TILE 16
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ A, const float* __restrict__ B,
                float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int row = blockIdx.y * TILE + ty;
    unsigned int col = blockIdx.x * TILE + tx;
    unsigned int batch = blockIdx.z;
    const float* Ab = A + (size_t)batch * M * K;
    const float* Bb = B + (size_t)batch * K * N;
    float acc = 0.0f;
    unsigned int tiles = (K + TILE - 1) / TILE;
    for (unsigned int t = 0; t < tiles; ++t) {
        unsigned int aCol = t * TILE + tx;
        unsigned int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? Ab[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? Bb[bRow * N + col] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < TILE; ++i) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)batch * M * N + row * N + col] = acc;
}
"#;

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

/// An owned device-resident f32 buffer. Free it with [`ZenEngine::free_dev`] (returns it to the
/// pool) — it does NOT free on Drop (no engine handle here).
pub struct DevTensor {
    pub(crate) buf: zengpu::BufferHandle,
    pub(crate) len: usize,
}

impl DevTensor {
    pub fn len(&self) -> usize {
        self.len
    }
}

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

    /// Compiled raw HIP pipeline, building and caching it under `name` on first use.
    fn pipeline_hip(
        &self,
        name: &'static str,
        src: &'static str,
        block: [u32; 3],
    ) -> Result<zengpu::PipelineHandle> {
        if let Some(cached) = self.pipelines.lock().unwrap().get(name) {
            return Ok(cached.pipeline);
        }
        let desc = zengpu::ShaderDesc::hip(src);
        let entry = "zsl_kernel";
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

    pub fn upload_dev(&self, data: &[f32]) -> Result<DevTensor> {
        let buf = self.upload(data).map_err(|e| err(e.to_string()))?;
        Ok(DevTensor { buf, len: data.len() })
    }

    pub fn alloc_dev(&self, len: usize) -> Result<DevTensor> {
        let buf = self.alloc(len).map_err(|e| err(e.to_string()))?;
        Ok(DevTensor { buf, len })
    }

    pub fn download_dev(&self, t: &DevTensor) -> Result<Vec<f32>> {
        self.download(t.buf, t.len).map_err(|e| err(e.to_string()))
    }

    pub fn free_dev(&self, t: DevTensor) {
        self.recycle(t.buf, t.len);
    }

    pub fn matmul_dev(
        &self,
        a: &DevTensor,
        b: &DevTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<DevTensor> {
        let c = self.alloc_dev(m * n)?;

        let pipeline = if self.backend == BackendPreference::Hip {
            self.pipeline_hip("sgemm_tiled", TILED_SGEMM_HIP, [16, 16, 1])?
        } else {
            self.pipeline_for(&SGEMM, "sgemm", [16, 16, 1])?
        };

        let bindings = Bindings {
            buffers:  &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        Ok(c)
    }

    pub fn matmul_batched_dev(
        &self,
        a: &DevTensor,
        b: &DevTensor,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<DevTensor> {
        let c = self.alloc_dev(batch * m * n)?;

        let pipeline = if self.backend == BackendPreference::Hip {
            self.pipeline_hip("bgemm_tiled", TILED_BGEMM_HIP, [16, 16, 1])?
        } else {
            self.pipeline_for(&BGEMM, "bgemm", [16, 16, 1])?
        };

        let bindings = Bindings {
            buffers:  &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars:  &[Scalar::U32(m as u32), Scalar::U32(n as u32), Scalar::U32(k as u32)],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        Ok(c)
    }

    fn run_binary_dev(
        &self,
        a: &DevTensor,
        b: &DevTensor,
        shader: &ZslShader,
        name: &'static str,
    ) -> Result<DevTensor> {
        if a.len != b.len {
            return Err(err(format!(
                "device tensor length mismatch: {} != {}",
                a.len, b.len
            )));
        }

        let c = self.alloc_dev(a.len)?;
        let pipeline = self.pipeline_for(shader, name, [256, 1, 1])?;
        let bindings = Bindings {
            buffers:  &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars:  &[Scalar::U32(a.len as u32)],
        };
        let grid = [(a.len as u32 + 255) / 256, 1, 1];
        self.device.dispatch(pipeline, bindings, grid).map_err(|e| err(e.to_string()))?;

        Ok(c)
    }

    pub fn add_dev(&self, a: &DevTensor, b: &DevTensor) -> Result<DevTensor> {
        self.run_binary_dev(a, b, &ADD, "add")
    }

    pub fn sub_dev(&self, a: &DevTensor, b: &DevTensor) -> Result<DevTensor> {
        self.run_binary_dev(a, b, &SUB, "sub")
    }

    pub fn mul_dev(&self, a: &DevTensor, b: &DevTensor) -> Result<DevTensor> {
        self.run_binary_dev(a, b, &MUL, "mul")
    }

    pub fn div_dev(&self, a: &DevTensor, b: &DevTensor) -> Result<DevTensor> {
        self.run_binary_dev(a, b, &DIV, "div")
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

        let pipeline = if self.backend == BackendPreference::Hip {
            self.pipeline_hip("sgemm_tiled", TILED_SGEMM_HIP, [16, 16, 1])?
        } else {
            self.pipeline_for(&SGEMM, "sgemm", [16, 16, 1])?
        };

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

        let pipeline = if self.backend == BackendPreference::Hip {
            self.pipeline_hip("bgemm_tiled", TILED_BGEMM_HIP, [16, 16, 1])?
        } else {
            self.pipeline_for(&BGEMM, "bgemm", [16, 16, 1])?
        };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_resident_matmul() -> Result<()> {
        let engine = match ZenEngine::new() {
            Ok(engine) => engine,
            Err(error) => {
                println!("skipping dev_resident_matmul: {error}");
                return Ok(());
            }
        };

        const SIZE: usize = 64;
        let a: Vec<f32> = (0..SIZE * SIZE)
            .map(|i| ((i * 17 + 3) % 101) as f32 / 101.0)
            .collect();
        let b: Vec<f32> = (0..SIZE * SIZE)
            .map(|i| ((i * 29 + 7) % 103) as f32 / 103.0)
            .collect();
        let mut expected = vec![0.0; SIZE * SIZE];
        for row in 0..SIZE {
            for col in 0..SIZE {
                for inner in 0..SIZE {
                    expected[row * SIZE + col] +=
                        a[row * SIZE + inner] * b[inner * SIZE + col];
                }
            }
        }

        let da = engine.upload_dev(&a)?;
        let db = engine.upload_dev(&b)?;
        let dc = engine.matmul_dev(&da, &db, SIZE, SIZE, SIZE)?;
        let actual = engine.download_dev(&dc)?;
        let max_err = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        println!("dev_resident max err: {max_err}");

        engine.free_dev(da);
        engine.free_dev(db);
        engine.free_dev(dc);

        assert!(max_err < 1e-3, "max error {max_err} exceeded tolerance");
        Ok(())
    }
}
