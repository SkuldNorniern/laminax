#![cfg(feature = "gpu")]

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zengpu::{
    BackendPreference, Bindings, BufferDesc, BufferUsage, ComputePipelineDesc, GpuDevice, Instance,
    MemoryUsage, PipelineHandle, Scalar, ShaderDesc,
};
use zengpu_spirv::{ZslShader, zsl};

use crate::RuntimeError;

pub type Result<T> = std::result::Result<T, RuntimeError>;

static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static DISPATCH_NS: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_NS: AtomicU64 = AtomicU64::new(0);
static ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static N_DISPATCH: AtomicU64 = AtomicU64::new(0);

fn prof() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZEN_PROF").is_ok())
}

pub fn prof_report() -> String {
    let upload_ms = UPLOAD_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
    let dispatch_ms = DISPATCH_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
    let download_ms = DOWNLOAD_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
    let alloc_ms = ALLOC_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
    let n_dispatch = N_DISPATCH.swap(0, Ordering::Relaxed);
    format!(
        "UPLOAD {upload_ms:.3} ms | DISPATCH {dispatch_ms:.3} ms | DOWNLOAD {download_ms:.3} ms | ALLOC {alloc_ms:.3} ms | dispatch count {n_dispatch}"
    )
}

fn err(s: impl Into<String>) -> RuntimeError {
    RuntimeError::Execution(s.into())
}

fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn cast_u8(v: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
}

const TILED_SGEMM_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32 }
    @workgroup_size(16, 16)
    kernel tiled_sgemm(
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        shared as_tile: array<f32, 256>
        shared bs_tile: array<f32, 256>
        let lx = local_id().x
        let ly = local_id().y
        let row = group_id().y * 16 + ly
        let col = group_id().x * 16 + lx
        let sum: f32 = 0.0
        let tile_count = (p.k + 15) / 16
        for tile in 0..tile_count {
            let ak = tile * 16 + lx
            let bk = tile * 16 + ly
            if row < p.m && ak < p.k {
                as_tile[ly * 16 + lx] = a[row * p.k + ak]
            } else {
                as_tile[ly * 16 + lx] = 0.0
            }
            if bk < p.k && col < p.n {
                bs_tile[ly * 16 + lx] = b[bk * p.n + col]
            } else {
                bs_tile[ly * 16 + lx] = 0.0
            }
            barrier()
            for q in 0..16 {
                sum = sum + as_tile[ly * 16 + q] * bs_tile[q * 16 + lx]
            }
            barrier()
        }
        if row < p.m && col < p.n {
            c[row * p.n + col] = sum
        }
    }
);

const TILED_BGEMM_ZSL: ZslShader = zsl!(
    push P { m: u32, n: u32, k: u32 }
    @workgroup_size(16, 16)
    kernel tiled_bgemm(
        a: device buffer<f32>,
        b: device buffer<f32>,
        c: device mut buffer<f32>,
        p: P,
    ) {
        shared as_tile: array<f32, 256>
        shared bs_tile: array<f32, 256>
        let lx = local_id().x
        let ly = local_id().y
        let row = group_id().y * 16 + ly
        let col = group_id().x * 16 + lx
        let batch = group_id().z
        let ao = batch * p.m * p.k
        let bo = batch * p.k * p.n
        let sum: f32 = 0.0
        let tile_count = (p.k + 15) / 16
        for tile in 0..tile_count {
            let ak = tile * 16 + lx
            let bk = tile * 16 + ly
            if row < p.m && ak < p.k {
                as_tile[ly * 16 + lx] = a[ao + row * p.k + ak]
            } else {
                as_tile[ly * 16 + lx] = 0.0
            }
            if bk < p.k && col < p.n {
                bs_tile[ly * 16 + lx] = b[bo + bk * p.n + col]
            } else {
                bs_tile[ly * 16 + lx] = 0.0
            }
            barrier()
            for q in 0..16 {
                sum = sum + as_tile[ly * 16 + q] * bs_tile[q * 16 + lx]
            }
            barrier()
        }
        if row < p.m && col < p.n {
            c[batch * p.m * p.n + row * p.n + col] = sum
        }
    }
);

const SOFTMAX_FWD_ZSL: ZslShader = zsl!(
    push P { rows: u32, d: u32 }
    @workgroup_size(256)
    kernel softmax_fwd(
        x: device buffer<f32>,
        y: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.rows {
            let base = r * p.d
            let m: f32 = -3.402823e38
            for j in 0..p.d {
                m = max(m, x[base + j])
            }
            let sum: f32 = 0.0
            for j in 0..p.d {
                let e = exp(x[base + j] - m)
                y[base + j] = e
                sum = sum + e
            }
            let inv = 1.0 / sum
            for j in 0..p.d {
                y[base + j] = y[base + j] * inv
            }
        }
    }
);

const SOFTMAX_BWD_ZSL: ZslShader = zsl!(
    push P { rows: u32, d: u32 }
    @workgroup_size(256)
    kernel softmax_bwd(
        y: device buffer<f32>,
        g: device buffer<f32>,
        dx: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.rows {
            let base = r * p.d
            let dot: f32 = 0.0
            for j in 0..p.d {
                dot = dot + g[base + j] * y[base + j]
            }
            for j in 0..p.d {
                dx[base + j] = y[base + j] * (g[base + j] - dot)
            }
        }
    }
);

const GELU_FWD_ZSL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel gelu_fwd(
        x: device buffer<f32>,
        y: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            let v = x[i]
            let t = tanh(0.7978845608 * (v + 0.044715 * v * v * v))
            y[i] = 0.5 * v * (1.0 + t)
        }
    }
);

const GELU_BWD_ZSL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel gelu_bwd(
        x: device buffer<f32>,
        g: device buffer<f32>,
        dx: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            let v = x[i]
            let inner = 0.7978845608 * (v + 0.044715 * v * v * v)
            let t = tanh(inner)
            let dinner = 0.7978845608 * (1.0 + 3.0 * 0.044715 * v * v)
            let dt = (1.0 - t * t) * dinner
            dx[i] = g[i] * (0.5 * (1.0 + t) + 0.5 * v * dt)
        }
    }
);

const ZERO_PAIR_ZSL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel zero_pair(
        a: device mut buffer<f32>,
        b: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            a[i] = 0.0
            b[i] = 0.0
        }
    }
);

const LAYERNORM_FWD_ZSL: ZslShader = zsl!(
    push P { rows: u32, d: u32, eps: f32 }
    @workgroup_size(256)
    kernel layernorm_fwd(
        x: device buffer<f32>,
        gamma: device buffer<f32>,
        beta: device buffer<f32>,
        out: device mut buffer<f32>,
        xhat: device mut buffer<f32>,
        invstd: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.rows {
            let base = r * p.d
            let mean: f32 = 0.0
            for j in 0..p.d {
                mean = mean + x[base + j]
            }
            mean = mean / p.d
            let var: f32 = 0.0
            for j in 0..p.d {
                let delta = x[base + j] - mean
                var = var + delta * delta
            }
            var = var / p.d
            let is = 1.0 / sqrt(var + p.eps)
            invstd[r] = is
            for j in 0..p.d {
                let h = (x[base + j] - mean) * is
                xhat[base + j] = h
                out[base + j] = h * gamma[j] + beta[j]
            }
        }
    }
);

const LAYERNORM_BWD_ZSL: ZslShader = zsl!(
    push P { rows: u32, d: u32 }
    @workgroup_size(256)
    kernel layernorm_bwd(
        g: device buffer<f32>,
        xhat: device buffer<f32>,
        invstd: device buffer<f32>,
        gamma: device buffer<f32>,
        dx: device mut buffer<f32>,
        dgamma: device mut buffer<f32>,
        dbeta: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.rows {
            let base = r * p.d
            let md: f32 = 0.0
            let mdx: f32 = 0.0
            for j in 0..p.d {
                let dh = g[base + j] * gamma[j]
                md = md + dh
                mdx = mdx + dh * xhat[base + j]
            }
            md = md / p.d
            mdx = mdx / p.d
            for j in 0..p.d {
                let dh = g[base + j] * gamma[j]
                dx[base + j] = invstd[r] * (dh - md - xhat[base + j] * mdx)
                atomic_add(dgamma, j, g[base + j] * xhat[base + j])
                atomic_add(dbeta, j, g[base + j])
            }
        }
    }
);

const COPY_ZSL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel copy(x: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            out[i] = x[i]
        }
    }
);

const SCALE_ZSL: ZslShader = zsl!(
    push P { n: u32, scale: f32 }
    @workgroup_size(256)
    kernel scale(x: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            out[i] = x[i] * p.scale
        }
    }
);

const TRANSPOSE2D_ZSL: ZslShader = zsl!(
    push P { rows: u32, cols: u32 }
    @workgroup_size(256)
    kernel transpose2d(x: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.rows * p.cols {
            let r = i / p.cols
            let c = i - r * p.cols
            out[c * p.rows + r] = x[i]
        }
    }
);

const TRANSPOSE_LAST2_ZSL: ZslShader = zsl!(
    push P { batch: u32, rows: u32, cols: u32 }
    @workgroup_size(256)
    kernel transpose_last2(x: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        let plane = p.rows * p.cols
        if i < p.batch * plane {
            let b = i / plane
            let q = i - b * plane
            let r = q / p.cols
            let c = q - r * p.cols
            out[b * plane + c * p.rows + r] = x[i]
        }
    }
);

const CE_FWD_ZSL: ZslShader = zsl!(
    push P { n: u32, v: u32 }
    @workgroup_size(256)
    kernel ce_fwd(
        logits: device buffer<f32>,
        tgt: device buffer<f32>,
        probs: device mut buffer<f32>,
        rowloss: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.n {
            let base = r * p.v
            let m: f32 = -3.402823e38
            for j in 0..p.v {
                m = max(m, logits[base + j])
            }
            let sum: f32 = 0.0
            for j in 0..p.v {
                let e = exp(logits[base + j] - m)
                probs[base + j] = e
                sum = sum + e
            }
            let inv = 1.0 / sum
            for j in 0..p.v {
                probs[base + j] = probs[base + j] * inv
            }
            let t = u32(tgt[r])
            rowloss[r] = -log(probs[base + t] + 1e-12)
        }
    }
);

const CE_BWD_ZSL: ZslShader = zsl!(
    push P { n: u32, v: u32, scale: f32 }
    @workgroup_size(256)
    kernel ce_bwd(
        probs: device buffer<f32>,
        tgt: device buffer<f32>,
        dlogits: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let r = id.x
        if r < p.n {
            let base = r * p.v
            let t = u32(tgt[r])
            for j in 0..p.v {
                let expected: f32 = 0.0
                if j == t {
                    expected = 1.0
                }
                dlogits[base + j] = (probs[base + j] - expected) * p.scale
            }
        }
    }
);

const EMB_GATHER_ZSL: ZslShader = zsl!(
    push P { n: u32, c: u32 }
    @workgroup_size(256)
    kernel emb_gather(
        weight: device buffer<f32>,
        idx: device buffer<f32>,
        out: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n * p.c {
            let r = i / p.c
            let col = i - r * p.c
            let row = u32(idx[r])
            out[i] = weight[row * p.c + col]
        }
    }
);

const EMB_SCATTER_ZSL: ZslShader = zsl!(
    push P { n: u32, c: u32 }
    @workgroup_size(256)
    kernel emb_scatter(
        g: device buffer<f32>,
        idx: device buffer<f32>,
        dw: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n * p.c {
            let r = i / p.c
            let col = i - r * p.c
            let row = u32(idx[r])
            atomic_add(dw, row * p.c + col, g[i])
        }
    }
);

const BIAS_ADD_ZSL: ZslShader = zsl!(
    push P { n: u32, c: u32 }
    @workgroup_size(256)
    kernel bias_add(
        x: device buffer<f32>,
        bias: device buffer<f32>,
        out: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            let col = i - (i / p.c) * p.c
            out[i] = x[i] + bias[col]
        }
    }
);

const BIAS_ROWSUM_ZSL: ZslShader = zsl!(
    push P { rows: u32, c: u32 }
    @workgroup_size(256)
    kernel bias_rowsum(
        g: device buffer<f32>,
        dbias: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let col = id.x
        if col < p.c {
            let sum: f32 = 0.0
            for r in 0..p.rows {
                sum = sum + g[r * p.c + col]
            }
            dbias[col] = sum
        }
    }
);

const SLICE_COLS_ZSL: ZslShader = zsl!(
    push P { rows: u32, cols: u32, len: u32, start: u32 }
    @workgroup_size(256)
    kernel slice_cols(x: device buffer<f32>, out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.rows * p.len {
            let r = i / p.len
            let l = i - r * p.len
            out[i] = x[r * p.cols + p.start + l]
        }
    }
);

const SLICE_COLS_BWD_ZSL: ZslShader = zsl!(
    push P { rows: u32, cols: u32, len: u32, start: u32 }
    @workgroup_size(256)
    kernel slice_cols_bwd(
        g: device buffer<f32>,
        dx: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.rows * p.len {
            let r = i / p.len
            let l = i - r * p.len
            dx[r * p.cols + p.start + l] = g[i]
        }
    }
);

const ZERO_ZSL: ZslShader = zsl!(
    push P { n: u32 }
    @workgroup_size(256)
    kernel zero(out: device mut buffer<f32>, p: P, id: global_id) {
        let i = id.x
        if i < p.n {
            out[i] = 0.0
        }
    }
);

const ADAM_STEP_ZSL: ZslShader = zsl!(
    push P { lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32, n: u32 }
    @workgroup_size(256)
    kernel adam_step(
        w: device mut buffer<f32>,
        g: device buffer<f32>,
        m: device mut buffer<f32>,
        v: device mut buffer<f32>,
        p: P,
        id: global_id,
    ) {
        let i = id.x
        if i < p.n {
            let gi = g[i]
            let mi = p.b1 * m[i] + (1.0 - p.b1) * gi
            let vi = p.b2 * v[i] + (1.0 - p.b2) * gi * gi
            m[i] = mi
            v[i] = vi
            let wi = w[i]
            if p.wd != 0.0 {
                wi = wi - p.lr * p.wd * wi
            }
            wi = wi - p.lr * (mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps)
            w[i] = wi
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

const COPY_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ out, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = x[i];
}
"#;

const SCALE_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ out,
                unsigned int N, float scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = x[i] * scale;
}
"#;

const TRANSPOSE2D_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ out,
                unsigned int rows, unsigned int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = rows * cols;
    if (i < n) {
        unsigned int r = i / cols, c = i - r * cols;
        out[c * rows + r] = x[i];
    }
}
"#;

const TRANSPOSE_LAST2_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ out,
                unsigned int batch, unsigned int rows, unsigned int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int plane = rows * cols, n = batch * plane;
    if (i < n) {
        unsigned int b = i / plane, p = i - b * plane;
        unsigned int r = p / cols, c = p - r * cols;
        out[b * plane + c * rows + r] = x[i];
    }
}
"#;

const SOFTMAX_FWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ y,
                unsigned int R, unsigned int D) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    const float* xr = x + (size_t)r * D; float* yr = y + (size_t)r * D;
    float m = -3.402823e38f;
    for (unsigned int j=0;j<D;++j) m = fmaxf(m, xr[j]);
    float s = 0.f;
    for (unsigned int j=0;j<D;++j) { float e = __expf(xr[j]-m); yr[j]=e; s+=e; }
    float inv = 1.f/s;
    for (unsigned int j=0;j<D;++j) yr[j]*=inv;
}
"#;

const SOFTMAX_BWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ y, const float* __restrict__ g,
                float* __restrict__ dx, unsigned int R, unsigned int D) {
    unsigned int r = blockIdx.x*blockDim.x+threadIdx.x; if (r>=R) return;
    const float* yr=y+(size_t)r*D; const float* gr=g+(size_t)r*D; float* o=dx+(size_t)r*D;
    float dot=0.f; for (unsigned int j=0;j<D;++j) dot+=gr[j]*yr[j];
    for (unsigned int j=0;j<D;++j) o[j]=yr[j]*(gr[j]-dot);
}
"#;

const GELU_FWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ y, unsigned int N){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  float v=x[i]; float t=tanhf(0.7978845608f*(v+0.044715f*v*v*v));
  y[i]=0.5f*v*(1.f+t);
}
"#;

const GELU_BWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, const float* __restrict__ g,
                float* __restrict__ dx, unsigned int N){
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  float v=x[i]; float inner=0.7978845608f*(v+0.044715f*v*v*v); float t=tanhf(inner);
  float dinner=0.7978845608f*(1.f+3.f*0.044715f*v*v); float dt=(1.f-t*t)*dinner;
  dx[i]=g[i]*(0.5f*(1.f+t)+0.5f*v*dt);
}
"#;

const ZERO_PAIR_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(float* __restrict__ a, float* __restrict__ b, unsigned int N) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  a[i]=0.f; b[i]=0.f;
}
"#;

const LAYERNORM_FWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                const float* __restrict__ beta, float* __restrict__ out,
                float* __restrict__ xhat, float* __restrict__ invstd,
                unsigned int R, unsigned int D, float eps){
  unsigned int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R)return;
  const float* xr=x+(size_t)r*D; float* o=out+(size_t)r*D; float* xh=xhat+(size_t)r*D;
  float mean=0.f; for(unsigned int j=0;j<D;++j) mean+=xr[j]; mean/=D;
  float var=0.f; for(unsigned int j=0;j<D;++j){float d=xr[j]-mean; var+=d*d;} var/=D;
  float is=rsqrtf(var+eps); invstd[r]=is;
  for(unsigned int j=0;j<D;++j){float h=(xr[j]-mean)*is; xh[j]=h; o[j]=h*gamma[j]+beta[j];}
}
"#;

const LAYERNORM_BWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ g, const float* __restrict__ xhat,
                const float* __restrict__ invstd, const float* __restrict__ gamma,
                float* __restrict__ dx, float* __restrict__ dgamma,
                float* __restrict__ dbeta, unsigned int R, unsigned int D){
  unsigned int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R)return;
  const float* gr=g+(size_t)r*D; const float* xh=xhat+(size_t)r*D; float* o=dx+(size_t)r*D;
  float is=invstd[r];
  float md=0.f, mdx=0.f;
  for(unsigned int j=0;j<D;++j){float dh=gr[j]*gamma[j]; md+=dh; mdx+=dh*xh[j];}
  md/=D; mdx/=D;
  for(unsigned int j=0;j<D;++j){
    float dh=gr[j]*gamma[j];
    o[j]=is*(dh-md-xh[j]*mdx);
    atomicAdd(&dgamma[j], gr[j]*xh[j]);
    atomicAdd(&dbeta[j],  gr[j]);
  }
}
"#;

const CE_FWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ logits, const float* __restrict__ tgt,
                float* __restrict__ probs, float* __restrict__ rowloss,
                unsigned int N, unsigned int V) {
  unsigned int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=N)return;
  const float* x=logits+(size_t)r*V; float* p=probs+(size_t)r*V;
  float m=-3.402823e38f;
  for(unsigned int j=0;j<V;++j)m=fmaxf(m,x[j]);
  float sum=0.f;
  for(unsigned int j=0;j<V;++j){float e=__expf(x[j]-m);p[j]=e;sum+=e;}
  float inv=1.f/sum;
  for(unsigned int j=0;j<V;++j)p[j]*=inv;
  unsigned int t=(unsigned int)tgt[r];
  rowloss[r]=-logf(p[t]+1e-12f);
}
"#;

const CE_BWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ probs, const float* __restrict__ tgt,
                float* __restrict__ dlogits, unsigned int N, unsigned int V,
                float scale) {
  unsigned int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=N)return;
  unsigned int t=(unsigned int)tgt[r];
  const float* p=probs+(size_t)r*V; float* d=dlogits+(size_t)r*V;
  for(unsigned int j=0;j<V;++j)d[j]=(p[j]-(j==t?1.f:0.f))*scale;
}
"#;

const EMB_GATHER_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ weight, const float* __restrict__ idx,
                float* __restrict__ out, unsigned int N, unsigned int C) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int total=N*C; if(i>=total)return;
  unsigned int r=i/C, c=i-r*C, row=(unsigned int)idx[r];
  out[i]=weight[(size_t)row*C+c];
}
"#;

const EMB_SCATTER_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ g, const float* __restrict__ idx,
                float* __restrict__ dw, unsigned int N, unsigned int C) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int total=N*C; if(i>=total)return;
  unsigned int r=i/C, c=i-r*C, row=(unsigned int)idx[r];
  atomicAdd(&dw[(size_t)row*C+c],g[i]);
}
"#;

const BIAS_ADD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                float* __restrict__ out, unsigned int N, unsigned int C) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N)out[i]=x[i]+bias[i%C];
}
"#;

const BIAS_ROWSUM_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ g, float* __restrict__ dbias,
                unsigned int rows, unsigned int C) {
  unsigned int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=C)return;
  float sum=0.f; for(unsigned int r=0;r<rows;++r)sum+=g[(size_t)r*C+c];
  dbias[c]=sum;
}
"#;

const SLICE_COLS_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ x, float* __restrict__ out,
                unsigned int R, unsigned int C, unsigned int L, unsigned int start) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int total=R*L; if(i>=total)return;
  unsigned int r=i/L, l=i-r*L; out[i]=x[(size_t)r*C+start+l];
}
"#;

const SLICE_COLS_BWD_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(const float* __restrict__ g, float* __restrict__ dx,
                unsigned int R, unsigned int C, unsigned int L, unsigned int start) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int total=R*L; if(i>=total)return;
  unsigned int r=i/L, l=i-r*L; dx[(size_t)r*C+start+l]=g[i];
}
"#;

const ZERO_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(float* __restrict__ out, unsigned int N) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N)out[i]=0.f;
}
"#;

const ADAM_STEP_HIP: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void zsl_kernel(float* __restrict__ w, const float* __restrict__ g,
                float* __restrict__ m, float* __restrict__ v,
                float lr, float b1, float b2, float eps, float wd,
                float bc1, float bc2, unsigned int N) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
  float gi=g[i];
  float mi=b1*m[i]+(1.f-b1)*gi;
  float vi=b2*v[i]+(1.f-b2)*gi*gi;
  m[i]=mi; v[i]=vi;
  float wi=w[i];
  if(wd!=0.f)wi-=lr*wd*wi;
  wi-=lr*(mi/bc1)/(sqrtf(vi/bc2)+eps);
  w[i]=wi;
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
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
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
        let (instance, backend) = build_instance().map_err(|e| err(format!("ZenGPU init: {e}")))?;
        let adapters = instance.enumerate_adapters();
        if adapters.is_empty() {
            return Err(err("no ZenGPU adapters found"));
        }
        let adapter = adapters.get(index).ok_or_else(|| {
            err(format!(
                "adapter {index} out of range ({} found)",
                adapters.len()
            ))
        })?;
        let device_name = adapter.info().name.clone();
        let device: Arc<dyn GpuDevice> = Arc::from(
            adapter
                .open(zengpu::DeviceRequest::default())
                .map_err(|e| err(format!("open device: {e}")))?,
        );
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
        let start = prof().then(Instant::now);
        let result = self.device.create_buffer(BufferDesc {
            size: (n * 4) as u64,
            usage: BufferUsage::STORAGE | BufferUsage::READBACK,
            memory: MemoryUsage::GpuOnly,
        });
        if let Some(start) = start {
            ALLOC_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        result
    }

    fn recycle(&self, buf: zengpu::BufferHandle, n: usize) {
        let mut pool = self.pool.lock().unwrap();
        let already_pooled = pool
            .values()
            .any(|bucket| bucket.iter().any(|&pooled| pooled == buf));
        debug_assert!(
            !already_pooled,
            "device buffer recycled twice: index={}, generation={}",
            buf.index(),
            buf.generation(),
        );
        // Keep the release-build guard as a last line of defence. Destroying a
        // duplicate here would free the allocation still represented by the
        // first pool entry, so the only safe response is to retain one entry.
        if already_pooled {
            return;
        }
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
        name: &str,
        block: [u32; 3],
    ) -> Result<PipelineHandle> {
        if let Some(cached) = self.pipelines.lock().unwrap().get(name) {
            return Ok(cached.pipeline);
        }
        let (desc, entry) = self.pick(shader);
        let sh = self
            .device
            .create_shader(desc)
            .map_err(|e| err(e.to_string()))?;
        let pipeline = self
            .device
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry,
                block,
            })
            .map_err(|e| err(e.to_string()))?;
        self.pipelines.lock().unwrap().insert(
            name.to_owned(),
            CachedPipeline {
                shader: sh,
                pipeline,
            },
        );
        Ok(pipeline)
    }

    /// Compiled raw HIP pipeline, building and caching it under `name` on first use.
    fn pipeline_hip(
        &self,
        name: &str,
        src: &'static str,
        block: [u32; 3],
    ) -> Result<PipelineHandle> {
        if let Some(cached) = self.pipelines.lock().unwrap().get(name) {
            return Ok(cached.pipeline);
        }
        let desc = zengpu::ShaderDesc::hip(src);
        let entry = "zsl_kernel";
        let sh = self
            .device
            .create_shader(desc)
            .map_err(|e| err(e.to_string()))?;
        let pipeline = self
            .device
            .create_compute_pipeline(ComputePipelineDesc {
                shader: sh,
                entry,
                block,
            })
            .map_err(|e| err(e.to_string()))?;
        self.pipelines.lock().unwrap().insert(
            name.to_owned(),
            CachedPipeline {
                shader: sh,
                pipeline,
            },
        );
        Ok(pipeline)
    }

    /// Select the portable ZSL kernel unless HIP has an override and it has not
    /// been disabled for parity testing. Raw and portable variants use distinct
    /// cache keys so an engine can safely compile both.
    fn kernel_pipeline(
        &self,
        name: &'static str,
        zsl: &ZslShader,
        raw_hip: Option<&'static str>,
        block: [u32; 3],
    ) -> Result<PipelineHandle> {
        if self.backend == BackendPreference::Hip
            && raw_hip.is_some()
            && std::env::var("ZEN_FORCE_ZSL").is_err()
        {
            self.pipeline_hip(name, raw_hip.unwrap(), block)
        } else {
            let name_zsl = format!("{name}_zsl");
            self.pipeline_for(zsl, &name_zsl, block)
        }
    }

    fn upload(&self, data: &[f32]) -> zengpu::Result<zengpu::BufferHandle> {
        let buf = self.alloc(data.len())?;
        let start = prof().then(Instant::now);
        let result = self.device.write_buffer(buf, 0, cast_f32(data));
        if let Some(start) = start {
            UPLOAD_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        result?;
        Ok(buf)
    }

    fn download(&self, buf: zengpu::BufferHandle, n: usize) -> zengpu::Result<Vec<f32>> {
        let start = prof().then(Instant::now);
        let result = self.device.read_buffer(buf, 0, (n * 4) as u64);
        if let Some(start) = start {
            DOWNLOAD_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let raw = result?;
        Ok(cast_u8(&raw).to_vec())
    }

    pub fn upload_dev(&self, data: &[f32]) -> Result<DevTensor> {
        let buf = self.upload(data).map_err(|e| err(e.to_string()))?;
        Ok(DevTensor {
            buf,
            len: data.len(),
        })
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

    fn dispatch_profiled(
        &self,
        pipeline: zengpu::PipelineHandle,
        bindings: Bindings<'_>,
        grid: [u32; 3],
    ) -> Result<()> {
        let start = prof().then(Instant::now);
        let result = self.device.dispatch(pipeline, bindings, grid);
        if let Some(start) = start {
            DISPATCH_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N_DISPATCH.fetch_add(1, Ordering::Relaxed);
        }
        result.map_err(|e| err(e.to_string()))
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

        let pipeline = self.kernel_pipeline(
            "sgemm_tiled",
            &TILED_SGEMM_ZSL,
            Some(TILED_SGEMM_HIP),
            [16, 16, 1],
        )?;

        let bindings = Bindings {
            buffers: &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(m as u32),
                Scalar::U32(n as u32),
                Scalar::U32(k as u32),
            ],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, 1];
        self.device
            .dispatch(pipeline, bindings, grid)
            .map_err(|e| err(e.to_string()))?;

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

        let pipeline = self.kernel_pipeline(
            "bgemm_tiled",
            &TILED_BGEMM_ZSL,
            Some(TILED_BGEMM_HIP),
            [16, 16, 1],
        )?;

        let bindings = Bindings {
            buffers: &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(m as u32),
                Scalar::U32(n as u32),
                Scalar::U32(k as u32),
            ],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32];
        self.device
            .dispatch(pipeline, bindings, grid)
            .map_err(|e| err(e.to_string()))?;

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
        let pipeline = self.kernel_pipeline(name, shader, None, [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[a.buf.index(), b.buf.index(), c.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(a.len as u32)],
        };
        let grid = [(a.len as u32 + 255) / 256, 1, 1];
        self.device
            .dispatch(pipeline, bindings, grid)
            .map_err(|e| err(e.to_string()))?;

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

    pub fn copy_dev(&self, x: &DevTensor) -> Result<DevTensor> {
        let out = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline("copy", &COPY_ZSL, Some(COPY_HIP), [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(x.len as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(x.len as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn scale_dev(&self, x: &DevTensor, scale: f32) -> Result<DevTensor> {
        let out = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline("scale", &SCALE_ZSL, Some(SCALE_HIP), [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(x.len as u32), Scalar::F32(scale)],
        };
        self.dispatch_profiled(pipeline, bindings, [(x.len as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn transpose2d_dev(&self, x: &DevTensor, rows: usize, cols: usize) -> Result<DevTensor> {
        if x.len != rows * cols {
            return Err(err("transpose2d_dev input length mismatch"));
        }
        let out = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline(
            "transpose2d",
            &TRANSPOSE2D_ZSL,
            Some(TRANSPOSE2D_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(rows as u32), Scalar::U32(cols as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(x.len as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn transpose_last2_dev(
        &self,
        x: &DevTensor,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Result<DevTensor> {
        if x.len != batch * rows * cols {
            return Err(err("transpose_last2_dev input length mismatch"));
        }
        let out = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline(
            "transpose_last2",
            &TRANSPOSE_LAST2_ZSL,
            Some(TRANSPOSE_LAST2_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(batch as u32),
                Scalar::U32(rows as u32),
                Scalar::U32(cols as u32),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [(x.len as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn softmax_dev(&self, x: &DevTensor, rows: usize, d: usize) -> Result<DevTensor> {
        if x.len != rows * d {
            return Err(err("softmax_dev input length mismatch"));
        }
        let y = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline(
            "softmax_fwd",
            &SOFTMAX_FWD_ZSL,
            Some(SOFTMAX_FWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), y.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(rows as u32), Scalar::U32(d as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(rows as u32 + 255) / 256, 1, 1])?;
        Ok(y)
    }

    pub fn softmax_bwd_dev(
        &self,
        y: &DevTensor,
        g: &DevTensor,
        rows: usize,
        d: usize,
    ) -> Result<DevTensor> {
        if y.len != rows * d || g.len != y.len {
            return Err(err("softmax_bwd_dev input length mismatch"));
        }
        let dx = self.alloc_dev(y.len)?;
        let pipeline = self.kernel_pipeline(
            "softmax_bwd",
            &SOFTMAX_BWD_ZSL,
            Some(SOFTMAX_BWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[y.buf.index(), g.buf.index(), dx.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(rows as u32), Scalar::U32(d as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(rows as u32 + 255) / 256, 1, 1])?;
        Ok(dx)
    }

    pub fn gelu_dev(&self, x: &DevTensor, n: usize) -> Result<DevTensor> {
        if x.len != n {
            return Err(err("gelu_dev input length mismatch"));
        }
        let y = self.alloc_dev(n)?;
        let pipeline = self.kernel_pipeline(
            "gelu_fwd",
            &GELU_FWD_ZSL,
            Some(GELU_FWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), y.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(n as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1])?;
        Ok(y)
    }

    pub fn gelu_bwd_dev(&self, x: &DevTensor, g: &DevTensor, n: usize) -> Result<DevTensor> {
        if x.len != n || g.len != n {
            return Err(err("gelu_bwd_dev input length mismatch"));
        }
        let dx = self.alloc_dev(n)?;
        let pipeline = self.kernel_pipeline(
            "gelu_bwd",
            &GELU_BWD_ZSL,
            Some(GELU_BWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), g.buf.index(), dx.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(n as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1])?;
        Ok(dx)
    }

    pub fn layernorm_dev(
        &self,
        x: &DevTensor,
        gamma: &DevTensor,
        beta: &DevTensor,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(DevTensor, DevTensor, DevTensor)> {
        if x.len != rows * d || gamma.len != d || beta.len != d {
            return Err(err("layernorm_dev input length mismatch"));
        }
        let out = self.alloc_dev(x.len)?;
        let xhat = self.alloc_dev(x.len)?;
        let invstd = self.alloc_dev(rows)?;
        let pipeline = self.kernel_pipeline(
            "layernorm_fwd",
            &LAYERNORM_FWD_ZSL,
            Some(LAYERNORM_FWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[
                x.buf.index(),
                gamma.buf.index(),
                beta.buf.index(),
                out.buf.index(),
                xhat.buf.index(),
                invstd.buf.index(),
            ],
            textures: &[],
            scalars: &[
                Scalar::U32(rows as u32),
                Scalar::U32(d as u32),
                Scalar::F32(eps),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [(rows as u32 + 255) / 256, 1, 1])?;
        Ok((out, xhat, invstd))
    }

    pub fn layernorm_bwd_dev(
        &self,
        g: &DevTensor,
        xhat: &DevTensor,
        invstd: &DevTensor,
        gamma: &DevTensor,
        rows: usize,
        d: usize,
    ) -> Result<(DevTensor, DevTensor, DevTensor)> {
        if g.len != rows * d || xhat.len != g.len || invstd.len != rows || gamma.len != d {
            return Err(err("layernorm_bwd_dev input length mismatch"));
        }
        let dx = self.alloc_dev(g.len)?;
        // The backward kernel atomically accumulates these cross-row reductions.
        let dgamma = self.alloc_dev(d)?;
        let dbeta = self.alloc_dev(d)?;
        let zero_pipeline = self.kernel_pipeline(
            "zero_pair",
            &ZERO_PAIR_ZSL,
            Some(ZERO_PAIR_HIP),
            [256, 1, 1],
        )?;
        let zero_bindings = Bindings {
            buffers: &[dgamma.buf.index(), dbeta.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(d as u32)],
        };
        self.dispatch_profiled(zero_pipeline, zero_bindings, [(d as u32 + 255) / 256, 1, 1])?;
        let pipeline = self.kernel_pipeline(
            "layernorm_bwd",
            &LAYERNORM_BWD_ZSL,
            Some(LAYERNORM_BWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[
                g.buf.index(),
                xhat.buf.index(),
                invstd.buf.index(),
                gamma.buf.index(),
                dx.buf.index(),
                dgamma.buf.index(),
                dbeta.buf.index(),
            ],
            textures: &[],
            scalars: &[Scalar::U32(rows as u32), Scalar::U32(d as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(rows as u32 + 255) / 256, 1, 1])?;
        Ok((dx, dgamma, dbeta))
    }

    fn zero_dev(&self, out: &DevTensor) -> Result<()> {
        let pipeline = self.kernel_pipeline("zero", &ZERO_ZSL, Some(ZERO_HIP), [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(out.len as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(out.len as u32 + 255) / 256, 1, 1])
    }

    pub fn zeros_dev(&self, len: usize) -> Result<DevTensor> {
        let out = self.alloc_dev(len)?;
        if let Err(error) = self.zero_dev(&out) {
            self.free_dev(out);
            return Err(error);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_dev(
        &self,
        w: &DevTensor,
        g: &DevTensor,
        m: &mut DevTensor,
        v: &mut DevTensor,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
        n: usize,
    ) -> Result<()> {
        if w.len != n || g.len != n || m.len != n || v.len != n {
            return Err(err("adam_step_dev input length mismatch"));
        }
        let pipeline = self.kernel_pipeline(
            "adam_step",
            &ADAM_STEP_ZSL,
            Some(ADAM_STEP_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[w.buf.index(), g.buf.index(), m.buf.index(), v.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::F32(lr),
                Scalar::F32(b1),
                Scalar::F32(b2),
                Scalar::F32(eps),
                Scalar::F32(wd),
                Scalar::F32(bc1),
                Scalar::F32(bc2),
                Scalar::U32(n as u32),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1])
    }

    pub fn cross_entropy_dev(
        &self,
        logits: &DevTensor,
        targets: &DevTensor,
        n: usize,
        v: usize,
    ) -> Result<(DevTensor, DevTensor)> {
        if logits.len != n * v || targets.len != n {
            return Err(err("cross_entropy_dev input length mismatch"));
        }
        let probs = self.alloc_dev(logits.len)?;
        let rowloss = self.alloc_dev(n)?;
        let pipeline =
            self.kernel_pipeline("ce_fwd", &CE_FWD_ZSL, Some(CE_FWD_HIP), [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[
                logits.buf.index(),
                targets.buf.index(),
                probs.buf.index(),
                rowloss.buf.index(),
            ],
            textures: &[],
            scalars: &[Scalar::U32(n as u32), Scalar::U32(v as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1])?;
        Ok((probs, rowloss))
    }

    pub fn cross_entropy_bwd_dev(
        &self,
        probs: &DevTensor,
        targets: &DevTensor,
        n: usize,
        v: usize,
        scale: f32,
    ) -> Result<DevTensor> {
        if probs.len != n * v || targets.len != n {
            return Err(err("cross_entropy_bwd_dev input length mismatch"));
        }
        let dlogits = self.alloc_dev(probs.len)?;
        let pipeline =
            self.kernel_pipeline("ce_bwd", &CE_BWD_ZSL, Some(CE_BWD_HIP), [256, 1, 1])?;
        let bindings = Bindings {
            buffers: &[probs.buf.index(), targets.buf.index(), dlogits.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(n as u32),
                Scalar::U32(v as u32),
                Scalar::F32(scale),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [(n as u32 + 255) / 256, 1, 1])?;
        Ok(dlogits)
    }

    pub fn embedding_dev(
        &self,
        weight: &DevTensor,
        idx: &DevTensor,
        vocab: usize,
        n: usize,
        c: usize,
    ) -> Result<DevTensor> {
        if weight.len != vocab * c || idx.len != n {
            return Err(err("embedding_dev input length mismatch"));
        }
        let out = self.alloc_dev(n * c)?;
        let pipeline = self.kernel_pipeline(
            "emb_gather",
            &EMB_GATHER_ZSL,
            Some(EMB_GATHER_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[weight.buf.index(), idx.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(n as u32), Scalar::U32(c as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [((n * c) as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn embedding_bwd_dev(
        &self,
        g: &DevTensor,
        idx: &DevTensor,
        vocab: usize,
        n: usize,
        c: usize,
    ) -> Result<DevTensor> {
        if g.len != n * c || idx.len != n {
            return Err(err("embedding_bwd_dev input length mismatch"));
        }
        let dw = self.alloc_dev(vocab * c)?;
        self.zero_dev(&dw)?;
        let pipeline = self.kernel_pipeline(
            "emb_scatter",
            &EMB_SCATTER_ZSL,
            Some(EMB_SCATTER_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[g.buf.index(), idx.buf.index(), dw.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(n as u32), Scalar::U32(c as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [((n * c) as u32 + 255) / 256, 1, 1])?;
        Ok(dw)
    }

    pub fn bias_add_dev(&self, x: &DevTensor, bias: &DevTensor, c: usize) -> Result<DevTensor> {
        if c == 0 || x.len % c != 0 || bias.len != c {
            return Err(err("bias_add_dev input length mismatch"));
        }
        let out = self.alloc_dev(x.len)?;
        let pipeline = self.kernel_pipeline(
            "bias_add",
            &BIAS_ADD_ZSL,
            Some(BIAS_ADD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), bias.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(x.len as u32), Scalar::U32(c as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(x.len as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn bias_rowsum_dev(&self, g: &DevTensor, rows: usize, c: usize) -> Result<DevTensor> {
        if g.len != rows * c {
            return Err(err("bias_rowsum_dev input length mismatch"));
        }
        let dbias = self.alloc_dev(c)?;
        let pipeline = self.kernel_pipeline(
            "bias_rowsum",
            &BIAS_ROWSUM_ZSL,
            Some(BIAS_ROWSUM_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[g.buf.index(), dbias.buf.index()],
            textures: &[],
            scalars: &[Scalar::U32(rows as u32), Scalar::U32(c as u32)],
        };
        self.dispatch_profiled(pipeline, bindings, [(c as u32 + 255) / 256, 1, 1])?;
        Ok(dbias)
    }

    pub fn slice_cols_dev(
        &self,
        x: &DevTensor,
        r: usize,
        c: usize,
        len: usize,
        start: usize,
    ) -> Result<DevTensor> {
        if x.len != r * c || start + len > c {
            return Err(err("slice_cols_dev input length mismatch"));
        }
        let out = self.alloc_dev(r * len)?;
        let pipeline = self.kernel_pipeline(
            "slice_cols",
            &SLICE_COLS_ZSL,
            Some(SLICE_COLS_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[x.buf.index(), out.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(r as u32),
                Scalar::U32(c as u32),
                Scalar::U32(len as u32),
                Scalar::U32(start as u32),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [((r * len) as u32 + 255) / 256, 1, 1])?;
        Ok(out)
    }

    pub fn slice_cols_bwd_dev(
        &self,
        g: &DevTensor,
        r: usize,
        c: usize,
        len: usize,
        start: usize,
    ) -> Result<DevTensor> {
        if g.len != r * len || start + len > c {
            return Err(err("slice_cols_bwd_dev input length mismatch"));
        }
        let dx = self.alloc_dev(r * c)?;
        self.zero_dev(&dx)?;
        let pipeline = self.kernel_pipeline(
            "slice_cols_bwd",
            &SLICE_COLS_BWD_ZSL,
            Some(SLICE_COLS_BWD_HIP),
            [256, 1, 1],
        )?;
        let bindings = Bindings {
            buffers: &[g.buf.index(), dx.buf.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(r as u32),
                Scalar::U32(c as u32),
                Scalar::U32(len as u32),
                Scalar::U32(start as u32),
            ],
        };
        self.dispatch_profiled(pipeline, bindings, [((r * len) as u32 + 255) / 256, 1, 1])?;
        Ok(dx)
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

        let pipeline = self.kernel_pipeline(name, shader, None, [256, 1, 1])?;

        let mut scalars = vec![Scalar::U32(n as u32)];
        scalars.extend_from_slice(extra_scalars);

        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars: &scalars,
        };
        let grid = [(n as u32 + 255) / 256, 1, 1];
        let start = prof().then(Instant::now);
        let result = self.device.dispatch(pipeline, bindings, grid);
        if let Some(start) = start {
            DISPATCH_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N_DISPATCH.fetch_add(1, Ordering::Relaxed);
        }
        result.map_err(|e| err(e.to_string()))?;

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

        let pipeline = self.kernel_pipeline(name, shader, None, [256, 1, 1])?;

        let mut scalars = vec![Scalar::U32(n as u32)];
        scalars.extend_from_slice(extra_scalars);

        let bindings = Bindings {
            buffers: &[ba.index(), bb.index()],
            textures: &[],
            scalars: &scalars,
        };
        let grid = [(n as u32 + 255) / 256, 1, 1];
        let start = prof().then(Instant::now);
        let result = self.device.dispatch(pipeline, bindings, grid);
        if let Some(start) = start {
            DISPATCH_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N_DISPATCH.fetch_add(1, Ordering::Relaxed);
        }
        result.map_err(|e| err(e.to_string()))?;

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

        let pipeline = self.kernel_pipeline(
            "sgemm_tiled",
            &TILED_SGEMM_ZSL,
            Some(TILED_SGEMM_HIP),
            [16, 16, 1],
        )?;

        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(m as u32),
                Scalar::U32(n as u32),
                Scalar::U32(k as u32),
            ],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, 1];
        let start = prof().then(Instant::now);
        let result = self.device.dispatch(pipeline, bindings, grid);
        if let Some(start) = start {
            DISPATCH_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N_DISPATCH.fetch_add(1, Ordering::Relaxed);
        }
        result.map_err(|e| err(e.to_string()))?;

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

        let pipeline = self.kernel_pipeline(
            "bgemm_tiled",
            &TILED_BGEMM_ZSL,
            Some(TILED_BGEMM_HIP),
            [16, 16, 1],
        )?;

        let bindings = Bindings {
            buffers: &[ba.index(), bb.index(), bc.index()],
            textures: &[],
            scalars: &[
                Scalar::U32(m as u32),
                Scalar::U32(n as u32),
                Scalar::U32(k as u32),
            ],
        };
        let grid = [(n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32];
        let start = prof().then(Instant::now);
        let result = self.device.dispatch(pipeline, bindings, grid);
        if let Some(start) = start {
            DISPATCH_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N_DISPATCH.fetch_add(1, Ordering::Relaxed);
        }
        result.map_err(|e| err(e.to_string()))?;

        let out = self
            .download(bc, batch * m * n)
            .map_err(|e| err(e.to_string()))?;

        self.recycle(ba, a.len());
        self.recycle(bb, b.len());
        self.recycle(bc, batch * m * n);
        Ok(out)
    }

    pub fn sum(&self, a: &[f32]) -> Result<f32> {
        Ok(a.iter().sum())
    }

    pub fn mean(&self, a: &[f32]) -> Result<f32> {
        if a.is_empty() {
            return Ok(0.0);
        }
        Ok(a.iter().sum::<f32>() / a.len() as f32)
    }
}

impl Drop for ZenEngine {
    fn drop(&mut self) {
        let mut destroyed = HashSet::new();
        for bucket in self.pool.lock().unwrap().drain() {
            for buf in bucket.1 {
                if destroyed.insert(buf) {
                    self.device.destroy_buffer(buf);
                }
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
                    expected[row * SIZE + col] += a[row * SIZE + inner] * b[inner * SIZE + col];
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
