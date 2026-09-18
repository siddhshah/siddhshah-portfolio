---
type: ProjectLayout
title: GPU-Accelerated CNN Inference
colors: colors-a
date: '2026-05-23'
client: ''
description: >-
  Fused CUDA C++ convolution kernel for a LeNet-5 forward pass, using 16x16x16
  WMMA Tensor Cores to cut inference time by 36% on an NVIDIA A40.
featuredImage:
  type: ImageBlock
  url: /images/cuda-thumb.png
  altText: Nsight Compute pipe utilization chart for the WMMA kernel
media:
  type: ImageBlock
  url: /images/cuda-nsight-sol-wmma.png
  altText: >-
    Nsight Compute Speed of Light page for the fused WMMA kernel on an NVIDIA
    A40, showing compute throughput at 73.8% against 17.8% memory throughput
---
This project is a hand-written CUDA C++ inference kernel for the forward-pass convolution of a modified LeNet-5, classifying 10,000 Fashion-MNIST images on an NVIDIA A40 GPU cluster. It was the final project for ECE 408: Applied Parallel Programming at UIUC.

The goal was not to make a convolution work — it was to make one *fast*, and to understand exactly where the time goes. Every optimization below was measured with NVIDIA Nsight Compute and Nsight Systems rather than assumed, and the final kernel reaches **63.95 ms of total Op Time at 0.8714 reference accuracy**, down from a 100.58 ms tiled GEMM baseline — a 36% reduction.

### Part 1: Convolution as an implicit unroll

A convolution can be rewritten as a matrix multiplication: flatten each `K x K` receptive field into a column, and the whole layer becomes `C = A x B`, where `A` is the mask reshaped to `(Map_out, Channel*K*K)` and `B` is the input unrolled to `(Channel*K*K, Batch*H_out*W_out)`.

The naive way to do this is to materialize the unrolled matrix in global memory. That costs an extra `K*K`-fold blowup in memory traffic before a single multiply happens. Instead, the kernel *fuses* the three stages — unroll, matmul, and the output permutation back into `(b, m, h, w)` order — so the unrolled matrix never exists anywhere but in shared memory tiles:

```cuda
for (int t = 0; t < (K_unrolled + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
    const int k_mask  = t * TILE_WIDTH + tx;
    const int k_input = t * TILE_WIDTH + ty;

    // load mask tile from A
    if (m < Map_out && k_mask < K_unrolled)
        tile_mask[ty][tx] = mask[(size_t)(m) * K_unrolled + k_mask];
    else tile_mask[ty][tx] = 0.0f;

    // load tile of B by unrolling the input on the fly
    if (col < W_unrolled && k_input < K_unrolled) {
        const int c   = k_input / (K * K);
        const int rem = k_input % (K * K);
        const int p   = rem / K;
        const int q   = rem % K;

        const int b       = (int)(col / image_size);
        const int spatial = (int)(col % image_size);
        const int h_out   = spatial / Width_out;
        const int w_out   = spatial % Width_out;

        tile_input[ty][tx] = in_4d(b, c, h_out + p, w_out + q);
    } else tile_input[ty][tx] = 0.0f;

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
        acc += tile_mask[ty][k] * tile_input[k][tx];
    __syncthreads();
}
```

Each thread block computes one 16x16 tile of the output; the index arithmetic in the inner load is what turns a global-memory gather into an implicit unroll. This fused kernel became the baseline that everything else is measured against.

### Part 2: Moving the inner product onto Tensor Cores

The tiled version still spends its time in the CUDA cores doing scalar fused multiply-adds. The A40 (sm_86) can do far better: its Tensor Cores execute a full 16x16x16 matrix multiply-accumulate per warp instruction through the WMMA API.

Rewriting the inner loop around `wmma::fragment` means each warp owns one 16x16 sub-tile, loads FP16 operands out of shared memory, and accumulates in FP32 — the mixed precision is native to the fragment type, so there is no separate conversion pass:

```cuda
__shared__ __half As[BLOCK_M][WMMA_K];
__shared__ __half Bs[WMMA_K][BLOCK_N];
__shared__ float  Cs[BLOCK_M][BLOCK_N];

wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);

for (int kt = 0; kt < num_k_tiles; ++kt) {
    // ... cooperative loads of As and Bs (implicit unroll as above) ...
    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[warp_m * WMMA_M][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &Bs[0][warp_n * WMMA_N], BLOCK_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
}
```

This alone took total Op Time from 100.58 ms to 70.46 ms — a 30% improvement, and the single largest win in the project. The Nsight Compute Compute Workload Analysis page confirms *why*: the baseline shows ALU pipeline utilization at 53.2% with the Tensor row completely idle, while the WMMA kernel shows ALU at 67.1% with real Tensor pipeline activity. The Speed of Light page shifts from "balanced" to compute-heavy. The work moved onto the hardware it was supposed to move onto.

![Nsight Compute Workload Analysis for the FP32 baseline kernel](/images/cuda-nsight-cwa-baseline.png "FP32 baseline on CUDA cores: ALU is the busiest pipeline at 53.2%, and all three Tensor rows are empty.")

![Nsight Compute Workload Analysis for the WMMA kernel](/images/cuda-nsight-cwa-wmma.png "WMMA kernel: ALU rises to 67.1% and the Tensor All and Tensor FP rows now show activity — the matmul is running on Tensor Cores.")

Stacked on top were smaller changes: the convolution mask in `__constant__` memory (read by every block, never written), `__restrict__` on all global pointers, `#pragma unroll` on the cooperative load loops, and block tile dimensions left as `-D` sweepable parameters so tiling could be tuned rather than guessed.

### Part 3: Per-layer kernel dispatch

Tensor Cores are rigid in a way that scalar code is not: a 16x16x16 fragment computes 16 output rows whether or not the layer has 16 output feature maps. LeNet's first convolution has only 4, so 12 of every 16 rows of the fragment were computing padding — 75% of the Tensor Core throughput thrown away.

The fix is to pick the fragment *shape* per layer. The `m8n32k16` variant computes 8 rows instead of 16, which for `Map_out = 4` wastes half the M dimension rather than three quarters, doubling useful fragment utilization on Conv1 to 50%:

```cuda
#ifndef SMALL_M_THRESHOLD
#define SMALL_M_THRESHOLD 8
#endif

if (Map_out < SMALL_M_THRESHOLD) {
    dim3 block_s(THREADS_PER_BLOCK_S, 1, 1);
    dim3 grid_s((unsigned int)((N_total + BLOCK_N_S - 1) / BLOCK_N_S),
                (unsigned int)((Map_out + BLOCK_M_S - 1) / BLOCK_M_S), 1);
    fused_wmma_kernel_8x32<<<grid_s, block_s>>>(device_input, device_output,
                                               Batch, Map_out, Channel, Height, Width, K);
} else {
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((N_total + BLOCK_N - 1) / BLOCK_N),
              (unsigned int)((Map_out + BLOCK_M - 1) / BLOCK_M), 1);
    fused_wmma_kernel_16x16<<<grid, block>>>(device_input, device_output,
                                            Batch, Map_out, Channel, Height, Width, K);
}
```

A threshold of 8 routes `Map_out = 4` onto the narrow path and `Map_out = 16` onto the wide one, dropping Conv1 from 39.40 ms to 37.21 ms and putting both layers under 40 ms per-layer Op Time.

The same "shape has to match the layer" effect shows up in the plain tiled baseline, which is what suggested the dispatch in the first place. Sweeping `TILE_WIDTH` at batch 10000:

| TILE_WIDTH | Conv1 | Conv2 | Total |
| --- | --- | --- | --- |
| 8 | 45.30 ms | 53.64 ms | 98.94 ms |
| 16 | 52.75 ms | 31.36 ms | **84.11 ms** |
| 24 | 98.56 ms | 51.58 ms | 150.14 ms |
| 32 | 132.79 ms | 80.27 ms | 213.06 ms |

Conv1 is *faster* at TW=8 while Conv2 is faster at TW=16 — the same M-dimension waste argument, one level down. TW=24 breaks because it is not a multiple of the 32-thread warp; TW=32 hits the 1024-threads-per-block limit and destroys occupancy.

### A negative result worth keeping

Not everything stacked. Applying FP16 with `__half2` SIMD *by itself* to the tiled kernel — packing two halves into a register and multiplying both with `__hmul2` — landed at 93.76 ms, roughly 10 ms **slower** than the same kernel without it.

The explanation is in the instruction mix: by the time the inner-product loop runs, every operand already lives in shared memory at register latency, so the kernel is bound by FMA throughput rather than bandwidth. The per-element `F2F` conversions and the `__low2float`/`__high2float` extraction needed to accumulate in FP32 occupy exactly the issue slots that would otherwise be issuing FMAs. The SIMD multiply saves less than the conversion costs.

This is not an argument against FP16 — it is an argument about *where* the conversion happens. Inside a WMMA fragment the operands stay FP16 end to end and the Tensor Core consumes them natively, which is why the same precision choice is a 30% win in `req_1` and a 12% loss here.

Constant memory told a similar story: applied naively to the baseline matmul it was a wash (99.12 ms), because `k_mask` varies across the warp so every thread reads a *different* address and the constant cache serializes them. It only pays inside the WMMA loader, where the access pattern is genuinely broadcast-like.

### Results

At batch size 10,000:

| Variant | Conv1 | Conv2 | Total Op Time |
| --- | --- | --- | --- |
| M2 fused baseline (FP32, CUDA cores) | 58.23 ms | 42.35 ms | 100.58 ms |
| + WMMA Tensor Cores | 42.95 ms | 27.51 ms | 70.46 ms |
| + tile sweep (BLOCK_N_TILES=4) | 39.40 ms | 26.67 ms | 66.07 ms |
| + per-layer fragment dispatch | **37.21 ms** | **26.73 ms** | **63.95 ms** |

Accuracy held at exactly 0.8714 across all 10,000 images at every step, matching the reference implementation — worth stating explicitly, because the interesting failure mode of an FP16 rewrite is a kernel that is fast and quietly wrong. The FP32 accumulator inside the fragment is what buys that.

### For code:

<https://github.com/siddhshah/CUDA-CNN-Kernel>
