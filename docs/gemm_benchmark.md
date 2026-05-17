# GEMM Kernel Performance Benchmark

This document compares the performance of our Triton GEMM implementation against NumPy and vendor BLAS libraries (cuBLAS / hipBLAS) on both NVIDIA and AMD platforms.

## Test Environment

### NVIDIA
- **GPU**: H100
- **Triton**: Latest
- **cuBLAS**: Via PyTorch CUDA backend

### AMD
- **GPU**: MI325X
- **ROCm**: 7.2.3
- **Triton**: 3.6.0+rocm7.2.3
- **hipBLAS**: Via PyTorch ROCm backend

---

## Benchmark Results

### 512 × 1024 @ 1024 × 512

| Implementation     | Time (ms) | Relative to BLAS |
|--------------------|-----------|------------------|
| NumPy (H100)       | 3.06      | 102×             |
| Triton (H100)      | 1.23      | 41×              |
| cuBLAS (H100)      | 0.03      | 1×               |
| NumPy (MI325X)     | 0.25      | 8.3×             |
| Triton (MI325X)    | 1.47      | 49×              |
| hipBLAS (MI325X)   | 0.03      | 1×               |

### 1024 × 2048 @ 2048 × 1024

| Implementation     | Time (ms) | Relative to BLAS |
|--------------------|-----------|------------------|
| NumPy (H100)       | 22.81     | 207×             |
| Triton (H100)      | 2.36      | 21.5×            |
| cuBLAS (H100)      | 0.11      | 1×               |
| NumPy (MI325X)     | 1.09      | 18.2×            |
| Triton (MI325X)    | 3.34      | 55.7×            |
| hipBLAS (MI325X)   | 0.06      | 1×               |

### 2048 × 4096 @ 4096 × 2048

| Implementation     | Time (ms) | Relative to BLAS |
|--------------------|-----------|------------------|
| NumPy (H100)       | 185.48    | 281×             |
| Triton (H100)      | 9.05      | 13.7×            |
| cuBLAS (H100)      | 0.66      | 1×               |
| NumPy (MI325X)     | 6.18      | 17.2×            |
| Triton (MI325X)    | 6.83      | 19.0×            |
| hipBLAS (MI325X)   | 0.36      | 1×               |

---

## Key Observations

1. **Triton vs NumPy**
   - On H100: Triton is consistently 2–20× faster than NumPy.
   - On MI325X: Triton is slower than NumPy in most cases (especially at smaller sizes).

2. **Triton vs Vendor BLAS**
   - cuBLAS (NVIDIA) is 13–280× faster than our Triton implementation.
   - hipBLAS (AMD) is 19–55× faster than our Triton implementation.

3. **NVIDIA vs AMD**
   - Our Triton GEMM performs relatively better on NVIDIA than on AMD.
   - This suggests there is still room for optimization on the HIP side (tiling, memory access patterns, etc.).

---

## Next Steps

- Further optimize the Triton GEMM kernel (especially for AMD)
- Add more sizes and batch dimensions
- Profile with `triton.profiler` or vendor tools
- Consider using Triton’s autotune feature

---

*Benchmark generated on 2026-05-18*