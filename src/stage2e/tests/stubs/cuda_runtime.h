#ifndef STAGE2E_TEST_CUDA_RUNTIME_H
#define STAGE2E_TEST_CUDA_RUNTIME_H

// Host-only declarations for syntax-checking checkpoint code on machines that
// do not have the CUDA toolkit. This file is never on the production include path.
#include <cstddef>

// CUDA 执行空间限定符 fallback (host-only 编译时全部置空)
// types.h 已定义 __host__/__device__ fallback; 此处补充 kernel/共享内存限定符
#ifndef __global__
  #define __global__
#endif
#ifndef __shared__
  #define __shared__
#endif
#ifndef __restrict__
  #define __restrict__
#endif
#ifndef __constant__
  #define __constant__
#endif
#ifndef __managed__
  #define __managed__
#endif

// CUDA 内建向量类型 (仅 host 语法检查用, 不实际使用)
struct float2 {
    float x;
    float y;
};
struct float3 {
    float x;
    float y;
    float z;
};
struct float4 {
    float x;
    float y;
    float z;
    float w;
};
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
};

struct cudaDeviceProp {
    char name[256];
    std::size_t totalGlobalMem;
    int major;
    int minor;
};

cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceReset();
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaMemGetInfo(std::size_t*, std::size_t*);
cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaMemset(void*, int, std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);

#endif
