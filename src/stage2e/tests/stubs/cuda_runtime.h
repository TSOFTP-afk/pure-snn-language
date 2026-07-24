#ifndef STAGE2E_TEST_CUDA_RUNTIME_H
#define STAGE2E_TEST_CUDA_RUNTIME_H

// Host-only declarations for syntax-checking checkpoint code on machines that
// do not have the CUDA toolkit. This file is never on the production include path.
#include <cstddef>

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
