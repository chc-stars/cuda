#include "kernel.cuh"

__global__ void vecAdd(float* a, float* b, float* c, int n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


