// 10.并行前缀和(扫描)

// 题目10：并行前缀和（扫描）
// 描述: 编写一个CUDA程序，实现数组的并行前缀和（扫描）计算。

// 要求:

// 初始化一个大小为N的数组A。
// 使用CUDA内核函数计算数组的前缀和数组B，使得B[i] = A[0] + A[1] + ... + A[i]。
// 打印输出结果数组B。

#include "src/common/error.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void scan_up_sweep(int* d_out, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int k = idx * stride * 2;

    if (k + stride < N) {
        d_out[k + stride * 2 - 1] += d_out[k + stride - 1];
    }
}


__global__ void scan_down_sweep(int* d_out, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int k = idx * stride * 2;
    if (k + stride < N) {
        int temp = d_out[k + stride - 1];
        d_out[k + stride - 1] = d_out[k + stride * 2 - 1];
        d_out[k + stride * 2 - 1] += temp;

    }

}

void prefix_sum(int* h_out, const int* h_in, int N) {

    int* d_out;
    size_t size = N * sizeof(int);
    CHECK(cudaMalloc((void**)&d_out, size));
    CHECK(cudaMemcpy(d_out, h_in, size, cudaMemcpyHostToDevice));

    int threads = 512;
    int blocks = (N + threads * 1) / threads;

    // 向上扫描
    for (int stride = 1; stride < N; stride *= 2) {
        scan_up_sweep << <blocks, threads >> > (d_out, N, stride);

    }

    cudaMemset(&d_out[N - 1], 0, sizeof(int));

    // 向下扫描
    for (int stride = N / 2; stride >= 1; stride /= 2) {
        scan_down_sweep << <blocks, threads >> > (d_out, N, stride);
    }

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}

int runParallelPrefixSum(int h_in[], int h_out[], int N) {

  

    // 计算前缀和
    prefix_sum(h_out, h_in, N);

    // 打印输出结果
    std::cout << "Input: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_in[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;

}