// // 3. 向量归一化

// 题目3：向量归一化
// 描述: 编写一个CUDA程序，实现向量的归一化（即将向量的每个元素除以向量的长度）。

// 要求:

// 初始化一个大小为N的向量A。
// 使用CUDA内核函数计算归一化向量B，使得B[i] = A[i] / ||A||（其中||A||是向量A的欧几里得范数）。
// 打印输出结果向量B。

#include <cuda_runtime.h>
#include "src/common/error.cuh"
#include <iostream>
#include <cmath>
#include <device_launch_parameters.h>



__global__ void vecNorm(const float* a, float* b, int n) {
    __shared__ float sum[256]; // 共享内存数组
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // 计算每个元素的平方
    sum[tid] = (idx < n) ? a[idx] * a[idx] : 0.0f;

    __syncthreads();

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }

    // 将归约结果写入全局内存
    if (tid == 0) {
        atomicAdd(&b[0], sum[0]);
    }
}

__global__ void normalize(float* a, float* b, float norm, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        b[idx] = a[idx] / norm;
    }
}

int runVecNormalization(float a[], float b[],  int N) {
 
    float h_sum = 0.0f;
  

    float* d_a, * d_b;

    // 申请内存
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));

    // copy数据
    CHECK(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, &h_sum, sizeof(float), cudaMemcpyHostToDevice));

    // 定义block
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动内核计算向量大小
    vecNorm << <gridSize, blockSize >> > (d_a, d_b, N);
    CHECK(cudaDeviceSynchronize());

    // 将平方和从设备复制到主机
    CHECK(cudaMemcpy(&h_sum, d_b, sizeof(float), cudaMemcpyDeviceToHost));

    // 计算向量的L2范数
    float norm = sqrt(h_sum);

    // 归一化向量
    normalize << <gridSize, blockSize >> > (d_a, d_a, norm, N);
    CHECK(cudaDeviceSynchronize());

    // 将结果从设备复制到主机
    CHECK(cudaMemcpy(b, d_a, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    // 打印结果
    for (int i = 0; i < N; i++) {
        printf("b[%d] = %f\n", i, b[i]);
    }

    return 0;
}
