// 2、矩阵乘法

// 题目2：矩阵乘法
// 描述: 编写一个CUDA程序，实现两个矩阵的乘法。

// 要求:

// 初始化两个大小为MxN和NxP的矩阵A和B。
// 使用CUDA内核函数计算矩阵乘法C = A * B。
// 打印输出结果矩阵C。

#include "src/common/error.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>



__global__ void matrixMulti(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}


int runMatrixMulti(float a[], float b[], float c[], size_t N) {


    // 定义设备端向量指针
    float* d_a, * d_b, * d_c;

    // 分配设备端内存
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // 数据拷贝到设备端
    CHECK(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    // 定义线程块和网格的大小
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动CUDA内核
    matrixMulti << <gridSize, blockSize >> > (d_a, d_b, d_c, N);

    // 同步设备端
    CHECK(cudaDeviceSynchronize());

    // 将结果从设备端复制到主机端
    CHECK(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备端内存
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    // 打印结果
    for (int i = 0; i < N; ++i) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;

}

