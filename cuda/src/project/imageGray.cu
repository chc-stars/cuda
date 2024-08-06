
// 题目4：图像灰度化
// 描述: 编写一个CUDA程序，将一幅RGB图像转换为灰度图像。

// 要求:

// 初始化一个MxNx3的RGB图像。
// 使用CUDA内核函数计算灰度图像，灰度值计算公式为Gray = 0.299R + 0.587G + 0.114*B。
// 打印输出结果灰度图像。
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "device_launch_parameters.h"

// Time to run on CPU: 5381.61 ms
// Gray Image (CPU):
// Time to run kernel: 7.72288 ms
// Gray Image (GPU):

#define M 2600 // 图像高度
#define N 2600 // 图像宽度

__global__ void rgb2gray(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        unsigned char r = rgb[3 * idx];
        unsigned char g = rgb[3 * idx + 1];
        unsigned char b = rgb[3 * idx + 2];
        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void initializeImage(unsigned char* image, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
        image[i] = rand() % 256; // 随机初始化图像数据
    }
}

void printImage(unsigned char* image, int width, int height, int channels) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                printf("%d ", image[(i * width + j) * channels + k]);
            }
            printf(" | ");
        }
        printf("\n");
    }
}

void rgb2grayCPU(unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            unsigned char r = rgb[3 * idx];
            unsigned char g = rgb[3 * idx + 1];
            unsigned char b = rgb[3 * idx + 2];
            gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

int runImgGray() {
    int width = N;
    int height = M;
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    // 分配主机内存
    unsigned char* h_rgb = (unsigned char*)malloc(rgb_size);
    unsigned char* h_gray = (unsigned char*)malloc(gray_size);
    unsigned char* h_gray_cpu = (unsigned char*)malloc(gray_size);

    // 初始化RGB图像
    initializeImage(h_rgb, width, height);

    // 打印RGB图像
    std::cout << "RGB Image:\n";
    // printImage(h_rgb, width, height, 3);

    // CPU 计算灰度图像并测量时间
    auto start_cpu = std::chrono::high_resolution_clock::now();
    rgb2grayCPU(h_rgb, h_gray_cpu, width, height);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = stop_cpu - start_cpu;
    std::cout << "Time to run on CPU: " << duration_cpu.count() << " ms" << std::endl;

    // 打印灰度图像（CPU 计算结果）
    std::cout << "Gray Image (CPU):\n";
    // printImage(h_gray_cpu, width, height, 1);

    // 分配设备内存
    unsigned char* d_rgb, * d_gray;
    cudaMalloc((void**)&d_rgb, rgb_size);
    cudaMalloc((void**)&d_gray, gray_size);

    // 拷贝数据到设备
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);

    // 定义块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录起始事件
    cudaEventRecord(start, 0);

    // 启动CUDA内核
    rgb2gray << <gridSize, blockSize >> > (d_rgb, d_gray, width, height);

    // 记录结束事件
    cudaEventRecord(stop, 0);

    // 同步事件
    cudaEventSynchronize(stop);

    // 计算时间差
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time to run kernel: " << elapsedTime << " ms" << std::endl;

    // 拷贝灰度图像数据回主机
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);

    // 打印灰度图像
    std::cout << "Gray Image (GPU):\n";
    // printImage(h_gray, width, height, 1);

    // 释放CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放设备内存
    cudaFree(d_rgb);
    cudaFree(d_gray);

    // 释放主机内存
    free(h_rgb);
    free(h_gray);
    free(h_gray_cpu);

    return 0;
}
