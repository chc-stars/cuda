//// 9 直方图均衡化
//
//// 题目9：直方图均衡化
//// 描述: 编写一个CUDA程序，实现图像的直方图均衡化。
//
//// 要求:
//
//// 初始化一个大小为MxN的灰度图像。
//// 使用CUDA内核函数计算直方图，并进行均衡化处理。
//// 打印输出均衡化后的图像。
//
//#include <iostream>
//#include <vector>
//#include <chrono>
//#include <algorithm>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <device_launch_parameters.h>
//#include "src/project/histogramEqualization.cuh"
//#include <device_atomic_functions.hpp>
//
//const int M = 1200;
//const int N = 1200;
//const int NUM_BINS = 256;
//
//__global__ void compute_histogram(const unsigned char* image, int* histogram, int width, int height) {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//
//    while (tid < width * height) {
//        atomicAdd(&histogram[image[tid]], 1);
//        tid += stride;
//    }
//}
//
//__global__ void histogram_equalization(unsigned char* image, const int* histogram, int width, int height) {
//    __shared__ float cdf[NUM_BINS];
//
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    if (tid < NUM_BINS) {
//        int cum_sum = 0;
//        for (int i = 0; i <= tid; ++i) {
//            cum_sum += histogram[i];
//        }
//        cdf[tid] = (cum_sum - histogram[0]) / (float)(width * height - histogram[0]);
//    }
//
//    __syncthreads();
//
//    int stride = blockDim.x * gridDim.x;
//    while (tid < width * height) {
//        image[tid] = min(max(int(cdf[image[tid]] * (NUM_BINS - 1)), 0), 255);
//        tid += stride;
//    }
//}
//
//void print_image(const unsigned char* image, int width, int height) {
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            std::cout << static_cast<int>(image[i * width + j]) << " ";
//        }
//        std::cout << std::endl;
//    }
//}
//
//void histogram_equalization_cpu(unsigned char* image, int width, int height) {
//    int histogram[NUM_BINS] = { 0 };
//    int size = width * height;
//
//    // Calculate histogram
//    for (int i = 0; i < size; ++i) {
//        histogram[image[i]]++;
//    }
//
//    // Calculate cumulative distribution function (CDF)
//    float cdf[NUM_BINS] = { 0 };
//    int cum_sum = 0;
//    for (int i = 0; i < NUM_BINS; ++i) {
//        cum_sum += histogram[i];
//        cdf[i] = (cum_sum - histogram[0]) / (float)(size - histogram[0]);
//    }
//
//    // Equalize the image
//    for (int i = 0; i < size; ++i) {
//        image[i] = std::min(std::max(int(cdf[image[i]] * (NUM_BINS - 1)), 0), 255);
//    }
//}
//
//int runHistogramEqualization() {
//    unsigned char h_image[M * N];
//    unsigned char h_equalized_image[M * N];
//
//    // Initialize the image with random values
//    for (int i = 0; i < M * N; ++i) {
//        h_image[i] = rand() % 256;
//        h_equalized_image[i] = h_image[i];
//    }
//
//    // CUDA implementation
//    unsigned char* d_image;
//    int* d_histogram;
//
//    cudaMalloc((void**)&d_image, M * N * sizeof(unsigned char));
//    cudaMalloc((void**)&d_histogram, NUM_BINS * sizeof(int));
//
//    cudaMemcpy(d_image, h_image, M * N * sizeof(unsigned char), cudaMemcpyHostToDevice);
//    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));
//
//    int blockSize = 256;
//    int numBlocks = (M * N + blockSize - 1) / blockSize;
//
//    auto cuda_start = std::chrono::high_resolution_clock::now();
//
//    compute_histogram << <numBlocks, blockSize >> > (d_image, d_histogram, M, N);
//    cudaDeviceSynchronize();
//
//    histogram_equalization << <numBlocks, blockSize >> > (d_image, d_histogram, M, N);
//    cudaDeviceSynchronize();
//
//    auto cuda_end = std::chrono::high_resolution_clock::now();
//
//    cudaMemcpy(h_equalized_image, d_image, M * N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//    cudaFree(d_image);
//    cudaFree(d_histogram);
//
//    std::chrono::duration<double> cuda_duration = cuda_end - cuda_start;
//
//    // Print the CUDA equalized image
//    std::cout << "CUDA Equalized Image:" << std::endl;
//    print_image(h_equalized_image, M, N);
//
//    // C++ implementation
//    std::copy(h_image, h_image + M * N, h_equalized_image);
//
//    auto cpp_start = std::chrono::high_resolution_clock::now();
//
//    histogram_equalization_cpu(h_equalized_image, M, N);
//
//    auto cpp_end = std::chrono::high_resolution_clock::now();
//
//    std::chrono::duration<double> cpp_duration = cpp_end - cpp_start;
//
//    // Print the C++ equalized image
//    std::cout << "C++ Equalized Image:" << std::endl;
//    print_image(h_equalized_image, M, N);
//
//    // Output the timing results
//    std::cout << "CUDA Duration: " << cuda_duration.count() << " seconds" << std::endl;
//    std::cout << "C++ Duration: " << cpp_duration.count() << " seconds" << std::endl;
//
//    return 0;
//}
