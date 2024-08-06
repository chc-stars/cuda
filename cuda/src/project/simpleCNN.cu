// ���򻯰����

// ��Ŀ7���򻯰��������
// ����: ��дһ��CUDA����ʵ��һ���򻯰��2D����������

// Ҫ��:

// ��ʼ��һ��MxN����������һ��С��KxK�����˾���
// ʹ��CUDA�ں˺����������������
// ��ӡ����������


#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simpleCNN.cuh"


#define M 5  // ������������
#define N 5  // ������������
#define K 3  // �����˾���Ĵ�С

__global__ void convolution2D(float* input, float* kernel, float* output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int kernelRadius = k / 2;

    if (row < m && col < n) {
        float value = 0.0f;
        for (int i = -kernelRadius; i <= kernelRadius; ++i) {
            for (int j = -kernelRadius; j <= kernelRadius; ++j) {
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < m && curCol >= 0 && curCol < n) {
                    value += input[curRow * n + curCol] * kernel[(i + kernelRadius) * k + (j + kernelRadius)];
                }
            }
        }
        output[row * n + col] = value;
    }
}

void printMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int run() {
    std::vector<float> input = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    std::vector<float> kernel = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    std::vector<float> output(M * N, 0);

    float* d_input, * d_kernel, * d_output;

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_kernel, K * K * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemcpy(d_input, input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), K * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    convolution2D << <gridSize, blockSize >> > (d_input, d_kernel, d_output, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Input Matrix:" << std::endl;
    printMatrix(input, M, N);

    std::cout << "Kernel Matrix:" << std::endl;
    printMatrix(kernel, K, K);

    std::cout << "Output Matrix:" << std::endl;
    printMatrix(output, M, N);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}