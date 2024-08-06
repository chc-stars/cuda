#include "kernel.cuh"

__global__ void vecAdd(float* a, float* b, float* c, int n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


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