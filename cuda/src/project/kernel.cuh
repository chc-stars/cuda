#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>


__global__ void vecAdd(float* a, float* b, float* c, int n);

__global__ void rgb2gray(unsigned char* rgb, unsigned char* gray, int width, int height);