#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>


__global__ void vecAdd(float* a, float* b, float* c, int n);
