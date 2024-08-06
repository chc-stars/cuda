// cuda.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <chrono>

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "src/project/vecAdd.h"
#include "src/project/simpleCNN.cuh"
#include "src/project/imageGray.cuh"
#include "src/project/matrixMulti.cuh"
#include "src/project/parallelMergeSort.cuh"
#include "src/project//parallelPrefixSum.cuh"
#include "src/project/solveLinearEquations.cuh"
#include "src/project/vecNormalization.cuh"


int main()
{
  
    // 获取当前时间点
    auto start = std::chrono::high_resolution_clock::now();

    // ---------------  run ------------------
    const int N = 3;
    float a[N] = { 3, 2, 1 };
    float b[N] = { 0 };

    runVecNormalization(a, b,N);
    // ---------------  run ------------------
    // 获取当前时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}


