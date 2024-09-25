#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "device_launch_parameters.h"

#include "src/project/physicalSimulation.cuh"



//#define N  1024 // grid Size
#define BLOCK_SIZE 256  

__global__ void updateGrid(float *grid, float *newGrid, int n, float dt, float dx) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > 0 && idx < n - 1) {
		float left = grid[idx - 1];
		float right = grid[idx + 1];
		float center = grid[idx];

		newGrid[idx] = center + dt * (left - 2 * center + right) / (dx * dx);

	}
}


void runPhysicalSimulation(float *h_grid,  int N) {

	//float *h_grid = new float[N];
	float *h_newGrid = new float[N];
	float *d_grid, *d_newGrid;

	// init grid
	//for (int i = 0; i < N; ++i) {
	//	h_grid[i] = 0.0f;
	//}
	//h_grid[N / 2] = 1.0f;   // 初始条件

	// 
	cudaMalloc((void**)&d_grid, N * sizeof(float));
	cudaMalloc((void**)&d_newGrid, N * sizeof(float));

	// 将数据从主机传输到设备
	cudaMemcpy(d_grid, h_grid, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_newGrid, h_newGrid, N * sizeof(float), cudaMemcpyHostToDevice);

	// 定义网络和块大小
	dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE);

	// 时间步长和空间步长
	float dt = 0.01f;
	float dx = 1.0f;

	// 主循环
	for (int step = 0; step < 1000; ++step) {
		updateGrid << <blocks, threads >> > (d_grid, d_newGrid, N, dt, dx);
		// 交换指针
		float* temp = d_grid;
		d_grid = d_newGrid;
		d_newGrid = temp;

	}

	// 将结果设备传到主机
	cudaMemcpy(h_grid, d_grid, N * sizeof(float), cudaMemcpyDeviceToHost);

	// 释放设备内存
	cudaFree(d_grid);
	cudaFree(d_newGrid);


	for (int i = 0; i < N; ++i) {
		std::cout << "Grid[" << i << "] = " << h_grid[i] << std::endl;
	}

	delete[] h_grid;
	delete[] h_newGrid;

}