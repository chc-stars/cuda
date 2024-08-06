// 5、并行归并排序

// 题目5：并行归并排序
// 描述: 编写一个CUDA程序，实现并行归并排序算法。

// 要求:

// 初始化一个大小为N的无序数组A。
// 使用CUDA内核函数并行地对数组进行归并排序。
// 打印输出排序后的数组。

#include "src/common/error.cuh"

#include <device_launch_parameters.h>
#include "src/project/parallelMergeSort.cuh"


__device__ void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

__global__ void mergeSortKernel(int* arr, int* temp, int width, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * width * 2;
    int mid = min(left + width - 1, size - 1);
    int right = min(left + 2 * width - 1, size - 1);

    if (left < size && mid < size) {
        merge(arr, temp, left, mid, right);
    }
}

void mergeSort(int* arr, int size) {
    int* d_arr, * d_temp;
    size_t bytes = size * sizeof(int);

    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    for (int width = 1; width < size; width *= 2) {
        mergeSortKernel << <gridSize, blockSize >> > (d_arr, d_temp, width, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);
}

void initializeArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 * 5;
    }
}

void printArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<int> runParallelMergeSort(std::vector<int> arr) {

    int size = arr.size();
    initializeArray(arr.data(), size);

    std::cout << "Unsorted array:" << std::endl;
    printArray(arr.data(), size);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    mergeSort(arr.data(), size);

    cudaEventRecord(end, 0);

    // 同步事件
    cudaEventSynchronize(end);


    // 计算时间差
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "Time to run kernel: " << elapsedTime << " ms" << std::endl;

    std::cout << "Sorted array:" << std::endl;

    printArray(arr.data(), size);

    return arr;
}



