  // 8、求解线性方程组
  // 题目8：求解线性方程组（Ax = b）
  // 描述: 编写一个CUDA程序，求解线性方程组Ax = b。

  // 要求:

  // 初始化一个大小为NxN的矩阵A和一个大小为N的向量b。
  // 使用CUDA内核函数求解方程组，得到向量x。
  // 打印输出结果向量x。



#include "solveLinearEquations.cuh"



#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
            exit(-1); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status << std::endl; \
            exit(-1); \
        } \
    } while (0)

void solveLinearSystem(const std::vector<float>& A, const std::vector<float>& b, std::vector<float>& x, int n) {
    float* d_A, * d_b;
    int* d_pivot, * d_info;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_pivot, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int lda = n;

    // Perform LU factorization
    float* Aarray[] = { d_A };
    CHECK_CUBLAS(cublasSgetrfBatched(handle, n, Aarray, lda, d_pivot, d_info, 1));

    // Solve Ax = b using the LU factorization
    float* Barray[] = { d_b };
    CHECK_CUBLAS(cublasSgetrsBatched(handle, CUBLAS_OP_N, n, 1, (const float**)Aarray, lda, d_pivot, Barray, lda, d_info, 1));

    // Copy the result back to host
    CHECK_CUDA(cudaMemcpy(x.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_pivot));
    CHECK_CUDA(cudaFree(d_info));

    CHECK_CUBLAS(cublasDestroy(handle));
}

std::vector<float> runSolveLinearEquations(std::vector<float> A_, std::vector<float> b_, int n) {
  
 /*   const int n = 3;*/

    std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 10.0f
    };

    std::vector<float> b = { 6.0f, 15.0f, 25.0f };

    std::vector<float> x(n, 0.0f);

    solveLinearSystem(A, b, x, n);

    std::cout << "Solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;


    return x;
}
