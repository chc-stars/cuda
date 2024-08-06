
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> runSolveLinearEquations(std::vector<float> A_, std::vector<float> b_, int n);

// ******************   ÓÃÀý  ****************
//const int n = 3;
//
//std::vector<float> A = {
//    1.0f, 2.0f, 3.0f,
//    4.0f, 5.0f, 6.0f,
//    7.0f, 8.0f, 10.0f
//};
//
//std::vector<float> b = { 6.0f, 15.0f, 25.0f };