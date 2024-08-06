#include <cuda_runtime.h>
#include <iostream>

#define CHECK(call)                                    \
{                                                      \
    const cudaError_t error = call;                    \
    if (error != cudaSuccess) {                        \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1);                                       \
    }                                                  \
}
