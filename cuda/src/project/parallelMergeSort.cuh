#include <vector>

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <math.h>

std::vector<int> runParallelMergeSort(std::vector<int> arr);


// ******************   用例  ****************


//int size = 99040;  定义数组大小
//std::vector<int> arr(size);  
//
//std::vector<int> rg = runParallelMergeSort(arr);