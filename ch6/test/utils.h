//
// Created by moguw on 24-12-7.
//

#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

void log_test();

void checkCudaError(cudaError_t err, const char *action);

void generateRandomArray(int *&a, int n);
#endif //UTILS_H
