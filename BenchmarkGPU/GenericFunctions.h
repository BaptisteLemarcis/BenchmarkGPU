#ifndef __BENCHMARKGPU_GENERICFUNCTION_H
#define __BENCHMARKGPU_GENERICFUNCTION_H

#include <cudnn.h>
#include <cublas_v2.h>

void CheckCudNNError(cudnnStatus_t);

void CheckCudaError(cudaError_t);

void CheckCublasError(cublasStatus_t);

void FillVec(int, void*, float);

void printDeviceVectorToFile(int, float*, int);

void printDeviceVector(int, float*, int);

std::tuple<float*, float**> generateData(int);

#endif // __BENCHMARKGPU_GENERICFUNCTION_H