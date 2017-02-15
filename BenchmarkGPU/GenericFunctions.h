#ifndef __BENCHMARKGPU_GENERICFUNCTION_H
#define __BENCHMARKGPU_GENERICFUNCTION_H

#include <cudnn.h>
#include <cublas_v2.h>
#include <random>

void CheckError(cudnnStatus_t, char*, int);

void CheckError(cudaError_t, char*, int);

void CheckError(cublasStatus_t, char*, int);

/*void CheckCudNNError(cudnnStatus_t);

void CheckCudaError(cudaError_t);

void CheckCublasError(cublasStatus_t);*/

void printVectorToFile(int, float*, int);

void printDeviceVectorToFile(int, float*, int);

void printDeviceVector(int, float*, int);

//std::tuple<float*, float**> generateData(int);

int RoundUp(int, int);

#endif // __BENCHMARKGPU_GENERICFUNCTION_H