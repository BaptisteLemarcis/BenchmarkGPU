#ifndef __BENCHMARKGPU_GENERICFUNCTION_H
#define __BENCHMARKGPU_GENERICFUNCTION_H

void CheckCudNNError(cudnnStatus_t);

void CheckCudaError(cudaError_t);

void CheckCublasError(cublasStatus_t);

#endif // __BENCHMARKGPU_GENERICFUNCTION_H