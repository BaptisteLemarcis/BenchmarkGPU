#pragma once
#include <iostream>
#include <string.h>

void CheckCudNNError(cudnnStatus_t status) {
	std::string error;
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "CudNN Error : " << cudnnGetErrorString(status) << std::endl << __FILE__ << ":" << __LINE__ << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}

void CheckCudaError(cudaError_t status) {
	std::string error;
	if (status != 0) {
		std::cerr << "Cuda Error : " << status << std::endl << __FILE__ << ":" << __LINE__ << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}