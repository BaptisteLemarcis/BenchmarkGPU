#include <iostream>
#include <fstream>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <tuple>

#include "Logger.h"
#include "GenericFunctions.h"

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
		std::cerr << "CUDA Error : " << status << std::endl << __FILE__ << ":" << __LINE__ << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}

void CheckCublasError(cublasStatus_t status) {
	std::string error;
	if (status != 0) {
		std::cerr << "Cublas Error : " << status << std::endl << __FILE__ << ":" << __LINE__ << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}

void FillVec(int size, void* vec_d, float value)
{

	float* data = new float[size];
	for (int i = 0; i < size; i++) {
		data[i] = value;
	}
	CheckCudaError(cudaMemcpy(vec_d, data, size*sizeof(float), cudaMemcpyHostToDevice));
}

void printDeviceVectorToFile(int size, float* vec_d, int offset)
{
	float *vec;
	vec = new float[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, size * sizeof(float), cudaMemcpyDeviceToHost);
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "\t";
	for (int i = offset; i < size; i++)
	{
		toWrite << float(vec[i]) << " ";
	}
	Logger::instance()->writeLine(toWrite.str());
	delete[] vec;
}

void printDeviceVector(int size, float* vec_d, int offset)
{
	float *vec;
	vec = new float[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, size * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout.precision(7);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i = offset; i < size; i++)
	{
		std::cout << float(vec[i]) << " ";
	}
	std::cout << std::endl;
	delete[] vec;
}

std::tuple<float*, float**> generateData(int s)
{
	float* data = new float[s];
	float** labels = new float*[s];
	for (int i = 0; i < s/2; i++) data[i] = 0.f;
	for (int i = 0; i < s / 2; i++) {
		labels[i] = new float[2]{ 0.f, 0.f };
	}
	for (int i = s / 2; i < s; i++) data[i] = 1.f;
	for (int i = s / 2; i < s; i++) {
		labels[i] = new float[2]{ 1.f, 1.f };
	}

	return std::tuple<float*, float**>(data, labels);
}
