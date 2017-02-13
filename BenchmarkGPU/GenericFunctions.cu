#include <iostream>
#include <fstream>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <tuple>
#include <device_launch_parameters.h>
#include <random>
#include <curand_kernel.h>
#include <ctime>

#include "Logger.h"
#include "GenericFunctions.h"

void CheckError(cudnnStatus_t status, char* file, int line) {
	std::string error;
	if (status != CUDNN_STATUS_SUCCESS) {
		std::cerr << "CudNN Error " << status << ": " << cudnnGetErrorString(status) << std::endl << file << ":" << line << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}
void CheckError(cudaError_t status, char* file, int line) {
	std::string error;
	if (status != 0) {
		std::cerr << "CUDA Error " << status << ": " << cudaGetErrorString(status) << std::endl << file << ":" << line << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}
void CheckError(cublasStatus_t status, char* file, int line){
	std::string error;
	if (status != 0) {
		std::cerr << "Cublas Error : " << status << std::endl << file << ":" << line << std::endl;
		cudaDeviceReset();
		exit(1);
	}
}

__global__ void FillVecKernel(float* d_vec, float value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx <= size) return;
	d_vec[idx] = value;
}

__global__ void GenerateDataKernel(float* d_data, float* d_target) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idxTarget1 = 2 * idx;
	int idxTarget2 = (2 * (idx + 1)) - 1;

	if (blockIdx.x == 0) {
		d_data[idx] = 0.f;
		d_target[idxTarget1] = 1.f;
		d_target[idxTarget2] = 0.f;
	}
	else
	{
		d_data[idx] = 1.f;
		d_target[idxTarget1] = 0.f;
		d_target[idxTarget2] = 1.f;
	}

}

__global__ void InitDataDistributedKernel(float min, float max, float* d_vec, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;

	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);

	float value = curand_uniform(&state);
	float oldRange = (1.f - 0.f);
	float newRange = (max - (min));
	float newValue = (((value - 0.f) * newRange) / oldRange) + (min);

	d_vec[idx] = newValue;
}

void initDataDistributed(float min, float max, float* d_vec, int size) {
	InitDataDistributedKernel <<< RoundUp(size, 128), 128 >>> (min, max, d_vec, size);
}

void FillVec(int size, float* vec_d, float value)
{
	FillVecKernel <<<RoundUp(size, 128), 128 >>> (vec_d, value, size);
	/*float* data = new float[size];
	for (int i = 0; i < size; i++) {
		data[i] = value;
	}
	CheckError(cudaMemcpy(vec_d, data, size*sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);*/
}

void printDeviceVectorToFile(int size, float* vec_d, int offset)
{
	float *vec;
	vec = new float[size];
	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	CheckError(cudaMemcpy(vec, vec_d, size * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "\t";
	for (int i = offset; i < size; i++)
	{
		toWrite << float(vec[i]) << " ";
	}
	Logger::instance()->writeLine(toWrite.str());
	delete[] vec;
}

void printVectorToFile(int size, float* vec, int offset)
{
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "\t";
	for (int i = offset; i < size; i++)
	{
		toWrite << float(vec[i]) << " ";
	}
	Logger::instance()->writeLine(toWrite.str());
}

void printDeviceVector(int size, float* vec_d, int offset)
{
	float *vec;
	vec = new float[size];
	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	CheckError(cudaMemcpy(vec, vec_d, size * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	std::cout.precision(7);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i = offset; i < size; i++)
	{
		std::cout << float(vec[i]) << " ";
	}
	std::cout << std::endl;
	delete[] vec;
}

void generateData(int s, float* d_data, float* d_targets) {
	GenerateDataKernel <<<2, s / 2 >>> (d_data, d_targets);
}

/*std::tuple<float*, float**> generateData(int s)
{
	float* data = new float[s];
	float** labels = new float*[s];
	for (int i = 0; i < s; i++) {
		if (i % 2) {
			data[i] = 1.f;
			labels[i] = new float[2]{ 0.f, 1.f };
		}
		else {
			data[i] = 0.f;
			labels[i] = new float[2]{ 1.f, 0.f };
		}
		
	}
	return std::tuple<float*, float**>(data, labels);
}*/

int RoundUp(int nominator, int denominator)
{
	return (nominator + denominator - 1) / denominator;
}