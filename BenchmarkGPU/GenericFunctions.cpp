#include <iostream>
#include <fstream>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <sstream>
#include <tuple>
#include <random>

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

void printDeviceVectorToFile(int size, float* d_vec, int offset)
{
	float *vec;
	vec = new float[size];
	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	CheckError(cudaMemcpy(vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
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

void printDeviceVector(int size, float* d_vec, int offset)
{
	float *vec;
	vec = new float[size];
	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	CheckError(cudaMemcpy(vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	std::cout.precision(7);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i = offset; i < size; i++)
	{
		std::cout << float(vec[i]) << " ";
	}
	std::cout << std::endl;
	delete[] vec;
}

int RoundUp(int nominator, int denominator)
{
	return (nominator + denominator - 1) / denominator;
}