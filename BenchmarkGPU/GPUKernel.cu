#include <iostream>
#include <fstream>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <sstream>
#include <tuple>
#include <device_launch_parameters.h>
#include <random>
#include <curand_kernel.h>
#include <ctime>

#include "Logger.h"
#include "GenericFunctions.h"

#define BW 128

__global__ void FillVecKernel(float* d_vec, float value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx <= size) return;
	d_vec[idx] = value;
}

__global__ void GenerateDataKernel(float* d_data, float* d_target, int size) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idxTarget1 = 2 * idx;
	int idxTarget2 = (2 * (idx + 1)) - 1;

	if (idx >= size) return;

	if (idx%2 == 0) {
		d_data[idx] = 1.f;
		d_target[idxTarget1] = 1.f;
		d_target[idxTarget2] = 0.f;
	}
	else
	{
		d_data[idx] = 0.f;
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

__global__ void PrepareDataKernel(float* d_batchData, float* d_batchTarget, int start, int end, float* d_input, float* d_targets) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx + start >= end) return;
	int idxInput = idx + start;
	d_batchData[idx] = d_input[idxInput];

	d_batchTarget[2 * idx] = d_targets[2 * (idxInput)];
	d_batchTarget[(2 * idx) + 1] = d_targets[(2 * idxInput) + 1];
}

__global__ void SoftmaxLossKernel(float* d_target, float* d_loss_data, int size) {
	//int idxTarget = (blockDim.x * blockIdx.x + threadIdx.x) + blockIdx.y;
	int idx = ((blockDim.x * blockIdx.x + threadIdx.x) * blockDim.y) + blockIdx.y;

	if (idx >= size) return;

	if (d_target[idx] == 1.f)
		d_loss_data[idx] -= 1.f;
}

__global__ void SoftmaxErrorKernel(float* d_error, float* d_input, float* d_targets, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idx1 = 2 * idx;
	int idx2 = idx1 + 1;

	/*int idx3 = 2 * (idx+1);
	int idx4 = (2 * ((idx+1) + 1)) - 1;*/

	if (idx <= size) return;
	float z = d_input[idx1] - d_targets[idx1];
	*d_error = *d_error + (z*z);
	z = d_input[idx2] - d_targets[idx2];
	*d_error = *d_error + (z*z);
}

__global__ void ComputeConfusionMatrixKernel(float* d_input, float* d_targets, float* d_results, int size, int nbClass) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx >= size) return;

	if (d_input[idx] != d_targets[idx]) {
		d_results[(idx%nbClass)] += 1.f;
	}
}

void initDataDistributed(float min, float max, float* d_vec, int size) {
	InitDataDistributedKernel <<< RoundUp(size, BW), BW >>> (min, max, d_vec, size);
}

void FillVec(int size, float* vec_d, float value)
{
	FillVecKernel <<<RoundUp(size, BW), BW >>> (vec_d, value, size);
	/*float* data = new float[size];
	for (int i = 0; i < size; i++) {
	data[i] = value;
	}
	CheckError(cudaMemcpy(vec_d, data, size*sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);*/
}

void generateData(int s, float* d_data, float* d_targets) {
	GenerateDataKernel <<<RoundUp(s, BW), BW >> > (d_data, d_targets, s);
}

void prepareData(float* d_input, float* d_target, int b, float* d_batchData, float* d_batchTarget, int inputDim, int batchSize) {

	int start = b*inputDim;
	int end = (b + batchSize)*inputDim;
	int size = end - start;
	PrepareDataKernel <<< RoundUp(size, BW), BW >>> (d_batchData, d_batchTarget, start, end, d_input, d_target);
}

void softmaxError(int batchSize, float* d_batchError, float* d_input, float* d_targets) {
	SoftmaxErrorKernel <<<RoundUp(batchSize, BW), BW >>> (d_batchError, d_input, d_targets, batchSize);
}

void softmaxLoss(int batchSize, int inputDim, float* d_targets, float* d_loss_data) {
	dim3 b = { (unsigned int)RoundUp(batchSize,BW), (unsigned int)inputDim, 1 };
	dim3 t = { BW , 1, 1 };

	SoftmaxLossKernel <<<b, t >>>(d_targets, d_loss_data, batchSize*inputDim);
}

void computeConfusionMatrix(float* d_input, float* d_target, float* d_results, int size, int nbClass)
{
	std::stringstream ss("");
	ss << "Size : " << size << std::endl;
	Logger::instance()->writeLine(ss.str());
	ComputeConfusionMatrixKernel <<<RoundUp(size*nbClass, BW), BW >> >(d_input, d_target, d_results, size*nbClass, nbClass);
}
