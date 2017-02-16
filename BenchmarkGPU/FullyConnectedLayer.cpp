#include <vector>
#include <tuple>
#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <random>

#include "Logger.h"
#include "FullyConnectedLayer.h"
#include "GenericFunctions.h"
#include "GPUKernel.cuh"



FullyConnectedLayer::FullyConnectedLayer(int inputDim, int outputDim, int batchSize) : Layer(inputDim, outputDim, batchSize)
{
	CheckError(cudaMalloc(&m_d_neurons, sizeof(float) * inputDim*outputDim), __FILE__, __LINE__);
	CheckError(cudaMalloc(&m_d_bias, sizeof(float) * outputDim), __FILE__, __LINE__);

	initDataDistributed(-0.08f, 0.08f, m_d_neurons, inputDim*outputDim);
	initDataDistributed(-0.08f, 0.08f, m_d_bias, outputDim);

	/*for (auto&& iter : m_neurons)
		iter = static_cast<float>(distribution(generator));
	for (auto&& iter : m_bias)
		iter = static_cast<float>(distribution(generator));

	CheckError(cudaMemcpyAsync(m_d_neurons, &m_neurons[0], sizeof(float) * inputDim*outputDim, cudaMemcpyHostToDevice), __FILE__, __LINE__);
	CheckError(cudaMemcpyAsync(m_d_bias, &m_bias[0], sizeof(float) * outputDim, cudaMemcpyHostToDevice), __FILE__, __LINE__);*/
}

FullyConnectedLayer::~FullyConnectedLayer()
{
}

std::tuple<float, float*> FullyConnectedLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_input, float* d_target, float* d_onevec, bool training)
{
	float alpha = 1.0f, beta = 0.0f;
	float* d_outpout;
	CheckError(cudaMalloc(&d_outpout, sizeof(float) * m_batchSize * m_outputDim), __FILE__, __LINE__);
	// FC2 layer
	// Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
	CheckError(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		m_outputDim, m_batchSize, m_inputDim,
		&alpha,
		m_d_neurons, m_inputDim,
		d_input, m_inputDim,
		&beta,
		d_outpout, m_outputDim), __FILE__, __LINE__);

	// Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
	CheckError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		m_outputDim, m_batchSize, 1,
		&alpha,
		m_d_bias, m_outputDim,
		d_onevec, 1,
		&alpha,
		m_d_neurons, m_outputDim), __FILE__, __LINE__);

	return std::make_tuple(0.f, d_outpout);
}

float* FullyConnectedLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* dloss_data, float* targets, float* d_onevec, float* previousLayerOutput) {
	
	/*Logger::instance()->writeLine("FullyConnectedLayer bwd");*/
	
	float alpha(1), beta(0);

	float  *d_output; // A modifier
	//float* lstmOutput;

	// TODO
	CheckError(cudaMalloc(&m_d_grad, sizeof(float) * m_inputDim*m_outputDim), __FILE__, __LINE__);
	CheckError(cudaMalloc(&m_d_gradWeight, sizeof(float)  * m_inputDim), __FILE__, __LINE__);
	CheckError(cudaMalloc(&d_output, sizeof(float) * m_batchSize * m_inputDim), __FILE__, __LINE__);

	// FC2 layer
	// Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
	CheckError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m_inputDim, m_outputDim, m_batchSize,
		&alpha, previousLayerOutput, m_inputDim, dloss_data, m_outputDim, &beta, m_d_grad, m_inputDim), __FILE__, __LINE__);
	// Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
	CheckError(cublasSgemv(cublasHandle, CUBLAS_OP_N, m_outputDim, m_batchSize,
		&alpha, dloss_data, m_outputDim, d_onevec, 1, &beta, m_d_gradWeight, 1), __FILE__, __LINE__);

	// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
	CheckError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_inputDim, m_batchSize, m_outputDim,
		&alpha, m_d_neurons, m_inputDim, dloss_data, m_outputDim, &beta, d_output, m_inputDim), __FILE__, __LINE__);

// 	Logger::instance()->writeLine("\tdloss_data ======== 1");
// 	printDeviceVectorToFile(10, d_output, 0);

	return d_output;
}

void FullyConnectedLayer::initWeights(cudnnHandle_t & handle)
{

}

void FullyConnectedLayer::initEpoch(cudnnHandle_t & handle)
{
}

void FullyConnectedLayer::updateWeight(cublasHandle_t& cublasHandle, float lr)
{
	float alpha = -lr;

	CheckError(cublasSaxpy(cublasHandle, m_inputDim*m_outputDim,
		&alpha, m_d_grad, 1, m_d_neurons, 1), __FILE__, __LINE__);

	CheckError(cublasSaxpy(cublasHandle, m_outputDim,
		&alpha, m_d_gradWeight, 1, m_d_bias, 1), __FILE__, __LINE__);

	/*printDeviceVectorToFile(2, m_d_bias, 0);

	printDeviceVectorToFile(2, m_d_gradWeight, 0);*/
}

