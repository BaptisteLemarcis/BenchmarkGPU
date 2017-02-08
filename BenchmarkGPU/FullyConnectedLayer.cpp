#include <vector>
#include <tuple>
#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <random>

#include "FullyConnectedLayer.h"
#include "GenericFunctions.h"

FullyConnectedLayer::FullyConnectedLayer(int inputDim, int outputDim, int batchSize) : Layer(inputDim, outputDim, batchSize), m_neurons(inputDim*outputDim), m_bias(outputDim)
{
	CheckCudaError(cudaMalloc(&m_dNeurons, sizeof(float) * inputDim*outputDim));
	CheckCudaError(cudaMalloc(&m_dBias, sizeof(float) * outputDim));
	CheckCudaError(cudaMalloc(&m_dOutput, sizeof(float) * m_batchSize * m_ouputDim));

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-0.08f, 0.08f);
	for (auto&& iter : m_neurons)
		iter = static_cast<float>(distribution(generator));
	for (auto&& iter : m_bias)
		iter = static_cast<float>(distribution(generator));

	CheckCudaError(cudaMemcpyAsync(m_dNeurons, &m_neurons[0], sizeof(float) * inputDim*outputDim, cudaMemcpyHostToDevice));
	CheckCudaError(cudaMemcpyAsync(m_dBias, &m_bias[0], sizeof(float) * outputDim, cudaMemcpyHostToDevice));
}


FullyConnectedLayer::~FullyConnectedLayer()
{
}

std::tuple<float, float*> FullyConnectedLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* input, float** target)
{
	float alpha = 1.0f, beta = 0.0f;
	float *d_onevec;
	CheckCudaError(cudaMalloc(&d_onevec, sizeof(float)* m_batchSize));
	// FC2 layer
	// Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
	CheckCublasError(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		m_ouputDim, m_batchSize, m_inputDim,
		&alpha,
		m_dNeurons, m_inputDim,
		input, m_inputDim,
		&beta,
		m_dOutput, m_ouputDim));

	// Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
	CheckCublasError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		m_ouputDim, m_batchSize, 1,
		&alpha,
		m_dBias, m_ouputDim,
		d_onevec, 1,
		&alpha,
		m_dNeurons, m_ouputDim));

	return std::make_tuple(0.f, m_dOutput);
}

void FullyConnectedLayer::backward(cudnnHandle_t& handle, float* input) {

}

void FullyConnectedLayer::initWeights(cudnnHandle_t & handle)
{

}

void FullyConnectedLayer::initEpoch(cudnnHandle_t & handle)
{
}

