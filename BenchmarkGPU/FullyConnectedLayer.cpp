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

std::tuple<float, float*> FullyConnectedLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* input, float** target, float* d_onevec)
{
	float alpha = 1.0f, beta = 0.0f;
	
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

float* FullyConnectedLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* dloss_data, float** targets, float* d_onevec) {
	float alpha(1), beta(0);

	float *d_gfc2, *d_gfc2bias, *d_dfc2, *d_pfc2, *d_pfc2bias; // A modifier
	float* lstmOutput;


	// TODO
	CheckCudaError(cudaMalloc(&d_gfc2, sizeof(float) /* * fc2.pneurons.size()*/));
	CheckCudaError(cudaMalloc(&d_gfc2bias, sizeof(float) /* * fc2.pbias.size()*/));
	CheckCudaError(cudaMalloc(&d_dfc2, sizeof(float) * m_batchSize /* * fc2.inputs*/));
	CheckCudaError(cudaMemcpyAsync(d_pfc2, NULL /*&fc2.pneurons[0]*/, sizeof(float) /* * fc2.pneurons.size()*/, cudaMemcpyHostToDevice));
	CheckCudaError(cudaMemcpyAsync(d_pfc2bias, NULL /*&fc2.pbias[0]*/, sizeof(float) /* * fc2.pbias.size()*/, cudaMemcpyHostToDevice));

	// FC2 layer
	// Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
	CheckCublasError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m_inputDim, m_ouputDim, m_batchSize,
		&alpha, lstmOutput, m_inputDim, dloss_data, m_ouputDim, &beta, d_gfc2, m_inputDim));
	// Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
	CheckCublasError(cublasSgemv(cublasHandle, CUBLAS_OP_N, m_ouputDim, m_batchSize,
		&alpha, dloss_data, m_ouputDim, d_onevec, 1, &beta, d_gfc2bias, 1));
	// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
	CheckCublasError(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_inputDim, m_batchSize, m_ouputDim,
		&alpha, d_pfc2, m_inputDim, dloss_data, m_ouputDim, &beta, d_dfc2, m_inputDim));

	return d_dfc2;
}

void FullyConnectedLayer::initWeights(cudnnHandle_t & handle)
{

}

void FullyConnectedLayer::initEpoch(cudnnHandle_t & handle)
{
}

