#pragma once

#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>

class Layer
{
public:
	Layer(int, int, int);
	~Layer();

	int getOutputDim();

	virtual std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float**, float*) = 0;
	virtual float* backward(cudnnHandle_t&, cublasHandle_t&, float*, float**, float*) = 0;
	virtual void initWeights(cudnnHandle_t&) = 0;
	virtual void initEpoch(cudnnHandle_t&) = 0;

protected:
	int m_inputDim;
	int m_ouputDim;
	int m_batchSize;;
};

