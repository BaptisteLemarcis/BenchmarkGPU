#pragma once
#include "Layer.h"
class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(int, int);
	~SoftmaxLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, bool);
	float* backward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);
	void updateWeight(cublasHandle_t&, float);

private:
	cudnnTensorDescriptor_t m_outputDesc;
	float* m_d_output;
};

