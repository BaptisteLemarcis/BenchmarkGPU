#pragma once
#include "Layer.h"
class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(int, int);
	~SoftmaxLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float**);
	void backward(cudnnHandle_t&, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);

private:
	cudnnTensorDescriptor_t m_outputDesc;
	float* m_output;
};

