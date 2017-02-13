#ifndef __BENCHMARKGPU_DENSE_H
#define __BENCHMARKGPU_DENSE_H
#include <vector>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int, int, int);
	~FullyConnectedLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*);
	float* backward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);
	void updateWeight(cublasHandle_t&, float);

private:
	float* m_d_neurons;
	float* m_d_grad;
	float* m_d_gradWeight;
	float* m_d_bias;
};

#endif