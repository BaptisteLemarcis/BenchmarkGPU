#ifndef __BENCHMARKGPU_DENSE_H
#define __BENCHMARKGPU_DENSE_H
#include <vector>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int, int, int);
	~FullyConnectedLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float**);
	void backward(cudnnHandle_t&, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);

private:
	std::vector<float> m_neurons, m_bias;

	float* m_dNeurons;
	float* m_dOutput;
	float* m_dBias;
};

#endif