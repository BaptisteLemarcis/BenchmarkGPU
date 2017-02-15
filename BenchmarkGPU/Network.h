#ifndef __BENCHMARKGPU_NETWORK_H
#define __BENCHMARKGPU_NETWORK_H

#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "Layer.h"

class Network {
public:
	/**
	\brief Call Trainer(int, int, float, int, int, float, int, bool, int, int) with (0, 128, 0.01f, 4, 512, 0.2f, 10, false, 10).
	*/
	Network();

	/**
	*	\brief .
	*	\param batchSize - 
	*	\param learningRate - 
	*	\param inputSize - Number of features
	*	\param outputDim -
	*	\param seqLength -
	*/
	Network(int, float, int, int, int);

	~Network();

	/**
	*	\brief .
	*	\param d_data -  
	*	\param d_labels -
	*	\param epochNumber -
	*	\param nbData -
	*/
	void train(float*, float*, int, int);

	void addLayer(Layer&);

	cudnnHandle_t& getHandle();
private:
	void trainEpoch(int, int, int, int, float*, float*);
	float forward(float*, float*, float*, std::vector<float*>*);
	void backward(std::vector<float*>&, float*, float*);

private:
	int m_gpuid;
	float m_learningRate;
	int m_batchSize;
	int m_seqLength;
	int m_inputDim;
	int m_outputDim;

	cudnnHandle_t m_handle;
	cublasHandle_t m_cublasHandle;

	std::vector<std::reference_wrapper<Layer>> m_layers;
	void validateBatch();
};

#endif // __BENCHMARKGPU_TRAINER_H