#pragma once

#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>

class Layer
{
public:
	/**
	*	\brief .
	*	\param inputDim -
	*	\param outputDim -
	*	\param batchSize -
	*/
	Layer(int, int, int);

	~Layer();

	/**
	*	\brief .
	*	\return Size of the output dimension for this layer
	*/
	int getOutputDim();

	/**
	*	\brief .
	*	\param cudnnHandle -
	*	\param cublasHandle -
	*	\param input - Number of features
	*	\param targets -
	*	\param d_onevec -
	*/
	virtual std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*) = 0;
	
	/**
	*	\brief .
	*	\param cudnnHandle -
	*	\param cublasHandle -
	*	\param input - Number of features
	*	\param targets -
	*	\param d_onevec -
	*	\param previousLayerOutput -
	*/
	virtual float* backward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, float*) = 0;

	/**
	*	\brief .
	*	\param cudnnHandle -
	*/
	virtual void initWeights(cudnnHandle_t&) = 0;

	/**
	*	\brief .
	*	\param cudnnHandle -
	*/
	virtual void initEpoch(cudnnHandle_t&) = 0;

	/**
	*	\brief .
	*	\param cublasHandle - - 
	*	\param lr - 
	*/
	virtual void updateWeight(cublasHandle_t&, float) = 0;

protected:
	int m_inputDim;
	int m_outputDim;
	int m_batchSize;
};

