#ifndef __BENCHMARKGPU_TRAINER_H
#define __BENCHMARKGPU_TRAINER_H

#include <cublas_v2.h>

class Trainer {
public:
	/**
	\brief Call Trainer(int, int, float, int) with (0, 128, 0.01, 500).
	*/
	Trainer();
	/**
	*	\brief .
	*	\param GPUID - 
	*	\param batchSize - 
	*	\param learningRate - 
	*	\param epochNumber - 
	*/
	Trainer(int, int, float, int); // GPUID, BatchSize
	~Trainer();

	/**
	*	\brief .
	*	\param seqLength - 
	*	\param data - 
	*	\param output - 
	*	\param trainingSpace - 
	*	\param workspace - 
	*/
	void forwardTraining(int, float*, float*, void*, void*);
private:
	int m_gpuid;
	int m_batchSize;

	cudnnHandle_t m_handle;
	cudnnTensorDescriptor_t m_srcTensorDesc, m_dstTensorDesc, m_biasTensorDesc;
	cudnnRNNDescriptor_t m_rnnDesc;
	cudnnActivationDescriptor_t  m_activDesc;
	cudnnFilterDescriptor_t m_filterDesc;
	cudnnDataType_t m_dataType;
	cudnnTensorFormat_t m_tensorFormat;
	cublasHandle_t m_cublasHandle;
	size_t m_workspaceSize;
	size_t m_trainingSize;
	float m_learningRate;
	int m_epochNumber;
};

#endif // __BENCHMARKGPU_TRAINER_H