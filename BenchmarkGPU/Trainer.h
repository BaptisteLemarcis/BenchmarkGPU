#pragma once

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
	cudnnTensorDescriptor_t m_dataTensor;
	cudnnTensorDescriptor_t m_outputTensor;
	size_t m_workspaceSize;
	size_t m_trainingSize;
	cudnnRNNDescriptor_t m_rnnDesc;
	float m_learningRate;
	int m_epochNumber;
};