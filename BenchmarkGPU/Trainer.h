#pragma once

class Trainer {
public:
	/**
	\brief Create an object Trainer with GPU zero and a batchSize of 128.
	*/
	Trainer();
	/**
	*	\brief Create an object Trainer with GPU zero and a batchSize of 128.
	*	\param zero - toto
	*	\param one - titi
	*/
	Trainer(int, int, float, int); // GPUID, BatchSize
	~Trainer();

	/**
	*	\brief Create an object Trainer with GPU zero and a batchSize of 128.
	*	\param zero - toto
	*	\param one - titi
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