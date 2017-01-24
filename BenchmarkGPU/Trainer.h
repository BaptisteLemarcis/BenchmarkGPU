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
	Trainer(int, int); // GPUID, BatchSize
	~Trainer();

	/**
	*	\brief Create an object Trainer with GPU zero and a batchSize of 128.
	*	\param zero - toto
	*	\param one - titi
	*/
	void doStuff(int, int);
private:
	int m_gpuid;
	int m_batchSize;

	cudnnHandle_t m_handle;
	cudnnTensorDescriptor_t m_dataTensor;
};