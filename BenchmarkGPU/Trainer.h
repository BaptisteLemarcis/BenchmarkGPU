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
	*	\param layerNumber -
	*	\param hiddenSize -
	*	\param dropout - 
	*	\param numberData -
	*	\param bidirectional -
	*	\param inputSize -
	*/
	Trainer(int, int, float, int, int, float, int, bool, int);

	~Trainer();

	/**
	*	\brief .
	*	\param data -  
	*	\param epochNumber -
	*/
	void train(float*, int);
private:
	void initGPUData(float*, int, float);


private:
	int m_gpuid;
	int m_batchSize;
	float m_learningRate;
	int m_layerNumber;
	int m_hiddenSize;
	float m_dropout;
	int m_epochNumber;
	int m_numberData;
	int m_seqLength;
	int m_inputSize;
	bool m_bidirectional;

	cudnnHandle_t m_handle;
	cudnnDataType_t m_dataType;
	cudnnTensorFormat_t m_tensorFormat;
	cublasHandle_t m_cublasHandle;
	size_t m_workspaceSize;
	size_t m_trainingSize;
	size_t m_weightsSize;
	size_t m_workSize;
	size_t m_reserveSize;

	cudnnRNNDescriptor_t m_rnnDesc;
	cudnnFilterDescriptor_t m_wDesc, m_dwDesc;
	cudnnDropoutDescriptor_t m_dropoutDesc;
	cudnnTensorDescriptor_t *m_xDesc, *m_yDesc, *m_dxDesc, *m_dyDesc;
	cudnnTensorDescriptor_t m_hxDesc, m_cxDesc;
	cudnnTensorDescriptor_t m_hyDesc, m_cyDesc;
	cudnnTensorDescriptor_t m_dhxDesc, m_dcxDesc;
	cudnnTensorDescriptor_t m_dhyDesc, m_dcyDesc;


	void *m_x;
	void *m_hx = NULL;
	void *m_cx = NULL;

	void *m_dx;
	void *m_dhx = NULL;
	void *m_dcx = NULL;

	void *m_y;
	void *m_hy = NULL;
	void *m_cy = NULL;

	void *m_dy;
	void *m_dhy = NULL;
	void *m_dcy = NULL;

	void *m_w;
	void *m_dw;

	void *m_workspace;
	void *m_reserveSpace;
};

#endif // __BENCHMARKGPU_TRAINER_H