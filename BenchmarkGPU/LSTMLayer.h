#pragma once
#include "Layer.h"
class LSTMLayer :
	public Layer
{
public:
	LSTMLayer(cudnnHandle_t&,int, int, int, int, int, float);
	~LSTMLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, bool);
	float* backward(cudnnHandle_t&, cublasHandle_t&, float*, float*, float*, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);
	void updateWeight(cublasHandle_t&, float);

private:
	void updateGrad(cudnnHandle_t&, float*);
	void updateGradParameters(cudnnHandle_t&, float*);
	void batchInit();

private:
	int m_batchSize;
	int m_layerNumber;
	int m_hiddenSize;
	float m_dropout;
	int m_seqLength;
	int m_inputSize;

	cudnnTensorFormat_t m_tensorFormat;
	size_t m_workspaceSize;
	size_t m_trainingSize;
	size_t m_weightsSize;
	size_t m_workSize;
	size_t m_reserveSize;
	size_t m_dataTypeSize;

	cudnnRNNDescriptor_t m_rnnDesc;
	cudnnFilterDescriptor_t m_weightsDesc, m_weightsGradientDesc;
	cudnnDropoutDescriptor_t m_dropoutDesc;
	cudnnTensorDescriptor_t *m_srcDataDesc, *m_dstDataDesc, *m_gradientInputDesc, *m_gradientOutputDesc;
	cudnnTensorDescriptor_t m_hiddenInputDesc, m_cellInputDesc;
	cudnnTensorDescriptor_t m_hiddenOutputDesc, m_cellOutputDesc;
	cudnnTensorDescriptor_t m_gradientHiddenInputDesc, m_gradientCellInputDesc;
	cudnnTensorDescriptor_t m_gradHiddenOutputDesc, m_gradCellOutputDesc;

	float *m_d_hiddenInput = NULL;
	float *m_d_cellInput = NULL;

	float *m_d_gradientInput;
	float *m_d_gradientHiddenInput = NULL;
	float *m_d_gradientCellInput = NULL;

	float *m_d_dstData;
	float *m_d_hiddenOutput = NULL;
	float *m_d_cellOutput = NULL;

	float *m_d_gradientOutput;
	float *m_d_gradHiddenOutput = NULL;
	float *m_d_gradCellOutput = NULL;

	float *m_d_weights;
	float *m_d_weightsGradient;

	float *m_d_workspace;
	float *m_d_reserveSpace;
};

