#pragma once
#include "Layer.h"
class LSTMLayer :
	public Layer
{
public:
	LSTMLayer(cudnnHandle_t&,int, int, int, int, int, float);
	~LSTMLayer();

	std::tuple<float, float*> forward(cudnnHandle_t&, cublasHandle_t&, float*, float**);
	void backward(cudnnHandle_t&, float*);
	void initWeights(cudnnHandle_t&);
	void initEpoch(cudnnHandle_t&);

private:
	void updateGrad(cudnnHandle_t&);
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

	float *m_hiddenInput = NULL;
	float *m_cellInput = NULL;

	float *m_gradientInput;
	float *m_gradientHiddenInput = NULL;
	float *m_gradientCellInput = NULL;

	float *m_dstData;
	float *m_hiddenOutput = NULL;
	float *m_cellOutput = NULL;

	float *m_gradientOutput;
	float *m_gradHiddenOutput = NULL;
	float *m_gradCellOutput = NULL;

	float *m_weights;
	float *m_weightsGradient;

	float *m_workspace;
	float *m_reserveSpace;
};

