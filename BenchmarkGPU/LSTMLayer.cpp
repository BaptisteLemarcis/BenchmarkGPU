#include <tuple>
#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <random>

#include "Logger.h"
#include "LSTMLayer.h"
#include "GenericFunctions.h"
#include "GPUKernel.cuh"

LSTMLayer::LSTMLayer(cudnnHandle_t& handle, int inputDim, int outputDim, int nbLayer, int batchSize, int seqLength, float dropout) : Layer(inputDim, outputDim, batchSize)
{
	m_batchSize = batchSize;
	m_layerNumber = nbLayer;
	m_hiddenSize = outputDim;
	m_dropout = dropout;
	m_inputSize = inputDim;
	m_seqLength = seqLength;
	m_tensorFormat = CUDNN_TENSOR_NCHW;
	m_dataTypeSize = sizeof(float);

	// Allocating
	//CheckError(cudaMalloc((void**)&m_srcData, m_seqLength * m_inputSize * m_batchSize * m_dataTypeSize));
	CheckError(cudaMalloc((void**)&m_d_hiddenInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_cellInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);

	CheckError(cudaMalloc((void**)&m_d_gradientInput, m_seqLength * m_inputSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_gradientHiddenInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_gradientCellInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

	CheckError(cudaMalloc((void**)&m_d_dstData, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_hiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_cellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

	CheckError(cudaMalloc((void**)&m_d_gradientOutput, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_gradHiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_gradCellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

	m_srcDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dstDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_gradientInputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_gradientOutputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));

	//
	// Create Descriptor
	//
	for (int i = 0; i < m_seqLength; i++) {
		CheckError(cudnnCreateTensorDescriptor(&m_srcDataDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnCreateTensorDescriptor(&m_dstDataDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnCreateTensorDescriptor(&m_gradientInputDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnCreateTensorDescriptor(&m_gradientOutputDesc[i]), __FILE__, __LINE__);
	}

	CheckError(cudnnCreateTensorDescriptor(&m_hiddenInputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_cellInputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_hiddenOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_cellOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_gradientHiddenInputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_gradientCellInputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_gradHiddenOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateTensorDescriptor(&m_gradCellOutputDesc), __FILE__, __LINE__);

	CheckError(cudnnCreateDropoutDescriptor(&m_dropoutDesc), __FILE__, __LINE__);

	CheckError(cudnnCreateRNNDescriptor(&m_rnnDesc), __FILE__, __LINE__);

	//
	// Setting up TensorDescriptor
	//

	int dimA[3];
	int strideA[3];

	for (int i = 0; i < m_seqLength; i++) {
		dimA[0] = m_batchSize;
		dimA[1] = m_inputSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		CheckError(cudnnSetTensorNdDescriptor(m_srcDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
		CheckError(cudnnSetTensorNdDescriptor(m_gradientInputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);

		dimA[0] = m_batchSize;
		dimA[1] = m_hiddenSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		CheckError(cudnnSetTensorNdDescriptor(m_dstDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
		CheckError(cudnnSetTensorNdDescriptor(m_gradientOutputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	}

	dimA[0] = m_layerNumber;
	dimA[1] = m_batchSize;
	dimA[2] = m_hiddenSize;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	CheckError(cudnnSetTensorNdDescriptor(m_hiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_cellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_hiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_cellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_gradientHiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_gradientCellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_gradHiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudnnSetTensorNdDescriptor(m_gradCellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);

	size_t stateSize;
	void *states;
	unsigned long long seed = 4122ull; // Pick a seed.

	CheckError(cudnnDropoutGetStatesSize(handle, &stateSize), __FILE__, __LINE__);

	CheckError(cudaMalloc(&states, stateSize), __FILE__, __LINE__);

	CheckError(cudnnSetDropoutDescriptor(m_dropoutDesc,
		handle,
		m_dropout,
		states,
		stateSize,
		seed), __FILE__, __LINE__);

	CheckError(cudnnSetRNNDescriptor(m_rnnDesc,
		m_hiddenSize,
		m_layerNumber,
		m_dropoutDesc,
		CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
		CUDNN_UNIDIRECTIONAL,
		CUDNN_LSTM,
		CUDNN_DATA_FLOAT), __FILE__, __LINE__);

	// -------------------------
	// Set up parameters
	// -------------------------
	// This needs to be done after the rnn descriptor is set as otherwise
	// we don't know how many parameters we have to allocate

	CheckError(cudnnCreateFilterDescriptor(&m_weightsDesc), __FILE__, __LINE__);
	CheckError(cudnnCreateFilterDescriptor(&m_weightsGradientDesc), __FILE__, __LINE__);

	CheckError(cudnnGetRNNParamsSize(handle, m_rnnDesc, m_srcDataDesc[0], &m_weightsSize, CUDNN_DATA_FLOAT), __FILE__, __LINE__);

	std::cout << "Number of params : " << (m_weightsSize / m_dataTypeSize) << std::endl;

	int dimW[3];
	dimW[0] = m_weightsSize / m_dataTypeSize;
	dimW[1] = 1;
	dimW[2] = 1;

	CheckError(cudnnSetFilterNdDescriptor(m_weightsDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW), __FILE__, __LINE__);
	CheckError(cudnnSetFilterNdDescriptor(m_weightsGradientDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW), __FILE__, __LINE__);

	CheckError(cudaMalloc((void**)&m_d_weights, m_weightsSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_weightsGradient, m_weightsSize), __FILE__, __LINE__);

	// -------------------------
	// Set up work space and reserved memory
	// -------------------------   
	// Need for every pass
	CheckError(cudnnGetRNNWorkspaceSize(handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_workSize), __FILE__, __LINE__);
	// Only needed in training, shouldn't be touched between passes.
	CheckError(cudnnGetRNNTrainingReserveSize(handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_reserveSize), __FILE__, __LINE__);

	CheckError(cudaMalloc((void**)&m_d_workspace, m_workSize), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&m_d_reserveSpace, m_reserveSize), __FILE__, __LINE__);

	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

LSTMLayer::~LSTMLayer()
{
	CheckError(cudaFree(m_d_hiddenInput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_cellInput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_dstData), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_hiddenOutput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_cellOutput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradientInput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradientHiddenInput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradientCellInput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradientOutput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradHiddenOutput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_gradCellOutput), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_workspace), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_reserveSpace), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_weights), __FILE__, __LINE__);
	CheckError(cudaFree(m_d_weightsGradient), __FILE__, __LINE__);

	CheckError(cudnnDestroyRNNDescriptor(m_rnnDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyFilterDescriptor(m_weightsDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyFilterDescriptor(m_weightsGradientDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyDropoutDescriptor(m_dropoutDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_hiddenInputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_cellInputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_hiddenOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_cellOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_gradientHiddenInputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_gradientCellInputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_gradHiddenOutputDesc), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_gradCellOutputDesc), __FILE__, __LINE__);

	for (int i = 0; i < m_seqLength; i++) {
		CheckError(cudnnDestroyTensorDescriptor(m_srcDataDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnDestroyTensorDescriptor(m_dstDataDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnDestroyTensorDescriptor(m_gradientInputDesc[i]), __FILE__, __LINE__);
		CheckError(cudnnDestroyTensorDescriptor(m_gradientOutputDesc[i]), __FILE__, __LINE__);
	}

	CheckError(cudaDeviceReset(), __FILE__, __LINE__);
}

std::tuple<float, float*> LSTMLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_input, float* d_target, float* d_onevec, bool training)
{
	if (training) {
		FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_d_hiddenOutput, 0.f);
		FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_d_cellOutput, 0.f);
		FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_d_dstData, 0.f);

		CheckError(cudnnRNNForwardTraining(handle,
			m_rnnDesc,
			m_seqLength,
			m_srcDataDesc,
			d_input,
			m_hiddenInputDesc,
			m_d_hiddenInput,
			m_cellInputDesc,
			m_d_cellInput,
			m_weightsDesc,
			m_d_weights,
			m_dstDataDesc,
			m_d_dstData,
			m_hiddenOutputDesc,
			m_d_hiddenOutput,
			m_cellOutputDesc,
			m_d_cellOutput,
			m_d_workspace,
			m_workSize,
			m_d_reserveSpace,
			m_reserveSize), __FILE__, __LINE__);
	} else {
		CheckError(cudnnRNNForwardInference(handle,
			m_rnnDesc,
			m_seqLength,
			m_srcDataDesc,
			d_input,
			m_hiddenInputDesc,
			m_d_hiddenInput,
			m_cellInputDesc,
			m_d_cellInput,
			m_weightsDesc,
			m_d_weights,
			m_dstDataDesc,
			m_d_dstData,
			m_hiddenOutputDesc,
			m_d_hiddenOutput,
			m_cellOutputDesc,
			m_d_cellOutput,
			m_d_workspace,
			m_workSize), __FILE__, __LINE__);
	}
	

	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	return std::make_tuple(0.f, m_d_dstData);
}

float* LSTMLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_loss_data, float* d_targets, float* d_onevec, float* d_previousLayerOutput) {
	
	updateGrad(handle, d_loss_data);
	updateGradParameters(handle, d_loss_data);

	return m_d_gradientInput;
}

void LSTMLayer::updateGrad(cudnnHandle_t& handle, float* d_grad)
{
	FillVec(m_seqLength * m_inputSize * m_batchSize, m_d_gradientInput, 0.f);

	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_d_gradientHiddenInput, 0.f);
	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_d_gradientCellInput, 0.f);

	CheckError(cudnnRNNBackwardData(handle,
		m_rnnDesc,
		m_seqLength,
		m_dstDataDesc,
		m_d_dstData,
		m_gradientOutputDesc,
		d_grad,
		m_gradHiddenOutputDesc,
		m_d_gradHiddenOutput,
		m_gradCellOutputDesc,
		m_d_gradCellOutput,
		m_weightsDesc,
		m_d_weights,
		m_hiddenInputDesc,
		m_d_hiddenInput,
		m_cellInputDesc,
		m_d_cellInput,
		m_gradientInputDesc,
		m_d_gradientInput,
		m_gradientHiddenInputDesc,
		m_d_gradientHiddenInput,
		m_gradientCellInputDesc,
		m_d_gradientCellInput,
		m_d_workspace,
		m_workSize,
		m_d_reserveSpace,
		m_reserveSize), __FILE__, __LINE__);

	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

void LSTMLayer::updateGradParameters(cudnnHandle_t& handle, float* d_input)
{

	CheckError(cudnnRNNBackwardWeights(handle,
		m_rnnDesc,
		m_seqLength,
		m_srcDataDesc,
		d_input,
		m_hiddenInputDesc,
		m_d_hiddenInput,
		m_dstDataDesc,
		m_d_dstData,
		m_d_workspace,
		m_workSize,
		m_weightsGradientDesc,
		m_d_weightsGradient,
		m_d_reserveSpace,
		m_reserveSize), __FILE__, __LINE__);

	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

void LSTMLayer::batchInit()
{
	int s = m_layerNumber * m_hiddenSize * m_batchSize;

	if (m_d_hiddenInput != NULL)
		FillVec(s, m_d_hiddenInput, 0.f);
	if (m_d_cellInput != NULL)
		FillVec(s, m_d_cellInput, 0.f);

	s = m_seqLength * m_hiddenSize * m_batchSize;
	FillVec(s, m_d_gradientOutput, 0.f);

	s = m_layerNumber * m_hiddenSize * m_batchSize;
	if (m_d_gradHiddenOutput != NULL)
		FillVec(s, m_d_gradHiddenOutput, 0.f);
	if (m_d_gradCellOutput != NULL)
		FillVec(s, m_d_gradCellOutput, 0.f);

	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

void LSTMLayer::initWeights(cudnnHandle_t& handle) {
	// Initialize weights
	int numLinearLayers = 8; // 2 for RELU/TANH, 8 for LSTM and 6 for GRU
	int totalNbParams = 0;

	for (int layer = 0; layer < m_layerNumber; layer++) {
		int nbParams = 0;
		for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
			cudnnFilterDescriptor_t linLayerMatDesc;
			CheckError(cudnnCreateFilterDescriptor(&linLayerMatDesc), __FILE__, __LINE__);
			float *d_linLayerMat;

			CheckError(cudnnGetRNNLinLayerMatrixParams(handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_d_weights,
				linLayerID,
				linLayerMatDesc,
				(void**)&d_linLayerMat), __FILE__, __LINE__);

			cudnnDataType_t dataType;
			cudnnTensorFormat_t format;
			int nbDims;
			int filterDimA[3];
			CheckError(cudnnGetFilterNdDescriptor(linLayerMatDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA), __FILE__, __LINE__);

			initDataDistributed(-0.08f, 0.08f, d_linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2]);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];

			CheckError(cudnnDestroyFilterDescriptor(linLayerMatDesc), __FILE__, __LINE__);

			cudnnFilterDescriptor_t linLayerBiasDesc;
			CheckError(cudnnCreateFilterDescriptor(&linLayerBiasDesc), __FILE__, __LINE__);
			float *d_linLayerBias;

			CheckError(cudnnGetRNNLinLayerBiasParams(handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_d_weights,
				linLayerID,
				linLayerBiasDesc,
				(void**)&d_linLayerBias), __FILE__, __LINE__);

			CheckError(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA), __FILE__, __LINE__);

			FillVec(filterDimA[0] * filterDimA[1] * filterDimA[2], d_linLayerBias, 1.f);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];
			CheckError(cudnnDestroyFilterDescriptor(linLayerBiasDesc), __FILE__, __LINE__);
		}
		totalNbParams += nbParams;
	}
}

void LSTMLayer::initEpoch(cudnnHandle_t & handle)
{
	CheckError(cudaMemset(m_d_weightsGradient, 0.f, m_weightsSize), __FILE__, __LINE__);
}

void LSTMLayer::updateWeight(cublasHandle_t& cublasHandle, float lr) {
	float alpha = -lr;

	CheckError(cublasSaxpy(cublasHandle, m_layerNumber*m_batchSize*m_hiddenSize,
		&alpha, m_d_gradientHiddenInput, 1, m_d_hiddenInput, 1), __FILE__, __LINE__);

	CheckError(cublasSaxpy(cublasHandle, m_weightsSize/m_dataTypeSize,
		&alpha, m_d_weightsGradient, 1, m_d_weights, 1), __FILE__, __LINE__);
}