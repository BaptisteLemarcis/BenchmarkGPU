#include <tuple>
#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <random>
#include "LSTMLayer.h"
#include "GenericFunctions.h"

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
	//CheckCudaError(cudaMalloc((void**)&m_srcData, m_seqLength * m_inputSize * m_batchSize * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_hiddenInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_cellInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize));

	CheckCudaError(cudaMalloc((void**)&m_gradientInput, m_seqLength * m_inputSize * m_batchSize * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_gradientHiddenInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_gradientCellInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));

	CheckCudaError(cudaMalloc((void**)&m_dstData, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_hiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_cellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));

	CheckCudaError(cudaMalloc((void**)&m_gradientOutput, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_gradHiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));
	CheckCudaError(cudaMalloc((void**)&m_gradCellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize));

	m_srcDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dstDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_gradientInputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_gradientOutputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));

	//
	// Create Descriptor
	//
	for (int i = 0; i < m_seqLength; i++) {
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_srcDataDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_dstDataDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradientInputDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradientOutputDesc[i]));
	}

	CheckCudNNError(cudnnCreateTensorDescriptor(&m_hiddenInputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_cellInputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_hiddenOutputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_cellOutputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradientHiddenInputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradientCellInputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradHiddenOutputDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_gradCellOutputDesc));

	CheckCudNNError(cudnnCreateDropoutDescriptor(&m_dropoutDesc));

	CheckCudNNError(cudnnCreateRNNDescriptor(&m_rnnDesc));

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

		CheckCudNNError(cudnnSetTensorNdDescriptor(m_srcDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradientInputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

		dimA[0] = m_batchSize;
		dimA[1] = m_hiddenSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		CheckCudNNError(cudnnSetTensorNdDescriptor(m_dstDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradientOutputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
	}

	dimA[0] = m_layerNumber;
	dimA[1] = m_batchSize;
	dimA[2] = m_hiddenSize;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	CheckCudNNError(cudnnSetTensorNdDescriptor(m_hiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_cellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_hiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_cellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradientHiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradientCellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradHiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_gradCellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

	size_t stateSize;
	void *states;
	unsigned long long seed = 4122ull; // Pick a seed.

	CheckCudNNError(cudnnDropoutGetStatesSize(handle, &stateSize));

	CheckCudaError(cudaMalloc(&states, stateSize));

	CheckCudNNError(cudnnSetDropoutDescriptor(m_dropoutDesc,
		handle,
		m_dropout,
		states,
		stateSize,
		seed));

	CheckCudNNError(cudnnSetRNNDescriptor(m_rnnDesc,
		m_hiddenSize,
		m_layerNumber,
		m_dropoutDesc,
		CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
		CUDNN_UNIDIRECTIONAL,
		CUDNN_LSTM,
		CUDNN_DATA_FLOAT));

	// -------------------------
	// Set up parameters
	// -------------------------
	// This needs to be done after the rnn descriptor is set as otherwise
	// we don't know how many parameters we have to allocate

	CheckCudNNError(cudnnCreateFilterDescriptor(&m_weightsDesc));
	CheckCudNNError(cudnnCreateFilterDescriptor(&m_weightsGradientDesc));

	CheckCudNNError(cudnnGetRNNParamsSize(handle, m_rnnDesc, m_srcDataDesc[0], &m_weightsSize, CUDNN_DATA_FLOAT));

	std::cout << "Number of params : " << (m_weightsSize / m_dataTypeSize) << std::endl;

	int dimW[3];
	dimW[0] = m_weightsSize / m_dataTypeSize;
	dimW[1] = 1;
	dimW[2] = 1;

	CheckCudNNError(cudnnSetFilterNdDescriptor(m_weightsDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW));
	CheckCudNNError(cudnnSetFilterNdDescriptor(m_weightsGradientDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW));

	CheckCudaError(cudaMalloc((void**)&m_weights, m_weightsSize));
	CheckCudaError(cudaMalloc((void**)&m_weightsGradient, m_weightsSize));

	// -------------------------
	// Set up work space and reserved memory
	// -------------------------   
	// Need for every pass
	CheckCudNNError(cudnnGetRNNWorkspaceSize(handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_workSize));
	// Only needed in training, shouldn't be touched between passes.
	CheckCudNNError(cudnnGetRNNTrainingReserveSize(handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_reserveSize));

	CheckCudaError(cudaMalloc((void**)&m_workspace, m_workSize));
	CheckCudaError(cudaMalloc((void**)&m_reserveSpace, m_reserveSize));

	CheckCudaError(cudaDeviceSynchronize());
}

LSTMLayer::~LSTMLayer()
{
	CheckCudaError(cudaFree(m_hiddenInput));
	CheckCudaError(cudaFree(m_cellInput));
	CheckCudaError(cudaFree(m_dstData));
	CheckCudaError(cudaFree(m_hiddenOutput));
	CheckCudaError(cudaFree(m_cellOutput));
	CheckCudaError(cudaFree(m_gradientInput));
	CheckCudaError(cudaFree(m_gradientHiddenInput));
	CheckCudaError(cudaFree(m_gradientCellInput));
	CheckCudaError(cudaFree(m_gradientOutput));
	CheckCudaError(cudaFree(m_gradHiddenOutput));
	CheckCudaError(cudaFree(m_gradCellOutput));
	CheckCudaError(cudaFree(m_workspace));
	CheckCudaError(cudaFree(m_reserveSpace));
	CheckCudaError(cudaFree(m_weights));
	CheckCudaError(cudaFree(m_weightsGradient));

	CheckCudNNError(cudnnDestroyRNNDescriptor(m_rnnDesc));
	CheckCudNNError(cudnnDestroyFilterDescriptor(m_weightsDesc));
	CheckCudNNError(cudnnDestroyFilterDescriptor(m_weightsGradientDesc));
	CheckCudNNError(cudnnDestroyDropoutDescriptor(m_dropoutDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_hiddenInputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_cellInputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_hiddenOutputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_cellOutputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradientHiddenInputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradientCellInputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradHiddenOutputDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradCellOutputDesc));

	for (int i = 0; i < m_seqLength; i++) {
		CheckCudNNError(cudnnDestroyTensorDescriptor(m_srcDataDesc[i]));
		CheckCudNNError(cudnnDestroyTensorDescriptor(m_dstDataDesc[i]));
		CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradientInputDesc[i]));
		CheckCudNNError(cudnnDestroyTensorDescriptor(m_gradientOutputDesc[i]));
	}

	CheckCudaError(cudaDeviceReset());
}

std::tuple<float, float*> LSTMLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* input, float** target)
{
	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_hiddenOutput, 0.f);
	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_cellOutput, 0.f);

	CheckCudNNError(cudnnRNNForwardTraining(handle,
		m_rnnDesc,
		m_seqLength,
		m_srcDataDesc,
		input,
		m_hiddenInputDesc,
		m_hiddenInput,
		m_cellInputDesc,
		m_cellInput,
		m_weightsDesc,
		m_weights,
		m_dstDataDesc,
		m_dstData,
		m_hiddenOutputDesc,
		m_hiddenOutput,
		m_cellOutputDesc,
		m_cellOutput,
		m_workspace,
		m_workSize,
		m_reserveSpace,
		m_reserveSize));

	CheckCudaError(cudaDeviceSynchronize());

	/*if (FLAGS_DEBUG) {
		logFile << "\tError " << sum << std::endl;
	}*/
	return std::make_tuple(0.f, m_dstData);
}

void LSTMLayer::backward(cudnnHandle_t& handle, float* input) {
	updateGrad(handle);
	updateGradParameters(handle, input);
}

void LSTMLayer::updateGrad(cudnnHandle_t& handle)
{
	FillVec(m_seqLength * m_inputSize * m_batchSize, m_gradientInput, 1.f);

	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_gradientHiddenInput, 0.f);
	FillVec(m_layerNumber * m_hiddenSize * m_batchSize, m_gradientCellInput, 0.f);

	CheckCudNNError(cudnnRNNBackwardData(handle,
		m_rnnDesc,
		m_seqLength,
		m_dstDataDesc,
		m_dstData,
		m_gradientOutputDesc,
		m_gradientOutput, // Need to be fill
		m_gradHiddenOutputDesc,
		m_gradHiddenOutput, // Need to be fill
		m_gradCellOutputDesc,
		m_gradCellOutput, // Need to be fill
		m_weightsDesc,
		m_weights, // Update after forward?
		m_hiddenInputDesc,
		m_hiddenInput, // Need to be fill
		m_cellInputDesc,
		m_cellInput, // Need to be fill
		m_gradientInputDesc,
		m_gradientInput,
		m_gradientHiddenInputDesc,
		m_gradientHiddenInput,
		m_gradientCellInputDesc,
		m_gradientCellInput,
		m_workspace,
		m_workSize,
		m_reserveSpace,
		m_reserveSize));

	CheckCudaError(cudaDeviceSynchronize());

	/*if (FLAGS_DEBUG) {
		logFile << "\tGradient output (backward)" << std::endl;
		printDeviceVectorToFile(NBDATADSP, (float *)m_gradientInput, 0);
	}*/

}

void LSTMLayer::updateGradParameters(cudnnHandle_t& handle, float* input)
{

	CheckCudNNError(cudnnRNNBackwardWeights(handle,
		m_rnnDesc,
		m_seqLength,
		m_srcDataDesc,
		input,
		m_hiddenInputDesc,
		m_hiddenInput,
		m_dstDataDesc,
		m_dstData,
		m_workspace,
		m_workSize,
		m_weightsGradientDesc,
		m_weightsGradient,
		m_reserveSpace,
		m_reserveSize));

	CheckCudaError(cudaDeviceSynchronize());
	/*if (FLAGS_DEBUG) {
		logFile << "\tWeights (backward)" << std::endl;
		printDeviceVectorToFile(NBDATADSP, (float *)m_weights, 0);
		logFile << "\tWeights Gradient (backward)" << std::endl;
		printDeviceVectorToFile(NBDATADSP, (float *)m_weightsGradient, 0);
	}*/
}

void LSTMLayer::batchInit()
{
	int s = m_layerNumber * m_hiddenSize * m_batchSize;
	float* hiddenInput = new float[s];
	for (int i = 0; i < s; i++) {
		hiddenInput[i] = 0.f;
	}
	float* cellInput = new float[s];
	for (int i = 0; i < s; i++) {
		cellInput[i] = 0.f;
	}

	if (m_hiddenInput != NULL)
		CheckCudaError(cudaMemcpy(m_hiddenInput, hiddenInput, s*m_dataTypeSize, cudaMemcpyHostToDevice));
	if (m_cellInput != NULL)
		CheckCudaError(cudaMemcpy(m_cellInput, cellInput, s*m_dataTypeSize, cudaMemcpyHostToDevice));

	s = m_seqLength * m_hiddenSize * m_batchSize;
	float* gradientOutput = new float[s];
	for (int i = 0; i < s; i++) {
		gradientOutput[i] = 0.f;
	}
	CheckCudaError(cudaMemcpy(m_gradientOutput, gradientOutput, s*m_dataTypeSize, cudaMemcpyHostToDevice));

	s = m_layerNumber * m_hiddenSize * m_batchSize;
	float* gradHiddenOutput = new float[s];
	for (int i = 0; i < s; i++) {
		gradHiddenOutput[i] = 0.f;
	}
	float* gradCellOutput = new float[s];
	for (int i = 0; i < s; i++) {
		gradCellOutput[i] = 0.f;
	}

	if (m_gradHiddenOutput != NULL)
		CheckCudaError(cudaMemcpy(m_gradHiddenOutput, gradHiddenOutput, s*m_dataTypeSize, cudaMemcpyHostToDevice));
	if (m_gradCellOutput != NULL)
		CheckCudaError(cudaMemcpy(m_gradCellOutput, gradCellOutput, s*m_dataTypeSize, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
}

void LSTMLayer::initWeights(cudnnHandle_t& handle) {
	// Initialize weights
	int numLinearLayers = 8; // 2 for RELU/TANH, 8 for LSTM and 6 for GRU
	int totalNbParams = 0;
	/*if (FLAGS_DEBUG) {
		logFile << "==========================================================" << std::endl;
	}*/
	for (int layer = 0; layer < m_layerNumber; layer++) {
		int nbParams = 0;
		for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
			cudnnFilterDescriptor_t linLayerMatDesc;
			CheckCudNNError(cudnnCreateFilterDescriptor(&linLayerMatDesc));
			float *linLayerMat;

			CheckCudNNError(cudnnGetRNNLinLayerMatrixParams(handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_weights,
				linLayerID,
				linLayerMatDesc,
				(void**)&linLayerMat));

			cudnnDataType_t dataType;
			cudnnTensorFormat_t format;
			int nbDims;
			int filterDimA[3];
			CheckCudNNError(cudnnGetFilterNdDescriptor(linLayerMatDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA));

			//initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));
			std::default_random_engine generator;
			std::uniform_real_distribution<float> distribution(-0.08f, 0.08f);
			float* localLinMat = new float[filterDimA[0] * filterDimA[1] * filterDimA[2]];
			for (int i = 0; i < filterDimA[0] * filterDimA[1] * filterDimA[2]; i++) {
				localLinMat[i] = distribution(generator);
			}
			cudaMemcpy(linLayerMat, localLinMat, filterDimA[0] * filterDimA[1] * filterDimA[2] * m_dataTypeSize, cudaMemcpyHostToDevice);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];

			CheckCudNNError(cudnnDestroyFilterDescriptor(linLayerMatDesc));

			cudnnFilterDescriptor_t linLayerBiasDesc;
			CheckCudNNError(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
			float *linLayerBias;

			CheckCudNNError(cudnnGetRNNLinLayerBiasParams(handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_weights,
				linLayerID,
				linLayerBiasDesc,
				(void**)&linLayerBias));

			CheckCudNNError(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA));

			//initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);
			FillVec(filterDimA[0] * filterDimA[1] * filterDimA[2], linLayerBias, 1.f);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];
			CheckCudNNError(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
		}
		totalNbParams += nbParams;
		/*if (FLAGS_DEBUG) {
			logFile << "Layer\t" << layer << "\t" << nbParams << std::endl;
		}*/
	}

	/*if (FLAGS_DEBUG) {
		logFile << "Totals\t\t" << totalNbParams << std::endl;
		logFile << "==========================================================" << std::endl;
	}*/
}

void LSTMLayer::initEpoch(cudnnHandle_t & handle)
{
	CheckCudaError(cudaMemset(m_weightsGradient, 0.f, m_weightsSize));
}
