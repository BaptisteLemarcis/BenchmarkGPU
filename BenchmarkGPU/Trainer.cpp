#include <iostream>
#include <string.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>

#include "Trainer.h"
#include "GenericFunctions.h"

Trainer::Trainer() : Trainer(0, 128, 0.01f, 4, 512, 0.f, 10, false, 10) {}

Trainer::Trainer(int GPUID, int batchSize, float learningRate, int layerNumber, int hiddenSize, float dropout, int numberData, bool bidirectional, int inputSize) {
	m_gpuid = GPUID;
	m_batchSize = batchSize;
	m_learningRate = learningRate;
	m_layerNumber = layerNumber;
	m_hiddenSize = hiddenSize;
	m_dropout = dropout;
	m_numberData = numberData;
	m_bidirectional = bidirectional;
	m_inputSize = inputSize;
	m_seqLength = 10;

	//
	// Listing GPU Devices
	//

	int gpuNumbers;
	CheckCudaError(cudaGetDeviceCount(&gpuNumbers));
	int i = 0;
	cudaDeviceProp prop;
	std::cout << "Cuda capable devices " << gpuNumbers << ":" << std::endl;
	for (i = 0; i < gpuNumbers; i++) {
		CheckCudaError(cudaGetDeviceProperties(&prop, i));
		std::cout << "\tdevice " << i << " (" << prop.name << ") : Proc " << prop.multiProcessorCount << ", Capabilities " << prop.major << "." << prop.minor << ", SmClock "<< (float)prop.clockRate*1e-3 <<" Mhz" << ", MemSize(Mb) " << (int)(prop.totalGlobalMem / (1024 * 1024)) << ", MemClock " << (float)prop.memoryClockRate*1e-3 << " Mhz" << std::endl;
	}
	
	//
	// Setting CUDA device
	//

	std::cout << "Using device " << m_gpuid << std::endl;
	CheckCudaError(cudaSetDevice(m_gpuid));

	//
	// Getting CudNN version
	//

	size_t version = cudnnGetVersion();
	std::cout << "CudNN version " << version << std::endl;

	//
	//	Setting up important var
	//
	m_dataType = CUDNN_DATA_FLOAT;
	m_tensorFormat = CUDNN_TENSOR_NCHW;

	//
	// Create CuDNN Handler
	//
	std::cout << "Creating cudnn handler " << std::endl;
	CheckCudNNError(cudnnCreate(&m_handle));

	//
	// Create Cublas Handler
	//
	std::cout << "Creating cublas handler " << std::endl;
	CheckCublasError(cublasCreate(&m_cublasHandle));

	// Allocating
	CheckCudaError(cudaMalloc((void**)&m_x, m_seqLength * m_inputSize * m_batchSize * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_hx, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_cx, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));

	CheckCudaError(cudaMalloc((void**)&m_dx, m_seqLength * m_inputSize * m_batchSize * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_dhx, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_dcx, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));

	CheckCudaError(cudaMalloc((void**)&m_y, m_seqLength * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_hy, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_cy, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));

	CheckCudaError(cudaMalloc((void**)&m_dy, m_seqLength * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_dhy, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));
	CheckCudaError(cudaMalloc((void**)&m_dcy, m_layerNumber * hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1) * sizeof(float)));

	m_xDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_yDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dxDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dyDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));

	//
	// Create Descriptor
	//

	std::cout << "Creating TensorDescriptor " << std::endl;
	for (int i = 0; i < m_seqLength; i++) {
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_xDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_yDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_dxDesc[i]));
		CheckCudNNError(cudnnCreateTensorDescriptor(&m_dyDesc[i]));
	}

	CheckCudNNError(cudnnCreateTensorDescriptor(&m_hxDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_cxDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_hyDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_cyDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dhxDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dcxDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dhyDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dcyDesc));

	CheckCudNNError(cudnnCreateDropoutDescriptor(&m_dropoutDesc));

	CheckCudNNError(cudnnCreateRNNDescriptor(&m_rnnDesc));

	//
	// Setting up TensorDescriptor
	//
	int dimA[3];
	int strideA[3];

	for (int i = 0; i < m_seqLength; i++) {
		dimA[0] = m_batchSize;
		dimA[1] = m_numberData;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		CheckCudNNError(cudnnSetTensorNdDescriptor(m_xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		CheckCudNNError(cudnnSetTensorNdDescriptor(m_dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

		dimA[0] = m_batchSize;
		dimA[1] = (m_bidirectional ? 2 * m_hiddenSize : m_hiddenSize);
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		CheckCudNNError(cudnnSetTensorNdDescriptor(m_yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		CheckCudNNError(cudnnSetTensorNdDescriptor(m_dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
	}

	dimA[0] = m_layerNumber*(m_bidirectional ? 2 : 1);
	dimA[1] = m_batchSize;
	dimA[2] = m_hiddenSize;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;	

	CheckCudNNError(cudnnSetTensorNdDescriptor(m_hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	CheckCudNNError(cudnnSetTensorNdDescriptor(m_dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

	size_t stateSize;
	void *states;
	unsigned long long seed = 4122ull; // Pick a seed.

	CheckCudNNError(cudnnDropoutGetStatesSize(m_handle, &stateSize));

	CheckCudaError(cudaMalloc(&states, stateSize));

	CheckCudNNError(cudnnSetDropoutDescriptor(m_dropoutDesc,
		m_handle,
		dropout,
		states,
		stateSize,
		seed));

	CheckCudNNError(cudnnSetRNNDescriptor(m_rnnDesc,
		m_hiddenSize,
		m_layerNumber,
		m_dropoutDesc,
		CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
		m_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
		CUDNN_LSTM,
		m_dataType));

	// -------------------------
	// Set up parameters
	// -------------------------
	// This needs to be done after the rnn descriptor is set as otherwise
	// we don't know how many parameters we have to allocate

	CheckCudNNError(cudnnCreateFilterDescriptor(&m_wDesc));
	CheckCudNNError(cudnnCreateFilterDescriptor(&m_dwDesc));

	CheckCudNNError(cudnnGetRNNParamsSize(m_handle, m_rnnDesc, m_xDesc[0], &m_weightsSize, CUDNN_DATA_FLOAT));

	int dimW[3];
	dimW[0] = m_weightsSize / sizeof(float);
	dimW[1] = 1;
	dimW[2] = 1;

	CheckCudNNError(cudnnSetFilterNdDescriptor(m_wDesc, m_dataType, m_tensorFormat, 3, dimW));
	CheckCudNNError(cudnnSetFilterNdDescriptor(m_dwDesc, m_dataType, m_tensorFormat, 3, dimW));

	CheckCudaError(cudaMalloc((void**)&m_w, m_weightsSize));
	CheckCudaError(cudaMalloc((void**)&m_dw, m_weightsSize));

	// -------------------------
	// Set up work space and reserved memory
	// -------------------------   
	// Need for every pass
	CheckCudNNError(cudnnGetRNNWorkspaceSize(m_handle, m_rnnDesc, m_seqLength, m_xDesc, &m_workSize));
	// Only needed in training, shouldn't be touched between passes.
	CheckCudNNError(cudnnGetRNNTrainingReserveSize(m_handle, m_rnnDesc, m_seqLength, m_xDesc, &m_reserveSize));

	CheckCudaError(cudaMalloc((void**)&m_workspace, m_workSize));
	CheckCudaError(cudaMalloc((void**)&m_reserveSpace, m_reserveSize));

	// *********************************************************************************************************
	// Initialise weights and inputs
	// *********************************************************************************************************

	CheckCudaError(cudaDeviceSynchronize());
}

Trainer::~Trainer() {
	
	CheckCudNNError(cudnnDestroy(m_handle));

	CheckCudNNError(cudnnDestroyRNNDescriptor(m_rnnDesc));

	CheckCublasError(cublasDestroy(m_cublasHandle));

	CheckCudaError(cudaSetDevice(m_gpuid));
	CheckCudaError(cudaDeviceReset());
}

void Trainer::train(float* data, int epochNumber)
{
	// Initialize weights
	int numLinearLayers = 8; // 2 for RELU/TANH, 8 for LSTM and 6 for GRU

	for (int layer = 0; layer < m_layerNumber * (m_bidirectional ? 2 : 1); layer++) {
		for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
			cudnnFilterDescriptor_t linLayerMatDesc;
			CheckCudNNError(cudnnCreateFilterDescriptor(&linLayerMatDesc));
			float *linLayerMat;

			CheckCudNNError(cudnnGetRNNLinLayerMatrixParams(m_handle,
				m_rnnDesc,
				layer,
				m_xDesc[0],
				m_wDesc,
				m_w,
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

			initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));

			CheckCudNNError(cudnnDestroyFilterDescriptor(linLayerMatDesc));

			cudnnFilterDescriptor_t linLayerBiasDesc;
			CheckCudNNError(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
			float *linLayerBias;

			CheckCudNNError(cudnnGetRNNLinLayerBiasParams(m_handle,
				m_rnnDesc,
				layer,
				m_xDesc[0],
				m_wDesc,
				m_w,
				linLayerID,
				linLayerBiasDesc,
				(void**)&linLayerBias));

			CheckCudNNError(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA));

			initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);

			CheckCudNNError(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
		}
	}
	CheckCudaError(cudaDeviceSynchronize());

	
	for (int iter = 0; iter < epochNumber; ++iter)
	{
		// Train
		//int imageid = iter % (train_size / context.m_batchSize);

		// Data initialization
		initGPUData((float*)m_x, m_seqLength * m_inputSize * m_batchSize, 1.f);
		if (m_hx != NULL) initGPUData((float*)m_hx, m_layerNumber *m_hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1), 1.f);
		if (m_cx != NULL) initGPUData((float*)m_cx, m_layerNumber * m_hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1), 1.f);

		// Label initialization
		initGPUData((float*)m_dy, m_seqLength * m_hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1), 1.f);
		if (m_dhy != NULL) initGPUData((float*)m_dhy, m_layerNumber * m_hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1), 1.f);
		if (m_dcy != NULL) initGPUData((float*)m_dcy, m_layerNumber * m_hiddenSize * m_batchSize * (m_bidirectional ? 2 : 1), 1.f);

		// Prepare current batch on device. Copy training img + label on cuda device
		/*checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
			sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
			sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));*/

		CheckCudaError(cudaDeviceSynchronize());

		cudaEvent_t start, stop;
		float timeForward, timeBackward1, timeBackward2;
		CheckCudaError(cudaEventCreate(&start));
		CheckCudaError(cudaEventCreate(&stop));

		CheckCudaError(cudaEventRecord(start));

		CheckCudNNError(cudnnRNNForwardTraining(m_handle,
			m_rnnDesc,
			m_seqLength,
			m_xDesc,
			m_x,
			m_hxDesc,
			m_hx,
			m_cxDesc,
			m_cx,
			m_wDesc,
			m_w,
			m_yDesc,
			m_y,
			m_hyDesc,
			m_hy,
			m_cyDesc,
			m_cy,
			m_workspace,
			m_workSize,
			m_reserveSpace,
			m_reserveSize));

		CheckCudaError(cudaEventRecord(stop));
		CheckCudaError(cudaEventSynchronize(stop));
		CheckCudaError(cudaEventElapsedTime(&timeForward, start, stop));

		CheckCudaError(cudaEventRecord(start));

		CheckCudNNError(cudnnRNNBackwardData(m_handle,
			m_rnnDesc,
			m_seqLength,
			m_yDesc,
			m_y,
			m_dyDesc,
			m_dy,
			m_dhyDesc,
			m_dhy,
			m_dcyDesc,
			m_dcy,
			m_wDesc,
			m_w,
			m_hxDesc,
			m_hx,
			m_cxDesc,
			m_cx,
			m_dxDesc,
			m_dx,
			m_dhxDesc,
			m_dhx,
			m_dcxDesc,
			m_dcx,
			m_workspace,
			m_workSize,
			m_reserveSpace,
			m_reserveSize));

		CheckCudaError(cudaEventRecord(stop));
		CheckCudaError(cudaEventSynchronize(stop));
		CheckCudaError(cudaEventElapsedTime(&timeBackward1, start, stop));

		CheckCudaError(cudaEventRecord(start));

		// cudnnRNNBackwardWeights adds to the data in dw.
		CheckCudaError(cudaMemset(m_dw, 0, m_weightsSize));

		CheckCudNNError(cudnnRNNBackwardWeights(m_handle,
			m_rnnDesc,
			m_seqLength,
			m_xDesc,
			m_x,
			m_hxDesc,
			m_hx,
			m_yDesc,
			m_y,
			m_workspace,
			m_workSize,
			m_dwDesc,
			m_dw,
			m_reserveSpace,
			m_reserveSize));

		CheckCudaError(cudaEventRecord(stop));

		CheckCudaError(cudaEventSynchronize(stop));
		CheckCudaError(cudaEventElapsedTime(&timeBackward2, start, stop));

		// Updating learningRate and weight thx to gradient
		//float learningRate = static_cast<float>(m_learningRate * pow((1.0 + m_learningRateGamma * iter), (-m_learningRatePower)));
		// weights[j] = weights[j] + a * (ans[i] - y) * examples[i][j];
		// weight = weight + learningRate * (expected - observed) * sample;

		printf("Forward (%3.0f ms)\n", timeForward);
		printf("Backward (%3.0f ms)\n", timeBackward1);
		printf("Backward Weight (%3.0f ms)\n", timeBackward2);
	}

	

	CheckCudaError(cudaDeviceSynchronize());
}

void Trainer::initGPUData(float * data, int nbData, float value)
{
	for (int i = 0; i < nbData; i++) {
		data[i] = value;
	}
}
