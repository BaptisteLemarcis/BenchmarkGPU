#include <iostream>
#include <string.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>

#include "Trainer.h"
#include "GenericFunctions.h"

Trainer::Trainer() : Trainer(0, 128, 0.01f, 500) {}

Trainer::Trainer(int GPUID, int batchSize, float learningRate, int epochNumber) {
	m_gpuid = GPUID;
	m_batchSize = batchSize;
	m_learningRate = learningRate;
	m_epochNumber = epochNumber;

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

	//
	// Create Descriptor
	//

	std::cout << "Creating TensorDescriptor " << std::endl;
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_srcTensorDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dstTensorDesc));
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_biasTensorDesc));

	CheckCudNNError(cudnnCreateRNNDescriptor(&m_rnnDesc));
	
	CheckCudNNError(cudnnCreateFilterDescriptor(&m_filterDesc));

	CheckCudNNError(cudnnCreateActivationDescriptor(&m_activDesc));

	//
	// Setting up TensorDescriptor
	//



	CheckCudNNError(cudnnSetActivationDescriptor(m_activDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));
}

Trainer::~Trainer() {
	
	CheckCudNNError(cudnnDestroy(m_handle));

	CheckCudNNError(cudnnDestroyTensorDescriptor(m_srcTensorDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_dstTensorDesc));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_biasTensorDesc));

	CheckCudNNError(cudnnDestroyRNNDescriptor(m_rnnDesc));

	CheckCudNNError(cudnnDestroyFilterDescriptor(m_filterDesc));

	CheckCudNNError(cudnnDestroyActivationDescriptor(m_activDesc));

	CheckCublasError(cublasDestroy(m_cublasHandle));

	CheckCudaError(cudaSetDevice(m_gpuid));
	cudaDeviceReset();
}

void Trainer::forwardTraining(int seqLength, float* data, float* output, void* trainingSpace, void* workspace)
{
	/*cudnnRNNForwardTraining(m_handle,
		m_rnnDesc,
		seqLength,
		&m_dataTensor,
		data,
		hxDesc,
		&hx,
		cxDesc,
		&cx,
		NULL, // filter
		NULL,
		&m_outputTensor,
		output,
		hyDesc,
		&hy,
		cyDesc,
		&cy,
		workspace,
		m_workspaceSize,
		trainingSpace,
		m_trainingSize);*/
}
