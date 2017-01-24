#include <iostream>
#include <string.h>
#include <cudnn.h>

#include "Trainer.h"
#include "GenericFunctions.h"

Trainer::Trainer() : Trainer(0, 128, 0.01, 500) {}

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
	std::cout << "GPU List:" << std::endl;
	for (i = 0; i < gpuNumbers; i++) {
		CheckCudaError(cudaGetDeviceProperties(&prop, i));
		std::cout << "\t" << prop.name << "(" << i << ")" << std::endl;
	}
	
	//
	// Setting Cuda device
	//

	std::cout << "Setting CudaDevice to " << m_gpuid << std::endl;
	CheckCudaError(cudaSetDevice(m_gpuid));
	std::cout << "CudaDevice set." << std::endl;

	//
	// Getting CudNN version
	//

	size_t version = cudnnGetVersion();
	std::cout << "Running on CudNN version : " << version << std::endl;

	//
	// Create Descriptor
	//

	std::cout << "Creating TensorDescriptor " << std::endl;
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_dataTensor));

	CheckCudNNError(cudnnCreateRNNDescriptor(&m_rnnDesc));

	CheckCudNNError(cudnnCreate(&m_handle));
}

Trainer::~Trainer() {
	CheckCudaError(cudaSetDevice(m_gpuid));
	CheckCudNNError(cudnnDestroy(m_handle));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_dataTensor));
}

void Trainer::forwardTraining(int seqLength, float* data, float* output, void* trainingSpace, void* workspace)
{
	cudnnRNNForwardTraining(m_handle,
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
		m_trainingSize);
}
