#include <tuple>
#include <cuda.h>
#include <cudnn.h>
#include "SoftmaxLayer.h"
#include "GenericFunctions.h"

SoftmaxLayer::SoftmaxLayer(int inputDim, int batchSize) : Layer(inputDim, inputDim, batchSize)
{
	//
	// Create Descriptor
	//
	CheckCudNNError(cudnnCreateTensorDescriptor(&m_outputDesc));

	// Allocating
	CheckCudaError(cudaMalloc(&m_output, m_inputDim*batchSize * sizeof(float)));

	//
	// Setting up TensorDescriptor
	//
	int dimA[3];
	int strideA[3];
	dimA[0] = batchSize;
	dimA[1] = inputDim;
	dimA[2] = 1;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	CheckCudNNError(cudnnSetTensorNdDescriptor(m_outputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

	CheckCudaError(cudaDeviceSynchronize());
}


SoftmaxLayer::~SoftmaxLayer()
{
	CheckCudaError(cudaFree(m_output));
	CheckCudNNError(cudnnDestroyTensorDescriptor(m_outputDesc));
}

std::tuple<float, float*> SoftmaxLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* input, float** targets)
{
	float alpha = float(1);
	float beta = float(0);

	CheckCudNNError(cudnnSoftmaxForward(handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_INSTANCE,
		&alpha,
		m_outputDesc, // m_dstDataDesc[0]
		input,
		&beta,
		m_outputDesc,
		m_output));

	float batchError = 0.f;
	for (int i = 0; i < m_batchSize; i++) {
		float* output = new float[getOutputDim() * m_batchSize];
		float* target = targets[i];
		CheckCudaError(cudaMemcpy(output, input, getOutputDim() * m_batchSize * sizeof(float), cudaMemcpyDeviceToHost));
		float sum = 0.f;
		for (int i = 0; i < getOutputDim(); i++) {
			float z = output[i] - target[i];
			sum += z*z;
		}
		sum /= getOutputDim();
		batchError += sum;
	}
	batchError /= m_batchSize;

	/*if (FLAGS_DEBUG) {
		logFile << "\tResulting softmax data" << std::endl;
		printDeviceVectorToFile(NBDATADSP, (float *)m_smaxData, 0);
	}*/

	return std::make_tuple(batchError, m_output);
}

void SoftmaxLayer::backward(cudnnHandle_t& handle, float* input) {
	
}

void SoftmaxLayer::initWeights(cudnnHandle_t &)
{

}

void SoftmaxLayer::initEpoch(cudnnHandle_t & handle)
{
}
