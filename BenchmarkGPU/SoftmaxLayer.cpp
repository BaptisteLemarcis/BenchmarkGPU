#include <tuple>
#include <cuda.h>
#include <cudnn.h>

#include "Logger.h"
#include "SoftmaxLayer.h"
#include "GenericFunctions.h"
#include "GPUKernel.cuh"

SoftmaxLayer::SoftmaxLayer(int inputDim, int batchSize) : Layer(inputDim, inputDim, batchSize)
{
	//
	// Create Descriptor
	//
	CheckError(cudnnCreateTensorDescriptor(&m_outputDesc), __FILE__, __LINE__);

	//
	// Setting up TensorDescriptor
	//
	int dimA[3];
	int strideA[3];
	dimA[0] = batchSize;
	dimA[1] = m_outputDim;
	dimA[2] = 1;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	CheckError(cudnnSetTensorNdDescriptor(m_outputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);

	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	/*delete[] dimA;
	delete[] strideA;*/
}

SoftmaxLayer::~SoftmaxLayer()
{
	//CheckError(cudaFree(&m_d_output), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_outputDesc), __FILE__, __LINE__);
}

std::tuple<float, float*> SoftmaxLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_input, float* d_targets, float* d_onevec, bool training)
{
	float alpha = float(1);
	float beta = float(0);

	// Allocating
	CheckError(cudaMalloc(&m_d_output, m_outputDim*m_batchSize * sizeof(float)), __FILE__, __LINE__);

	CheckError(cudnnSoftmaxForward(handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_INSTANCE,
		&alpha,
		m_outputDesc, // m_dstDataDesc[0]
		d_input,
		&beta,
		m_outputDesc,
		m_d_output), __FILE__, __LINE__);

	float* targets = new float[m_batchSize*m_outputDim];
	float* output = new float[m_batchSize*m_outputDim];
	float batchError = 0.f;
	CheckError(cudaMemcpy(targets, d_targets, m_batchSize*m_outputDim*sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	CheckError(cudaMemcpy(output, m_d_output, m_batchSize*m_outputDim*sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	for (int i = 0; i < m_batchSize; i++) {
		for (int j = 0; j < getOutputDim(); j++) {
			//float realOutput = output[i*getOutputDim() + j] < 0.5 ? 0.f : 1.f;
			float z = targets[i*getOutputDim() + j] - output[i*getOutputDim() + j];
			batchError += z*z;
		}
	}
	batchError /= m_batchSize;

	return std::make_tuple(batchError, m_d_output);
}

float* SoftmaxLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_loss_data, float* d_targets, float* d_onevec, float* previousLayerOutput) {
	// Softmax layer
	float *d_diffData, *d_gradData;

	cudnnTensorDescriptor_t diffTensorDesc;
	CheckError(cudnnCreateTensorDescriptor(&diffTensorDesc), __FILE__, __LINE__);
	int dimA[3];
	int strideA[3];
	dimA[0] = m_batchSize;
	dimA[1] = m_inputDim;
	dimA[2] = 1;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;
	CheckError(cudnnSetTensorNdDescriptor(diffTensorDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	CheckError(cudaMalloc((void**)&d_diffData, m_batchSize*m_inputDim * sizeof(float)), __FILE__, __LINE__);
	softmaxLoss(m_batchSize, m_inputDim, d_targets, d_loss_data);

	float scalVal = 1.0f / static_cast<float>(m_batchSize);
	CheckError(cublasSscal(cublasHandle, m_inputDim * m_batchSize, &scalVal, d_loss_data, 1), __FILE__, __LINE__);
	
	
	CheckError(cudaMalloc((void**)&d_gradData, m_batchSize*m_inputDim * sizeof(float)), __FILE__, __LINE__);
	
	float alpha(1), beta(0);

	CheckError(cudnnSoftmaxBackward(handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		diffTensorDesc,
		d_loss_data,
		diffTensorDesc,
		d_diffData,
		&beta,
		m_outputDesc,
		d_gradData), __FILE__, __LINE__);
	return d_loss_data;
}

void SoftmaxLayer::initWeights(cudnnHandle_t &)
{

}

void SoftmaxLayer::initEpoch(cudnnHandle_t & handle)
{
}

void SoftmaxLayer::updateWeight(cublasHandle_t&, float)
{
}
