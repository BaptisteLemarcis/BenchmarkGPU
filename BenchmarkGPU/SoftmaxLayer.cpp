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

std::tuple<float, float*> SoftmaxLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* input, float** targets, float* d_onevec)
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

float* SoftmaxLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* dloss_data, float** targets, float* d_onevec) {
	// Softmax layer
	//SoftmaxLossBackprop <<< RoundUp(m_batchSize, m_batchSize), m_batchSize >>> (labels, m_output, m_batchSize, dloss_data);

	cudnnTensorDescriptor_t diffTensorDesc;
	cudnnCreateTensorDescriptor(&diffTensorDesc);
	int dimA[3];
	int strideA[3];
	dimA[0] = m_batchSize;
	dimA[1] = m_inputDim;
	dimA[2] = 1;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;
	cudnnSetTensorNdDescriptor(diffTensorDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA);

	float *loss_data;
	cudaMalloc((void**)&loss_data, m_batchSize*m_inputDim * sizeof(float));
	cudaMemcpy(loss_data, dloss_data, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyDeviceToHost);

	float* diffData = new float[m_batchSize*m_inputDim];
	for (int b = 0; b < m_batchSize; b++) {
		float* target = targets[b];
		for (int i = 0; i < m_inputDim; i++) {
			if(target[i] == 1.f)
				loss_data[b * m_inputDim + m_inputDim] -= 1.f;
		}
	}

	float *d_diffData, *d_gradData;
	cudaMalloc((void**)&d_diffData, m_batchSize*m_inputDim * sizeof(float));
	cudaMalloc((void**)&d_gradData, m_batchSize*m_inputDim * sizeof(float));
	cudaMemcpy(d_diffData, diffData, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyHostToDevice);
	
	
	cudaDeviceSynchronize();
	
	float alpha(1), beta(0);

	cudnnSoftmaxBackward(handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		diffTensorDesc,
		dloss_data,
		diffTensorDesc,
		d_diffData,
		&beta,
		m_outputDesc,
		d_gradData);

	return d_gradData;
}

void SoftmaxLayer::initWeights(cudnnHandle_t &)
{

}

void SoftmaxLayer::initEpoch(cudnnHandle_t & handle)
{
}
