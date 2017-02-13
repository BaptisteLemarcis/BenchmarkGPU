#include <tuple>
#include <cuda.h>
#include <cudnn.h>
#include <device_launch_parameters.h>

#include "Logger.h"
#include "SoftmaxLayer.h"
#include "GenericFunctions.h"

__global__ void SoftmaxLossKernel(float* d_target, float* d_loss_data, int size) {
	//int idxTarget = (blockDim.x * blockIdx.x + threadIdx.x) + blockIdx.y;
	int idx = ((blockDim.x * blockIdx.x + threadIdx.x) * blockDim.y) + blockIdx.y;

	if (idx >= size) return;

	if (d_target[idx] == 1.f)
		d_loss_data[idx] -= 1.f;
}

__global__ void SoftmaxErrorKernel(float* d_error, float* d_input, float* d_targets, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idx1 = 2 * idx;
	int idx2 = (2 * (idx+1))-1;

	/*int idx3 = 2 * (idx+1);
	int idx4 = (2 * ((idx+1) + 1)) - 1;*/

	if (idx <= size) return;
	float z = d_input[idx1] - d_targets[idx1];
	*d_error += z*z;
	z = d_input[idx2] - d_targets[idx2];
	*d_error += z*z;
}

SoftmaxLayer::SoftmaxLayer(int inputDim, int batchSize) : Layer(inputDim, inputDim, batchSize)
{
	//
	// Create Descriptor
	//
	CheckError(cudnnCreateTensorDescriptor(&m_outputDesc), __FILE__, __LINE__);

	// Allocating
	CheckError(cudaMalloc(&m_d_output, m_outputDim*batchSize * sizeof(float)), __FILE__, __LINE__);

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
	CheckError(cudaFree(&m_d_output), __FILE__, __LINE__);
	CheckError(cudnnDestroyTensorDescriptor(m_outputDesc), __FILE__, __LINE__);
}

std::tuple<float, float*> SoftmaxLayer::forward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_input, float* d_targets, float* d_onevec)
{
	float alpha = float(1);
	float beta = float(0);

	CheckError(cudnnSoftmaxForward(handle,
		CUDNN_SOFTMAX_ACCURATE,
		CUDNN_SOFTMAX_MODE_INSTANCE,
		&alpha,
		m_outputDesc, // m_dstDataDesc[0]
		d_input,
		&beta,
		m_outputDesc,
		m_d_output), __FILE__, __LINE__);

	//printDeviceVectorToFile(2, m_output, 0);
	float* d_batchError, batchError;
	CheckError(cudaMalloc(&d_batchError, sizeof(float)), __FILE__, __LINE__);

	SoftmaxErrorKernel <<<RoundUp(m_batchSize, 128), 128 >>> (d_batchError, d_input, d_targets, m_batchSize);

	CheckError(cudaMemcpy(&batchError, d_batchError, sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	
	/*for (int i = 0; i < m_batchSize; i++) {
		float* output = new float[getOutputDim() * m_batchSize];
		float* target = targets[i];
		CheckError(cudaMemcpy(output, input, getOutputDim() * m_batchSize * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		float sum = 0.f;
		for (int i = 0; i < getOutputDim(); i++) {
			float z = output[i] - target[i];
			sum += z*z;
		}
		sum /= getOutputDim();
		batchError += sum;
	}*/
	batchError /= m_batchSize;

	/*if (FLAGS_DEBUG) {
		logFile << "\tResulting softmax data" << std::endl;
		printDeviceVectorToFile(NBDATADSP, (float *)m_smaxData, 0);
	}*/

	return std::make_tuple(batchError, m_d_output);
}

float* SoftmaxLayer::backward(cudnnHandle_t& handle, cublasHandle_t& cublasHandle, float* d_loss_data, float* d_targets, float* d_onevec, float* previousLayerOutput) {
	// Softmax layer
	//SoftmaxLossBackprop <<< RoundUp(m_batchSize, m_batchSize), m_batchSize >>> (labels, m_output, m_batchSize, dloss_data);
	float *d_diffData, *d_gradData;
	Logger::instance()->writeLine("Softmax bwd");

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

	/*float *loss_data = new float [m_batchSize*m_inputDim];
	//CheckError(cudaMalloc((void**)&loss_data, m_batchSize*m_inputDim * sizeof(float)), __FILE__, __LINE__);
	CheckError(cudaMemcpy(loss_data, dloss_data, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);*/
	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	/*printDeviceVectorToFile(m_batchSize*m_inputDim, dloss_data, 0);
	printVectorToFile(m_batchSize*m_inputDim, loss_data, 0);*/

	dim3 b = { (unsigned int)RoundUp(m_batchSize,128), (unsigned int)m_inputDim, 1};
	dim3 t = { 128 , 1, 1 };

	CheckError(cudaMalloc((void**)&d_diffData, m_batchSize*m_inputDim * sizeof(float)), __FILE__, __LINE__);

	SoftmaxLossKernel<<<b, t >>>(d_targets, d_loss_data, m_batchSize*m_inputDim);
	/*float* diffData = new float[m_batchSize*m_inputDim];
	for (int b = 0; b < m_batchSize; b++) {
		float* target = targets[b];
		for (int i = 0; i < m_inputDim; i++) {
			if(target[i] == 1.f)
				loss_data[b * m_inputDim + i] -= 1.f;
		}
	}*/

	//CheckError(cudaMemcpy(dloss_data, loss_data, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	float scalVal = 1.0f / static_cast<float>(m_batchSize);
	CheckError(cublasSscal(cublasHandle, m_inputDim * m_batchSize, &scalVal, d_loss_data, 1), __FILE__, __LINE__);
	
	/*Logger::instance()->writeLine("Output");
	printDeviceVectorToFile(2, m_output, 0);*/
	
	CheckError(cudaMalloc((void**)&d_gradData, m_batchSize*m_inputDim * sizeof(float)), __FILE__, __LINE__);
	//CheckError(cudaMemcpy(d_diffData, diffData, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	
	
	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	
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

	Logger::instance()->writeLine("d_loss_data");
	printDeviceVectorToFile(10, d_loss_data, 0);

	Logger::instance()->writeLine("d_diffData");
	printDeviceVectorToFile(10, d_diffData, 0);

	/*Logger::instance()->writeLine("\tdloss_data ======== 1");
	printDeviceVectorToFile(10, d_loss_data, 0);*/

	/*delete[] dimA;
	delete[] strideA;*/

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
