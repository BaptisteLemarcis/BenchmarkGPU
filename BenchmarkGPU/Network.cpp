#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <random>
#include <thread>
#include <sstream>
#include <vector>
#include <tuple>

#include "Layer.h"
#include "Network.h"
#include "Logger.h"
#include "GenericFunctions.h"

Network::Network() : Network(128, 0.01f, 4, 1) {}

Network::Network(int batchSize, float learningRate, int inputSize, int seqLength = 50) {
	m_batchSize = batchSize;
	m_learningRate = learningRate;
	m_inputDim = inputSize;
	m_seqLength = seqLength;

	//
	// Listing GPU Devices
	//
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	int gpuNumbers;
	CheckCudaError(cudaGetDeviceCount(&gpuNumbers));
	int i = 0;
	cudaDeviceProp prop;
	toWrite << "Cuda capable devices " << gpuNumbers << ":" << std::endl;
	for (i = 0; i < gpuNumbers; i++) {
		CheckCudaError(cudaGetDeviceProperties(&prop, i));
		toWrite << "\tdevice " << i << " (" << prop.name << ") : Proc " << prop.multiProcessorCount << ", Capabilities " << prop.major << "." << prop.minor << ", SmClock " << (float)prop.clockRate*1e-3 << " Mhz" << ", MemSize(Mb) " << (int)(prop.totalGlobalMem / (1024 * 1024)) << ", MemClock " << (float)prop.memoryClockRate*1e-3 << " Mhz" << std::endl;
	}
	m_gpuid = 0;

	//
	// Setting CUDA device
	//
	std::cout << "Using device " << m_gpuid << std::endl;
	toWrite << "Using device " << m_gpuid << std::endl;

	CheckCudaError(cudaSetDevice(m_gpuid));

	//
	// Getting CudNN version
	//

	size_t version = cudnnGetVersion();
	std::cout << "CudNN version " << version << std::endl;
	toWrite << "CudNN version " << version << std::endl;

	//
	// Create CuDNN Handler
	//
	CheckCudNNError(cudnnCreate(&m_handle));

	//
	// Create Cublas Handler
	//
	CheckCublasError(cublasCreate(&m_cublasHandle));
	Logger::instance()->writeLine(toWrite.str());
}

Network::~Network() {
	CheckCudNNError(cudnnDestroy(m_handle));
	CheckCublasError(cublasDestroy(m_cublasHandle));
	CheckCudaError(cudaSetDevice(m_gpuid));
	CheckCudaError(cudaDeviceReset());
}

void Network::train(float* data, float** labels, int epochNumber, int nbData)
{
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "==========================================================" << std::endl;
	toWrite << "=========================Training=========================" << std::endl;
	toWrite << "==========================================================" << std::endl;
	Logger::instance()->writeLine(toWrite.str());

	std::cout << "Training..." << std::endl;

	float timeTraining;
	cudaEvent_t start, stop;
	CheckCudaError(cudaEventCreate(&start));
	CheckCudaError(cudaEventCreate(&stop));
	CheckCudaError(cudaEventRecord(start));

	//
	//  Initialize weights of each layer
	//
	for (Layer& l : m_layers)
		l.initWeights(m_handle);

	CheckCudaError(cudaDeviceSynchronize());

	int nbBatch = std::ceil(double(nbData) / double(m_batchSize));
	toWrite.str("");
	toWrite.clear();
	toWrite << "Number of iteration per epoch : " << nbBatch << std::endl;
	Logger::instance()->writeLine(toWrite.str());
	std::cout << nbBatch << " batchs to run per iteration" << std::endl;

	for (int iter = 0; iter < epochNumber; iter++)
	{
		trainEpoch(iter, epochNumber, nbBatch, nbData, data, labels);
	}

	CheckCudaError(cudaDeviceSynchronize());
	CheckCudaError(cudaEventRecord(stop));
	CheckCudaError(cudaEventSynchronize(stop));
	CheckCudaError(cudaEventElapsedTime(&timeTraining, start, stop));

	toWrite.str("");
	toWrite.clear();
	toWrite << "==========================================================" << std::endl;
	toWrite << "=======================End Training=======================" << std::endl;
	toWrite << "==========================================================" << std::endl;
	toWrite << "Time " << timeTraining << "ms";
	Logger::instance()->writeLine(toWrite.str());
	printf("Training time (%3.0f ms)\n", timeTraining);
}

void Network::addLayer(Layer& l)
{
	m_layers.push_back(l);
}

cudnnHandle_t & Network::getHandle()
{
	return m_handle;
}

void Network::trainEpoch(int epoch, int nbEpoch, int nbBatch, int nbData, float* input, float** targets)
{
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	std::cout << "Epoch " << (epoch + 1) << " / " << nbEpoch << std::endl;
	toWrite << "==========================================================" << std::endl;
	Logger::instance()->writeLine(toWrite.str());
	int curNbBatch = 0;
	float error = 0.f;

	//
	//  Initialize weights of each layer
	//
	for (Layer& l : m_layers)
		l.initEpoch(m_handle);

	for (int b = 0; b < nbBatch; b++) {
		float *bData, **bTargets;
		bData = new float[m_inputDim * m_batchSize];
		bTargets = new float*[m_inputDim * m_batchSize];
		CheckCudaError(cudaMalloc((void**)&bData, m_seqLength * m_inputDim * m_batchSize * sizeof(float)));

		// Get data and target for current batch
		prepareData(input, targets, b, bData, bTargets);

		curNbBatch += m_batchSize;
		toWrite.str("");
		toWrite.clear();
		toWrite << "Epoch " << (epoch + 1) << " / " << nbEpoch << " [" << curNbBatch << " / " << nbData << "]";
		Logger::instance()->writeLine(toWrite.str());

		// Forward pass + get error
		auto localError = forward(bData, bTargets);

		toWrite.str("");
		toWrite.clear();
		toWrite << "\tError " << localError;
		Logger::instance()->writeLine(toWrite.str());

		error += localError;
		//softmaxLayer(bData);
		/*
		//
		//	Updating gradient
		//
		float norm =  2. / m_hiddenSize;
		float* gradOut = new float[m_seqLength * m_hiddenSize * m_batchSize];
		float* out = new float[m_seqLength * m_hiddenSize * m_batchSize];
		cudaMemcpy(gradOut, m_gradientOutput, m_batchSize*m_dataTypeSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(out, m_dstData, m_batchSize*m_dataTypeSize, cudaMemcpyDeviceToHost);

		for (int i = 0; i < m_hiddenSize; i++) {
			gradOut[i] = norm * (out[i]- bTargets[i]);
		}

		cudaMemcpy(m_gradientOutput, gradOut, m_batchSize*m_dataTypeSize, cudaMemcpyHostToDevice);*/

		//backward();
		//validateBatch();
	}
	error /= nbBatch;
	toWrite.str("");
	toWrite.clear();
	toWrite << "Batch Error " << error;
	Logger::instance()->writeLine(toWrite.str());

	/*float* tmpWGrad = new float[m_weightsSize / m_dataTypeSize];
	float* tmpW = new float[m_weightsSize / m_dataTypeSize];
	CheckCudaError(cudaMemcpy(tmpWGrad, m_weightsGradient, m_weightsSize, cudaMemcpyDeviceToHost));
	CheckCudaError(cudaMemcpy(tmpW, m_weights, m_weightsSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < m_weightsSize / m_dataTypeSize; i++) {
		tmpWGrad[i] = -m_learningRate*(m_batchLoss / tmpWGrad[i]);
	}
	CheckCudaError(cudaMemcpy(m_weightsGradient, tmpWGrad, m_weightsSize, cudaMemcpyHostToDevice));*/
}

float Network::forward(float* input, float** target)
{
	float error = 0;
	std::vector<std::reference_wrapper<Layer>>::iterator it = m_layers.begin();
	float* output;
	auto result = it->get().forward(m_handle, m_cublasHandle, input, target);
	output = std::get<1>(result);
	it++;
	for (; it != m_layers.end(); ++it) {
		result = it->get().forward(m_handle, m_cublasHandle, output, target);
		output = std::get<1>(result);
		error += std::get<0>(result);
	}
	return error;
}

void Network::backward(float* input)
{
	
}

void Network::prepareData(float* input, float** target, int b, float* d_batchData, float** bTarget) {
	//
	//	Loading data for this batch
	//
	int srcSize = m_inputDim * m_batchSize;
	float* batchData = new float[srcSize];

	for (int i = b*m_inputDim, j = 0; i < (b + m_batchSize)*m_inputDim; i++, j++) {
		batchData[j] = input[i];
		bTarget[j] = target[i];
	}
	cudaMemcpy(d_batchData, batchData, m_batchSize*m_inputDim * sizeof(float), cudaMemcpyHostToDevice);
	delete[] batchData;
}
