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
#include "GPUKernel.cuh"

Network::Network() : Network(128, 0.01f, 4, 1, 1) {}

Network::Network(int batchSize, float learningRate, int inputSize, int outputDim, int seqLength = 50) : m_batchSize(batchSize), m_learningRate(learningRate), m_inputDim(inputSize), m_seqLength(seqLength), m_outputDim(outputDim){

	//
	// Listing GPU Devices
	//
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	int gpuNumbers;
	CheckError(cudaGetDeviceCount(&gpuNumbers), __FILE__, __LINE__);
	int i = 0;
	cudaDeviceProp prop;
	toWrite << "Cuda capable devices " << gpuNumbers << ":" << std::endl;
	for (i = 0; i < gpuNumbers; i++) {
		CheckError(cudaGetDeviceProperties(&prop, i), __FILE__, __LINE__);
		toWrite << "\tdevice " << i << " (" << prop.name << ") : Proc " << prop.multiProcessorCount << ", Capabilities " << prop.major << "." << prop.minor << ", SmClock " << (float)prop.clockRate*1e-3 << " Mhz" << ", MemSize(Mb) " << (int)(prop.totalGlobalMem / (1024 * 1024)) << ", MemClock " << (float)prop.memoryClockRate*1e-3 << " Mhz" << std::endl;
	}
	m_gpuid = 0;

	//
	// Setting CUDA device
	//
	std::cout << "Using device " << m_gpuid << std::endl;
	toWrite << "Using device " << m_gpuid << std::endl;

	CheckError(cudaSetDevice(m_gpuid), __FILE__, __LINE__);

	//
	// Getting CudNN version
	//

	size_t version = cudnnGetVersion();
	std::cout << "CudNN version " << version << std::endl;
	toWrite << "CudNN version " << version << std::endl;

	//
	// Create CuDNN Handler
	//
	CheckError(cudnnCreate(&m_handle), __FILE__, __LINE__);

	//
	// Create Cublas Handler
	//
	CheckError(cublasCreate(&m_cublasHandle), __FILE__, __LINE__);
	Logger::instance()->writeLine(toWrite.str());
}

Network::~Network() {
	CheckError(cudnnDestroy(m_handle), __FILE__, __LINE__);
	CheckError(cublasDestroy(m_cublasHandle), __FILE__, __LINE__);
	CheckError(cudaSetDevice(m_gpuid), __FILE__, __LINE__);
	CheckError(cudaDeviceReset(), __FILE__, __LINE__);
}

void Network::train(float* d_data, float* d_labels, int epochNumber, int nbData)
{
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "==========================================================" << std::endl;
	toWrite << "=========================Training=========================" << std::endl;
	toWrite << "==========================================================" << std::endl;
	Logger::instance()->writeLine(toWrite.str());

	std::cout << "Training..." << std::endl;

	float timeTraining;
	cudaEvent_t start, stop;
	CheckError(cudaEventCreate(&start), __FILE__, __LINE__);
	CheckError(cudaEventCreate(&stop), __FILE__, __LINE__);
	CheckError(cudaEventRecord(start), __FILE__, __LINE__);

	//
	//  Initialize weights of each layer
	//
	for (Layer& l : m_layers)
		l.initWeights(m_handle);

	//CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	int nbBatch = std::ceil(double(nbData) / double(m_batchSize));
	toWrite.str("");
	toWrite.clear();
	toWrite << "Number of iteration per epoch : " << nbBatch << std::endl;
	Logger::instance()->writeLine(toWrite.str());
	std::cout << nbBatch << " batchs to run per iteration" << std::endl;

	for (int iter = 0; iter < epochNumber; iter++)
	{
		trainEpoch(iter, epochNumber, nbBatch, nbData, d_data, d_labels);
	}

	CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	CheckError(cudaEventRecord(stop), __FILE__, __LINE__);
	CheckError(cudaEventSynchronize(stop), __FILE__, __LINE__);
	CheckError(cudaEventElapsedTime(&timeTraining, start, stop), __FILE__, __LINE__);

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

void Network::trainEpoch(int epoch, int nbEpoch, int nbBatch, int nbData, float* d_input, float* d_targets)
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
		float *d_bData, *d_bTargets, *d_onevec;
		/*bData = new float[m_inputDim * m_batchSize];
		bTargets = new float*[m_inputDim * m_batchSize];*/
		CheckError(cudaMalloc((void**)&d_bData, m_seqLength * m_inputDim * m_batchSize * sizeof(float)), __FILE__, __LINE__);
		CheckError(cudaMalloc((void**)&d_bTargets, 2 * m_seqLength * m_inputDim * m_batchSize * sizeof(float)), __FILE__, __LINE__);

		// Get data and target for current batch
		prepareData(d_input, d_targets, b, d_bData, d_bTargets, m_inputDim, m_batchSize);

		curNbBatch += m_batchSize;
		toWrite.str("");
		toWrite.clear();
		toWrite << "Epoch " << (epoch + 1) << " / " << nbEpoch << " [" << curNbBatch << " / " << nbData << "]";
		Logger::instance()->writeLine(toWrite.str());

		CheckError(cudaMalloc(&d_onevec, sizeof(float) * m_batchSize), __FILE__, __LINE__);
		FillVec(m_batchSize, d_onevec, 0.f);

		std::vector<float*> d_output;

		// Forward pass + get error
		auto resultFwd = forward(d_bTargets, d_bTargets, d_onevec, &d_output);

		CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		//Backward pass
		backward(d_output, d_bTargets, d_onevec);

		CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		// Updating weights
		for (auto it = m_layers.begin(); it != m_layers.end(); ++it) {
			it->get().updateWeight(m_cublasHandle, m_learningRate);
		}

		toWrite.str("");
		toWrite.clear();
		toWrite << "\tBatch Error " << resultFwd;
		Logger::instance()->writeLine(toWrite.str());

		error += resultFwd;
		
		//validateBatch();
		CheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		/*for (auto o : d_output) {
			CheckError(cudaFree(o), __FILE__, __LINE__);
		}

		CheckError(cudaFree(d_bData), __FILE__, __LINE__);
		CheckError(cudaFree(d_bTargets), __FILE__, __LINE__);
		CheckError(cudaFree(d_onevec), __FILE__, __LINE__);*/
	}
	error /= nbBatch;
	toWrite.str("");
	toWrite.clear();
	toWrite << "Epoch average Error " << error;
	Logger::instance()->writeLine(toWrite.str());
}

float Network::forward(float* d_input, float* d_target, float* d_onevec, std::vector<float*>* d_output)
{
	float error = 0;
	std::vector<std::reference_wrapper<Layer>>::iterator it = m_layers.begin();
	
	std::tuple<float, float*> result = it->get().forward(m_handle, m_cublasHandle, d_input, d_target, d_onevec);
	d_output->push_back(std::get<1>(result));
	it++;
	for (; it != m_layers.end(); ++it) {
		result = it->get().forward(m_handle, m_cublasHandle, d_output->back(), d_target, d_onevec);
		d_output->push_back(std::get<1>(result));
		error += std::get<0>(result);
	}
	return error;
}

void Network::backward(std::vector<float*>& fwdOutput, float* target, float* d_onevec)
{
	float* d_loss_data;
	CheckError(cudaMalloc(&d_loss_data, sizeof(float) * m_batchSize * m_outputDim), __FILE__, __LINE__);
	CheckError(cudaMemcpy(d_loss_data, fwdOutput.back(), sizeof(float) * m_batchSize * m_outputDim, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	
	// Accounting for batch size in SGD
	//CheckError(cublasSscal(m_cublasHandle, m_outputDim * m_batchSize, &scalVal, dloss_data, 1), __FILE__, __LINE__);
	std::vector<std::reference_wrapper<Layer>>::reverse_iterator it = m_layers.rbegin();
	float* result = it->get().backward(m_handle, m_cublasHandle, d_loss_data, target, d_onevec, fwdOutput[fwdOutput.size() - 1]);

	++it;

	for (int i = fwdOutput.size() - 2; it != m_layers.rend(); ++it, i--) {
		result = it->get().backward(m_handle, m_cublasHandle, result, target, d_onevec, fwdOutput[i]);
	}

	CheckError(cudaFree(result), __FILE__, __LINE__);
	CheckError(cudaFree(d_loss_data), __FILE__, __LINE__);
}

void Network::validateBatch()
{
	throw std::logic_error("The method or operation is not implemented.");
}
