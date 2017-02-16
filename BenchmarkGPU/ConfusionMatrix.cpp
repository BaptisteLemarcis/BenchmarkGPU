#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <string.h>
#include <sstream>

#include "ConfusionMatrix.h"
#include "GPUKernel.cuh"
#include "Logger.h"


ConfusionMatrix::ConfusionMatrix()
{
}


ConfusionMatrix::~ConfusionMatrix()
{
}

float ConfusionMatrix::evaluate(float* d_data, float* d_target, int size, int nbClass)
{
	/*float* d_matrix;
	int fullSize = nbClass * size;
	cudaMalloc((void**)&d_matrix, nbClass * sizeof(float));
	FillVec(nbClass, d_matrix, 0.f);
	computeConfusionMatrix(d_data, d_target, d_matrix, size, nbClass);
	float* matrix = new float[nbClass];
	cudaMemcpy(matrix, d_matrix, nbClass * sizeof(float), cudaMemcpyDeviceToHost);*/
	std::stringstream ss("");
	float* results = new float[size * nbClass];
	float* targets = new float[size * nbClass];
	cudaMemcpy(results, d_data, size * nbClass * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(targets, d_target, size * nbClass * sizeof(float), cudaMemcpyDeviceToHost);
	
	int numCorrect = 0;
	int numWrong = 0;

	for (auto i = 0; i < size; i++) {
		int classDetected = 0;
		int classExpected = 0;
		//ss << "i (" << i << ") : ";
		for (auto j = 1; j < nbClass; j++) {
			//float realOutput = results[i*nbClass + j] < 0.5 ? 0.f : 1.f;
			//ss << results[i*nbClass + j] << "\t";
			if (results[i*nbClass + j] > results[i*nbClass + classDetected]) {
				classDetected = j;
			}
		}
		for (auto j = 1; j < nbClass; j++) {
			//float realOutput = results[i*nbClass + j] < 0.5 ? 0.f : 1.f;
			if (targets[i*nbClass + j] > targets[i*nbClass + classExpected]) {
				classExpected = j;
			}
		}
		//ss << " // " << classDetected << " ---- " << classExpected << std::endl;
		if (targets[i*nbClass + classDetected] == 1.f) {
			++numCorrect;
		}
		else {
			++numWrong;
		}
	}
	float acc = (numCorrect * 1.0) / (numCorrect + numWrong);
	return acc;
	//ss << "Accuracy : " << acc << std::endl;
	/*
	ss << "\t";
	for (auto i = 0; i < nbClass; i++)
		ss << i << "\t";
	ss << "< Class" << std::endl;
	for (auto i = 0; i < nbClass; i++) {
		ss << i << "\t";
		for (auto j = 0; j < nbClass; j++) {
			ss << matrix[i*nbClass + j] << "\t";
		}
		ss << std::endl;
	}*/
	//Logger::instance()->writeLine(ss.str());
	//Logger::instance()->flush();
}
