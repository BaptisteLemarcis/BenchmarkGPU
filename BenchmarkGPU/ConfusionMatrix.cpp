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

void ConfusionMatrix::evaluate(float* d_data, float* d_target, int size, int nbClass)
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
	for (auto i = 0; i < size; i++) {
		ss << i << "\t";
		for (auto j = 0; j < nbClass; j++) {
			ss << results[i*nbClass + j] << "\t";
		}
		ss << "(";
		for (auto j = 0; j < nbClass; j++) {
			ss << targets[i*nbClass + j] << "\t";
		}
		ss << ")" << std::endl;
	}

	/*ss << "\t";
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
	Logger::instance()->writeLine(ss.str());
	Logger::instance()->flush();
}
