#include <cuda.h>

#include "Criterion.h"
#include "MeanSquaredError.h"
#include "GenericFunctions.h"

float MeanSquaredError::evaluate(float* d_y, float* d_t, int dataLenght, int nbClass)
{

	float* targets = new float[dataLenght*nbClass];
	float* output = new float[dataLenght*nbClass];
	float error = 0.f;
	CheckError(cudaMemcpy(targets, d_t, dataLenght*nbClass * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	CheckError(cudaMemcpy(output, d_y, dataLenght*nbClass * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	for (int i = 0; i < dataLenght; i++) {
		for (int j = 0; j < nbClass; j++) {
			//float realOutput = output[i*getOutputDim() + j] < 0.5 ? 0.f : 1.f;
			float z = targets[i*nbClass + j] - output[i*nbClass + j];
			error += z*z;
		}
	}
	
	return error /= dataLenght;
}

MeanSquaredError::MeanSquaredError() : Criterion("MSE")
{
}


MeanSquaredError::~MeanSquaredError()
{
}
