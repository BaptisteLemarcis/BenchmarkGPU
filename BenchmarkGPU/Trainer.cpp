#include <cudnn.h>
#include "Trainer.h"
#include "GenericFunctions.h"

Trainer::Trainer() : Trainer(0, 128) {}

Trainer::Trainer(int GPUID, int batchSize) {
	m_gpuid = GPUID;
	m_batchSize = batchSize;

	std::cout << "Setting CudaDevice to " << m_gpuid << std::endl;
	CheckCudaError(cudaSetDevice(GPUID));
	std::cout << "CudaDevice set." << std::endl;

	size_t version = cudnnGetVersion();
	std::cout << "Running on CudNN version : " << version << std::endl;

	CheckCudNNError(cudnnCreate(&m_handle));
}

Trainer::~Trainer() {
	CheckCudNNError(cudnnDestroy(m_handle));
}

void Trainer::doStuff(int zero, int one)
{

}
