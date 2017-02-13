#include <tuple>
#include "Layer.h"

Layer::Layer(int inputDim, int outputDim, int batchSize): m_inputDim(inputDim), m_outputDim(outputDim), m_batchSize(batchSize)
{
}


Layer::~Layer()
{
}

int Layer::getOutputDim()
{
	return m_outputDim;
}
