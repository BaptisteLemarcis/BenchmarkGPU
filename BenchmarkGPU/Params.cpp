#include <map>
#include <stdexcept>
#include <sstream>

#include "Logger.h"
#include "Params.h"

Params Params::load(int argc, char ** argv)
{
	if (argc != 11) {
		printf("Usage:\n");
		printf("./BenchmarkGPU <nbLayers> <hiddenSize> <batchSize> <epoch> <inputSize> <nbData> <seqLength> <dropout> <learningRate> <outputDim>\n");
		throw std::invalid_argument("Invalid number of argument");
	}

	Params p;
	p.m_p["nbLayers"] = atof(argv[1]);
	p.m_p["hiddenSize"] = atof(argv[2]);
	p.m_p["batchSize"] = atof(argv[3]);
	p.m_p["epoch"] = atof(argv[4]);
	p.m_p["inputSize"] = atof(argv[5]);
	p.m_p["nbData"] = atof(argv[6]);
	p.m_p["seqLength"] = atof(argv[7]);
	p.m_p["dropout"] = atof(argv[8]);
	p.m_p["learningRate"] = atof(argv[9]);
	p.m_p["outputDim"] = atof(argv[10]);
	return p;
}

Params Params::load(std::string file)
{
	Params p;
	return p;
}

float & Params::operator[](std::string idx)
{
	return m_p[idx];
}

void Params::writeParamsToFile() {
	std::stringstream toWrite(std::stringstream::in | std::stringstream::out);
	toWrite << "==========================================================" << std::endl;
	toWrite << "Learning Rate\t" << m_p["learningRate"] << std::endl;
	toWrite << "Hidden Size\t\t" << (int)m_p["hiddenSize"] << std::endl;
	toWrite << "Input Size\t\t" << (int)m_p["inputSize"] << std::endl;
	toWrite << "Seq Length\t\t" << (int)m_p["seqLength"] << std::endl;
	toWrite << "Dropout\t\t\t" << m_p["dropout"] << std::endl;
	toWrite << "Epoch\t\t\t" << (int)m_p["epoch"] << std::endl;
	toWrite << "Batch Size\t\t" << (int)m_p["batchSize"] << std::endl;
	toWrite << "==========================================================";
	Logger::instance()->writeLine(toWrite.str());
}

