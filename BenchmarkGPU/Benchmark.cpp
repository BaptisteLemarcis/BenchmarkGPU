#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <iostream>
#include <cudnn.h>
#include <sstream>
#include <tuple>

#include "Logger.h"
#include "Params.h"
#include "Network.h"
#include "GenericFunctions.h"
#include "LSTMLayer.h"
#include "FullyConnectedLayer.h"
#include "SoftmaxLayer.h"

int main(int argc, char** argv) {
	
	Logger::instance()->setFile("BenchmarkGPU.log");

	Params p = Params::load(argc, argv);
	p.writeParamsToFile();
	auto data = generateData(p["nbData"] * p["inputSize"]);

	Network* n = new Network(p["batchSize"], p["learningRate"], p["inputSize"], p["outputDim"], p["seqLength"]);

	LSTMLayer lstm(n->getHandle(), p["inputSize"], p["hiddenSize"], p["nbLayers"], p["batchSize"], p["seqLength"], p["dropout"]);
	FullyConnectedLayer fc(p["hiddenSize"], p["outputDim"], p["batchSize"]);
	SoftmaxLayer sm(p["outputDim"], p["batchSize"]);

	n->addLayer(lstm);
	n->addLayer(fc);
	n->addLayer(sm);

	n->train(std::get<0>(data), std::get<1>(data), p["epoch"], p["nbData"]);

	delete n;

	std::cout << "Press a key to continue..." << std::endl;
	std::cin.ignore();

	return EXIT_SUCCESS;
}