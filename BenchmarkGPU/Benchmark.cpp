#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cudnn.h>
#include <MagicNumber.h>
#include <IDXFile.h>
#include "Trainer.h"

void main(int argc, char** argv) {
	//Trainer t;
	//t.forwardTraining(0, nullptr, nullptr, nullptr,nullptr);
	IDXFile* file = IDXFile::readFile("C:\\Users\\baptiste\\Downloads\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
	delete file;
	std::cout << "Press a key to continue..." << std::endl;
	std::cin.ignore();
}