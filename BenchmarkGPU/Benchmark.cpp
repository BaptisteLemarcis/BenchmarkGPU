#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <iostream>
#include <cudnn.h>
#include <MagicNumber.h>
#include <Data.h>
#include <IDXFile.h>
#include <bitmap_image.h>
#include "Trainer.h"

void main(int argc, char** argv) {
	Trainer t;
	t.forwardTraining(0, nullptr, nullptr, nullptr, nullptr);
	/*IDXFile* file = IDXFile::readFile("C:\\Users\\baptiste\\Downloads\\train-labels-idx1-ubyte\\train-images.idx3-ubyte");
	std::cout << "x : " << file->getData().getDimensionSize(1) << std::endl;
	std::cout << "y : " << file->getData().getDimensionSize(2) << std::endl;
	bitmap_image image(file->getData().getDimensionSize(1), file->getData().getDimensionSize(2));


	DataType* d = file->getData().getData();

	delete file;*/
	std::cout << "Press a key to continue..." << std::endl;
	std::cin.ignore();
}