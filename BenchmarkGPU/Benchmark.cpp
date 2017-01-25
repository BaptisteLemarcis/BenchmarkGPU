#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cudnn.h>

#include "Trainer.h"
#include "IDXFile.h"

void main() {
	Trainer t;
	//t.forwardTraining(0, nullptr, nullptr, nullptr,nullptr);
	std::cout << "Generating magic number" << std::endl;
	int number = IDXFile::magicNumberGenerator(FLOAT, 3);
	std::cout << std::hex << number << std::endl;
}