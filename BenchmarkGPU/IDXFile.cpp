#include <stdexcept>
#include <iostream>
#include <fstream>
#include "NotImplemetendException.h"
#include "IDXFile.h"

IDXFile::IDXFile(int magicNumber, int numDim, int dimSize[], float data[])
{
	m_magicNumber = magicNumber;
	m_dimensionSize = new int[numDim];
	m_nbDimension = numDim;
	m_data = data;
}

IDXFile::~IDXFile()
{
	delete m_dimensionSize;
	delete m_data;
}

int IDXFile::getMagicNumber()
{
	return m_magicNumber;
}

int IDXFile::getDimensionSize(int x)
{
	if (x >= m_nbDimension || x < 0) {
		throw std::invalid_argument("Received a negative dimension or a dimension higher than the number of available dimension.");
	}
	return m_dimensionSize[x];
}

int IDXFile::getNumberOfDimension()
{
	return m_nbDimension;
}

float * IDXFile::getDimension(int x)
{
	if (x >= m_nbDimension || x < 0) {
		throw std::invalid_argument("Received a negative dimension or a dimension higher than the number of available dimension.");
	}
	return nullptr;
}

float * IDXFile::getAllDimension()
{
	return m_data;
}

void IDXFile::appendDataToDimension(float *, int)
{
}

int IDXFile::magicNumberGenerator(byte dataType, int numDim)
{
	int magicNumber = 0;
	magicNumber <<= 8;
	magicNumber |= 0x00;
	magicNumber <<= 8;
	magicNumber |= 0x00;
	if (dataType != UBYTE && dataType != BYTE && dataType != SHORT && dataType != INT && dataType != FLOAT && dataType != DOUBLE)
		throw std::invalid_argument("Received a non supported data type.");
	magicNumber <<= 8;
	magicNumber |= dataType;
	magicNumber <<= 8;
	magicNumber |= numDim;

	return magicNumber;
}

IDXFile IDXFile::readFile(std::string path)
{
	std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		try {
			int size;
			char* buffer;

			// Read the magic number
			size = sizeof(int);
			buffer = new char[size];
			file.seekg(0, std::ios::beg);
			file.read(buffer, size);

			// Extracting the number of dimension

			file.close();
		}
		catch (std::exception& e) {
			file.close();
			std::cerr << e.what() << std::endl;
		}
	}
	throw new std::string("Error : File not opened (" + path + ")\n");
}

void IDXFile::writeFile(std::string path)
{
	//throw new NotImplementedException();
}
