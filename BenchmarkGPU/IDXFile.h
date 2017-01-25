#pragma once

#define UBYTE 0x08
#define BYTE 0x09
#define SHORT 0x0B
#define INT 0x0C
#define FLOAT 0x0D
#define DOUBLE 0x0E

typedef unsigned char byte;

class IDXFile {
public:
	/**
	*	\brief Get the MagicNumber of this IDXFile
	*	\param magicNumber - The magic number, \see MagicNumberGenerator to generate it.
	*	\param numDim - Number of dimensions
	*	\param dimSize - Size of each dimension
	*	\param data - Data to store
	*/
	IDXFile(int, int, int[], float[]);

	~IDXFile();

	/**
	*	\brief Get the MagicNumber of this IDXFile
	*	\return - The magic number
	*/
	int getMagicNumber();

	/**
	*	\brief Get the size of the desired dimension
	*	\param x - The desired dimension
	*	\return - The size of the dimension x
	*/
	int getDimensionSize(int);

	/**
	*	\brief Get the number of dimension of this IDXFile
	*	\return - The dimension number's
	*/
	int getNumberOfDimension();

	/**
	*	\brief Get all data of the desired dimension
	*	\param x - The desired dimension
	*	\return - The data
	*/
	float* getDimension(int);

	/**
	*	\brief Get all dimensions data's of this IDXFile
	*	\return - The data
	*/
	float* getAllDimension();

	/**
	*	\brief Write the current file in the IDX Format
	*	\param path - Path where to write the IDXFile
	*/
	void writeFile(std::string);

	/**
	*	\brief Append data to a specific dimension
	*	\param data - Data to append
	*	\param x - Dimension in witch the data need to be append
	*/
	void appendDataToDimension(float*, int);

	/**
	*	\brief Generate the magic Number for IDXFileFormat
	*	\param dataType - Data type, use defined type from IDXFile.h
	*	\param numDim - Number of data dimension's
	*	\return - The magic number
	*/
	static int magicNumberGenerator(byte, int);

	/**
	*	\brief Read a file in the IDX Format
	*	\param path - Path of the IDXFile to read
	*	\return - The IDXFile object
	*/
	static IDXFile readFile(std::string);
private:
	int m_magicNumber;
	int* m_dimensionSize;
	int m_nbDimension;
	float* m_data;
};