#pragma once
#include <string>

class Criterion
{
public:
	/**
	*	\brief .
	*	\param y - Output of neural network
	*	\param t - Target for each output
	*	\param dataLenght - Number of data (batch size)
	*	\param nbClass - Number of class (Output dim)
	*/
	virtual float evaluate(float*, float*, int, int) = 0;

	std::string getName();

protected:
	Criterion(std::string);
	~Criterion();

private:
	std::string m_name;
};

