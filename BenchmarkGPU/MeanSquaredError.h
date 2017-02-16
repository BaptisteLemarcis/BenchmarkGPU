#pragma once
#include "Criterion.h"

class MeanSquaredError : public Criterion
{
public:
	float evaluate(float*, float* , int, int);
	MeanSquaredError();
	~MeanSquaredError();
};

