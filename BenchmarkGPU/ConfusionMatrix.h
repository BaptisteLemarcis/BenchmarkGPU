#pragma once
#include <vector>

class ConfusionMatrix
{
public:
	ConfusionMatrix();
	~ConfusionMatrix();
	float evaluate(float*, float*, int, int);
private:
	std::vector<int*> m_matrix;
};

