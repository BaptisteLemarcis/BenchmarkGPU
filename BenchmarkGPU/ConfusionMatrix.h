#pragma once
#include <vector>

class ConfusionMatrix
{
public:
	ConfusionMatrix();
	~ConfusionMatrix();
	void evaluate(float*, float*, int, int);
private:
	std::vector<int*> m_matrix;
};

