#include "Criterion.h"

std::string Criterion::getName()
{
	return m_name;
}

Criterion::Criterion(std::string name)
{
	m_name = name;
}

Criterion::~Criterion()
{
}
