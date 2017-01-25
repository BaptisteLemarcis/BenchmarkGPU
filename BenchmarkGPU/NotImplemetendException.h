#pragma once
#include <stdexcept>

class NotImplementedException : public std::logic_error
{
public:
	virtual char const * what();
};