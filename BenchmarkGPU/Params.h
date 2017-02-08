#pragma once
#include <map>
class Params
{
public:
	~Params() {}
	void writeParamsToFile();
	static Params load(int, char**);
	static Params load(std::string);

	float& operator[](std::string idx);
private:
	Params() {};
	std::map<std::string, float> m_p;
};

