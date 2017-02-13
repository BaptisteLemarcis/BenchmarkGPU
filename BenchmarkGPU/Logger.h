#pragma once

#include <fstream>
#include <iostream>

enum LogLevel {
	INFO,
	DEBUG,
	ERROR
};

class Logger
{
public:
	static Logger* instance();
	~Logger();

	void setFile(std::string);
	void writeLine(std::string);
	
	//friend Logger& operator<<(Logger &os, T &t const);
	//friend Logger& operator<<(Logger& os, std::ostream&(*f)(std::ostream&));
private:
	std::string m_logPath;
	std::ofstream m_logFile;
	static Logger *m_instance;
	Logger();
};
