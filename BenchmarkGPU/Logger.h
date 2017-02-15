#pragma once

#include <fstream>
#include <iostream>
#include <vector>

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
	void flush();
	/*friend Logger<T>& operator<<(Logger &os, const T &t);
	friend Logger& operator<<(Logger& os, std::ostream&(*f)(std::ostream&));*/
private:
	std::string m_logPath;
	std::vector<std::string> lines;
	std::ofstream m_logFile;
	static Logger *m_instance;
	Logger();
};
