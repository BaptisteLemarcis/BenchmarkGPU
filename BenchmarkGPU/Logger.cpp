#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "Logger.h"

Logger *Logger::m_instance;

Logger::Logger(){}

Logger* Logger::instance()
{
	if (!m_instance) {
		m_instance = new Logger;
	}
	return m_instance;
}

Logger::~Logger(){
	m_logFile.close();
}

void Logger::setFile(std::string logPath)
{
	m_logPath = logPath;
	m_logFile.open(logPath, std::ios::out);
}

void Logger::writeLine(std::string line)
{
	m_logFile.precision(3);
	m_logFile.setf(std::ios::fixed, std::ios::floatfield);
	m_logFile << line << std::endl;
	m_logFile.flush();
}
