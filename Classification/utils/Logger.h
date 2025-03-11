#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <ctime>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstddef>
#include <Windows.h>
#include <iostream>
#if _WIN32
#include <io.h>
#else
#include<unistd.h>
#endif

#define LOG_LEVEL_OFF   0
#define LOG_LEVEL_FATAL 10
#define LOG_LEVEL_ERROR 20
#define LOG_LEVEL_WARN  30
#define LOG_LEVEL_INFO  40
#define LOG_LEVEL_DEBUG 50
#define LOG_LEVEL_TRACE 60
#define LOG_LEVEL_ALL   100

#define __LOG_EXTENSION__ "log"

#define fatal(str, ...) WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_FATAL, str, __VA_ARGS__)
#define error(str, ...) WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_ERROR, str, __VA_ARGS__)
#define warn(str, ...)  WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_WARN, str, __VA_ARGS__)
#define info(str, ...)  WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_INFO, str, __VA_ARGS__)
#define debug(str, ...) WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_DEBUG, str, __VA_ARGS__)
#define trace(str, ...) WriteLog(__FUNCTION__, __LINE__, LOG_LEVEL_TRACE, str, __VA_ARGS__)

class Logger
{
private:
    int logLevel;
    std::string getTimestamp();
    std::string logPath;
    

public:
    Logger();
    Logger(int level);
    void WriteLog(const char* funcName, int line, int level, const char* str, ...);
    std::string MakeLogFolder();
};
 
static Logger logger;

