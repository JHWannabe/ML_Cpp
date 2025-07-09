#include "logger.h"

// Constructor: Initializes logger with default log level
Logger::Logger()
{
    this->logLevel = LOG_LEVEL_ERROR;
    this->logPath = MakeLogFolder();
}

// Constructor: Initializes logger with specified log level
Logger::Logger(int level)
{
    this->logLevel = level;
    this->logPath = MakeLogFolder();
}

// Creates log folder if it does not exist and returns the log path
std::string Logger::MakeLogFolder()
{
    WCHAR executePath[MAX_PATH];
    GetModuleFileName(NULL, executePath, MAX_PATH);
    std::wstring tempW(executePath);
    std::string temp(tempW.begin(), tempW.end());
    std::string runtimePath = temp.substr(0, temp.rfind("\\") + 1);
    logPath = runtimePath + "Log";

    char logFileFullPath[256];
    strcpy_s(logFileFullPath, logPath.c_str());
    int nResult = _access(logFileFullPath, 0);

    if (nResult != 0)
    {
        std::wstring widestr = std::wstring(logPath.begin(), logPath.end());
        const wchar_t* widecstr = widestr.c_str();

        if (_wmkdir(widecstr) == 0)
            return logPath;
    }
    else
        return logPath;
}

// Returns current timestamp as a string
std::string Logger::getTimestamp()
{
    std::string timeStamp;

    time_t currentSec;
    struct tm t;
    currentSec = time(NULL);
    localtime_s(&t, &currentSec);

    std::ostringstream oss;
    oss.clear();
    oss << " " << std::setfill('0') << std::setw(2) << t.tm_year + 1900 << " " << t.tm_mon + 1 << " " << t.tm_mday;
    oss << " " << std::setfill('0') << std::setw(2) << t.tm_hour;
    oss << ":" << std::setfill('0') << std::setw(2) << t.tm_min;
    oss << ":" << std::setfill('0') << std::setw(2) << t.tm_sec << '\0';
    timeStamp = timeStamp + oss.str();
    return timeStamp;
}

// Writes a log message to file and optionally to console
void Logger::WriteLog(const char* funcName, int line, int lv, const char* str, ...)
{
    time_t currentSec;
    struct tm t;
    currentSec = time(NULL);
    localtime_s(&t, &currentSec);

    char logFileName[2048];
    char ch[2048];
    strcpy_s(ch, logPath.c_str());
    sprintf_s(logFileName, sizeof(logFileName), "%s\\%04d_%02d_%02d.log", ch, t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);

    FILE* fp = NULL;
    if (0 != fopen_s(&fp, logFileName, "a"))
    {
        puts("fail to open file pointer");
        return;
    }

    char level[10];
    // Set log level string
    switch (lv)
    {
        case(LOG_LEVEL_OFF): strcpy_s(level, "[TEST]"); break;
        case(LOG_LEVEL_FATAL): strcpy_s(level, "[FATAL]"); break;
        case(LOG_LEVEL_ERROR): strcpy_s(level, "[ERROR]"); break;
        case(LOG_LEVEL_WARN): strcpy_s(level, "[WARN] "); break;
        case(LOG_LEVEL_INFO): strcpy_s(level, "[INFO] "); break;
        case(LOG_LEVEL_DEBUG): strcpy_s(level, "[DEBUG]"); break;
        case(LOG_LEVEL_TRACE): strcpy_s(level, "[TRACE]"); break;
    }

    // Format log message
    char message[2048];
    sprintf_s(message, sizeof(message), "%s %s [%s:%d] - %s\n", level, getTimestamp().c_str(), funcName, line, str);

    va_list args;

    // Write formatted message to log file
    va_start(args, str);
    vfprintf(fp, message, args);
    va_end(args);

    // Optionally print to console if log level is sufficient
    va_start(args, str);
    if (this->logLevel >= lv)
        vprintf(message, args);
    va_end(args);

    if (fp != NULL)
        fclose(fp);

    return;
}