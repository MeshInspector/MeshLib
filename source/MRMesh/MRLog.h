#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRSpdlog.h"
#include <filesystem>

namespace MR
{

// Make default spd logger
class Logger
{
public:
    MRMESH_API static Logger& instance();
    
    // store this pointer if need to prolong logger life time (necessary to log something from destructors)
    MRMESH_API const std::shared_ptr<spdlog::logger>& getSpdLogger() const;

    // returns default logger pattern
    MRMESH_API std::string getDefaultPattern() const;

    // adds custom sink to logger
    MRMESH_API void addSink( const spdlog::sink_ptr& sink );
    MRMESH_API void removeSink( const spdlog::sink_ptr& sink );

    // return filename of first found file sink, if there is no one, returns {}
    MRMESH_API std::filesystem::path getLogFileName() const;
private:
    Logger();
    std::shared_ptr<spdlog::logger> logger_;
};

// Setups logger:
// 1) makes stdout sink
// 2) makes file sink (MRLog.txt)
// 3) redirect std streams to logger
// log level - trace
MRMESH_API void setupLoggerByDefault();

// Redirects stdcout stdcerr stdclog to default logger
// note: do not call this function directly if you use MR::setupLoggerByDefault()
MRMESH_API void redirectSTDStreamsToLogger();

}
