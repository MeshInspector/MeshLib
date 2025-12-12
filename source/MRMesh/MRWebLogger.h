#pragma once
#include "MRMeshFwd.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

// This class collects log entries for later sending to a web server
class WebLogger
{
public:
    struct LogEntry
    {
        std::chrono::system_clock::time_point timestamp;
        spdlog::level::level_enum logLevel;
        std::string message;
    };

    MRMESH_API static WebLogger& instance();

    MRMESH_API void setEnabled( bool enabled );
    MRMESH_API void setLogLevel( spdlog::level::level_enum lvl );

    // Extract a batch of log entries up to maxBatch size
    MRMESH_API std::vector<LogEntry> extractBatch( int maxBatch );

    // Collect a log entry
    void log( const std::string& message, spdlog::level::level_enum level );

private:
    bool enabled_ = true;
    spdlog::level::level_enum logLevel_ = spdlog::level::level_enum::info;

    std::mutex mutex_;
    std::vector<LogEntry> buffer_;
};

} // namespace MR 
