#include "MRWebLogger.h"

namespace MR
{

WebLogger& WebLogger::instance()
{
    static WebLogger instance;
    return instance;
}

void WebLogger::log( const std::string& message, spdlog::level::level_enum level )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    if ( !enabled_ || level < logLevel_ )
        return;

    buffer_.push_back(
        { .timestamp = std::chrono::system_clock::now(),
          .logLevel = level,
          .message = message } );
}

void WebLogger::setEnabled( bool enabled )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    enabled_ = enabled;

    if ( !enabled_ )
    {
        buffer_.clear();
    }
}

void WebLogger::setLogLevel( spdlog::level::level_enum lvl )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    logLevel_ = lvl;

    buffer_.erase(
        std::remove_if( buffer_.begin(), buffer_.end(), [lvl]( const LogEntry& e ) { return e.logLevel < lvl; } ),
        buffer_.end() );
}

std::vector<WebLogger::LogEntry> WebLogger::extractBatch( int maxBatch )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    if ( !enabled_ || logLevel_ == spdlog::level::level_enum::off )
        return {};
    if ( buffer_.size() < maxBatch )
        return {};

    std::vector<LogEntry> batch;
    batch.swap( buffer_ );
    return batch;
}

} // namespace MR
