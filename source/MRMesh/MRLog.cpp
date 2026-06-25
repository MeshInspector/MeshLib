#include "MRLog.h"

#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWinapi.h"

namespace MR
{

Logger& Logger::instance()
{
    static Logger theLogger;
    return theLogger;
}

const std::shared_ptr<spdlog::logger>& Logger::getSpdLogger() const
{
    return logger_;
}

std::string Logger::getDefaultPattern() const
{
    return "[%d/%m/%C %H:%M:%S.%e] [%^%l%$] %v";
}

void Logger::addSink( const spdlog::sink_ptr& sink )
{
    logger_->sinks().push_back( sink );
}

void Logger::removeSink( const spdlog::sink_ptr& sink )
{
    auto& sinks = logger_->sinks();
    sinks.erase( std::remove( sinks.begin(), sinks.end(), sink ) );
}

std::filesystem::path Logger::getLogFileName() const
{
    if ( !logger_ )
        return {};

    for ( const auto& sink : logger_->sinks() )
    {
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::rotating_file_sink_mt>( sink ) )
            return r->filename();
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::rotating_file_sink_st>( sink ) )
            return r->filename();
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::daily_file_sink_mt>( sink ) )
            return r->filename();
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::daily_file_sink_st>( sink ) )
            return r->filename();
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::basic_file_sink_mt>( sink ) )
            return r->filename();
        if ( auto r = std::dynamic_pointer_cast<spdlog::sinks::basic_file_sink_st>( sink ) )
            return r->filename();
    }

    return {};
}

Logger::Logger()
{
    logger_ = spdlog::get( "MainLogger" );
    if ( logger_ )
        return;

    logger_ = std::make_shared<spdlog::logger>( spdlog::logger( "MainLogger" ) );
    spdlog::register_logger( logger_ );

    spdlog::set_default_logger( logger_ );
#ifdef _WIN32
    SetConsoleOutputCP( CP_UTF8 );
#endif
}

}
