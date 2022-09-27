#include "MRLog.h"
#include "MRRestoringStreamsSink.h"
#include "MRSystem.h"
#include "MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include <boost/stacktrace.hpp>
#include <csignal>

#ifdef __MINGW32__
#include <windows.h>
#endif

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

namespace
{
void tryClearDirectory( const std::filesystem::path& dir )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( dir, ec ) )
        return;

    std::filesystem::remove_all( dir, ec );
}

void crashSignalHandler( int signal )
{
    spdlog::critical( "Crash signal: {}", signal );
    auto stacktrace = boost::stacktrace::stacktrace();
    for ( const auto& frame : stacktrace )
        spdlog::critical( "{} {} {}", frame.name(), frame.source_file(), frame.source_line() );
    std::exit( signal );
}

}

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

void setupLoggerByDefault()
{
    printStacktraceOnCrash();
    redirectSTDStreamsToLogger();
    // write log to console
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level( spdlog::level::trace );
    console_sink->set_pattern( Logger::instance().getDefaultPattern() );
    Logger::instance().addSink( console_sink );

    // write log to file
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t( now );
    auto fileName = GetTempDirectory();
    fileName /= "Logs";
    tryClearDirectory( fileName );

    fileName /= fmt::format( "MRLog_{:%Y-%m-%d_%H-%M-%S}_{}.txt", fmt::localtime( t ),
                std::chrono::milliseconds( now.time_since_epoch().count() ).count() % 1000 );

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>( utf8string( fileName ), 1024 * 1024 * 5, 1, true );
    file_sink->set_level( spdlog::level::trace );
    file_sink->set_pattern( Logger::instance().getDefaultPattern() );
    Logger::instance().addSink( file_sink );

#ifdef _WIN32
    auto msvc_sink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
    msvc_sink->set_level( spdlog::level::trace );
    msvc_sink->set_pattern( Logger::instance().getDefaultPattern() );
    Logger::instance().addSink( msvc_sink );
#endif

    auto logger = Logger::instance().getSpdLogger();

    logger->set_level( spdlog::level::trace );

    // update file on each msg
    logger->flush_on( spdlog::level::trace );

    spdlog::info( "MR Version info: {}", GetMRVersionString() );
}

void redirectSTDStreamsToLogger()
{
    auto restoringSink = std::make_shared<RestoringStreamsSink>();
    Logger::instance().addSink( restoringSink );
}

void printStacktraceOnCrash()
{
    std::signal( SIGTERM, crashSignalHandler );
    std::signal( SIGSEGV, crashSignalHandler );
    std::signal( SIGINT, crashSignalHandler );
    std::signal( SIGILL, crashSignalHandler );
    std::signal( SIGABRT, crashSignalHandler );
    std::signal( SIGFPE, crashSignalHandler );
}

}
