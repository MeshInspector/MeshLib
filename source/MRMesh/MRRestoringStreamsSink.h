#pragma once

#include "MRMeshFwd.h"

#pragma warning(push)
#pragma warning(disable:4275)
#pragma warning(disable:4251)
#pragma warning(disable:4273)
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#pragma warning(pop)

#include <mutex>
#include <streambuf>
#include <string>

namespace MR
{

// A custom streambuf that outputs things directly to the default `spdlog` logger.
class LoggingStreambuf : public std::streambuf
{
public:
    MRMESH_API explicit LoggingStreambuf( spdlog::level::level_enum level );

protected:
    MRMESH_API std::streamsize xsputn( const char_type* s, std::streamsize count ) override;
    MRMESH_API int_type overflow( int_type ch = traits_type::eof() ) override;

private:
    // The log level for this stream.
    spdlog::level::level_enum level_;
    std::mutex bufMutex_;
    std::string buf_;
};

// This class is provided as spdlog::base_sink, spdlog::logger to manage its lifetime 
// and restore std streams once logger is destructed
class RestoringStreamsSink : public spdlog::sinks::base_sink<spdlog::details::null_mutex>
{
public:
    MRMESH_API RestoringStreamsSink();
    MRMESH_API ~RestoringStreamsSink();

private:
    virtual void sink_it_( const spdlog::details::log_msg& ) override { }
    virtual void flush_() override { }

    std::streambuf* coutBuf_;
    std::streambuf* cerrBuf_;
    std::streambuf* clogBuf_;

    LoggingStreambuf spdCoutBuf_;
    LoggingStreambuf spdCerrBuf_;
    LoggingStreambuf spdClogBuf_;
};

} //namespace MR
