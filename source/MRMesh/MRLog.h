#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"

#include <filesystem>

namespace spdlog
{
class logger;
namespace sinks { class sink; }
using sink_ptr = std::shared_ptr<sinks::sink>;
}

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// Make default spd logger
class MR_BIND_IGNORE Logger
{
public:
    MRMESH_API static Logger& instance();

    /// simple methods to log messages
    /// for more optimal methods use getSpdLogger()
    MRMESH_API static void trace( std::string_view msg );
    MRMESH_API static void debug( std::string_view msg );
    MRMESH_API static void info( std::string_view msg );
    MRMESH_API static void warn( std::string_view msg );
    MRMESH_API static void error( std::string_view msg );
    MRMESH_API static void critical( std::string_view msg );

    /// store this pointer if need to prolong logger life time (necessary to log something from destructors)
    MRMESH_API const std::shared_ptr<spdlog::logger>& getSpdLogger() const;

    /// returns default logger pattern
    MRMESH_API std::string getDefaultPattern() const;

    /// adds custom sink to logger
    MRMESH_API void addSink( const spdlog::sink_ptr& sink );
    MRMESH_API void removeSink( const spdlog::sink_ptr& sink );

    /// return filename of first found file sink, if there is no one, returns {}
    MRMESH_API std::filesystem::path getLogFileName() const;
private:
    Logger();
    std::shared_ptr<spdlog::logger> logger_;
};

/// \}

}
