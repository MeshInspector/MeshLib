#pragma once

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#if __GNUC__ == 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#pragma warning(push)
#pragma warning(disable:4275)
#pragma warning(disable:4251)
#pragma warning(disable:4273)
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#ifdef _WIN32
#include <spdlog/sinks/msvc_sink.h>
#endif
#pragma warning(pop)

#if __GNUC__ == 13
#pragma GCC diagnostic pop
#endif

#if (defined(__APPLE__) && defined(__clang__)) || defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

namespace MR
{
#if FMT_VERSION < 80000
inline const std::string& runtimeFmt( const std::string& str )
{
    return str;
}
#else
inline auto runtimeFmt( const std::string& str )
{
    return fmt::runtime( str );
}
#endif
}
