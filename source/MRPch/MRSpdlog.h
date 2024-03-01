#pragma once

// we need to make default visibility of sinks for dynamic_cast
// to be able to find objects from other shared libraries (particularity on Apple)
#ifndef _WIN32
namespace spdlog
{

namespace details
{

struct __attribute__((visibility("default"))) console_mutex;

struct __attribute__((visibility("default"))) console_nullmutex;

} // namespace details

namespace sinks
{

template<typename Mutex>
class __attribute__((visibility("default"))) rotating_file_sink;

template<typename Mutex, typename FileNameCalc>
class __attribute__((visibility("default"))) daily_file_sink;

template<typename Mutex>
class __attribute__((visibility("default"))) basic_file_sink;

template<typename ConsoleMutex>
class __attribute__((visibility("default"))) ansicolor_stdout_sink;

template<typename ConsoleMutex>
class __attribute__((visibility("default"))) ansicolor_stderr_sink;

} //namespace sinks

} //namespace spdlog
#endif

#include "MRFmt.h"

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
#pragma warning(disable:4275) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'fmt::v10::format_error'
#pragma warning(disable:4251)
#pragma warning(disable:4273)
#include <spdlog/spdlog.h>
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
