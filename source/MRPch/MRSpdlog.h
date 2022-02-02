#pragma once

#ifdef __EMSCRIPTEN__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#pragma warning(push)
#pragma warning(disable:4275)
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

#ifdef __EMSCRIPTEN__
#pragma clang diagnostic pop
#endif