#pragma once

#pragma warning(push)
#pragma warning(disable:4275) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'fmt::v10::format_error'
#include <spdlog/tweakme.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#pragma warning(pop)

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
