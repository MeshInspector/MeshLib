#pragma once

#pragma warning(push)
#pragma warning(disable:4275) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'fmt::v10::format_error'
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <spdlog/tweakme.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic pop
#endif
#pragma warning(pop)

#include "MRPch/MRBindingMacros.h"

namespace MR
{
[[deprecated( "Use fmt::runtime" )]]
MR_BIND_IGNORE inline auto runtimeFmt( std::string_view str )
{
    return fmt::runtime( str );
}
}
