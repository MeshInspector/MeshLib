#pragma once

#pragma warning(push)
#pragma warning(disable:4275) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'fmt::v10::format_error'
#include <spdlog/tweakme.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
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

#ifndef MR_NO_GETTEXT_MACROS
#define f_tr( ... ) fmt::runtime( MR::Locale::translate( __VA_ARGS__ ) )
#endif // MR_NO_GETTEXT_MACROS
