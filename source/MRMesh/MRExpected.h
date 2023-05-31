#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{

/// return type for a void function that can produce an error string
using VoidOrErrStr = tl::expected<void, std::string>;

/// Common operation canceled line for all
inline std::string operationCanceledLine()
{
    return "Operation was canceled";
}

/// Returns tl::expected error with `operationCanceledLine()`
inline tl::unexpected<std::string> tlOperationCanceled()
{
    return tl::make_unexpected( operationCanceledLine() );
}

} //namespace MR
