#pragma once

#include "MRMacros.h"

#define MR_ON_INIT \
static auto MR_CONCAT( __mrOnInit_, __LINE__ ) = ::MR::detail::FuncCallHelper{} ->* [] () -> void

namespace MR::detail
{

class FuncCallHelper
{
public:
    template <typename Func>
    auto operator ->*( Func&& func )
    {
        func();
        return *this;
    }
};

} // namespace MR::detail
