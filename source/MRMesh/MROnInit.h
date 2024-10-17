#pragma once

#include "MRMacros.h"

#define MR_ON_INIT( ... ) \
namespace { auto MR_CONCAT( __mrOnInit_, __LINE__ ) = ::MR::detail::FuncCallHelper{} ->* [] () -> void __VA_ARGS__ ; }

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
