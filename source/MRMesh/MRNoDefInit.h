#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// this class is similar to T, but does not make default initialization of the fields for best performance
template <typename T>
struct NoDefInit : T
{
    constexpr NoDefInit() noexcept : T( noInit ) {}
    using T::operator=;
};

} // namespace MR
