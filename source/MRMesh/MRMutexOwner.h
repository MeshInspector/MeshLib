#pragma once

#include "MRPch/MRBindingMacros.h"
#include <mutex>

namespace MR
{

/// This class exists to provide copy and move constructors and assignment operations for std::mutex
/// which actually does nothing
struct MutexOwner
{
    MutexOwner() noexcept = default;
    MutexOwner( const MutexOwner& ) noexcept {}
    MutexOwner( MutexOwner&& ) noexcept {}
    MutexOwner& operator =( const MutexOwner& ) noexcept { return *this; }
    MutexOwner& operator =( MutexOwner&& ) noexcept { return *this; }

    MR_BIND_IGNORE std::mutex m;
};

} //namespace MR
