#pragma once

#include <boost/signals2/signal.hpp>

namespace MR
{

/// This class wraps boost::signals2::signal adding copy constructor and copy assignment operator,
/// which do nothing, but allow putting this as a member in copyables classes
template<typename T>
struct Signal : boost::signals2::signal<T>
{
    Signal() noexcept = default;
    Signal( const Signal& ) noexcept {}
    Signal( Signal&& ) noexcept = default;
    Signal& operator =( const Signal& ) noexcept { return *this; }
    Signal& operator =( Signal&& ) noexcept = default;
};

} //namespace MR
