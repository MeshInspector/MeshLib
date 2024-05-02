#pragma once

#if (defined(__APPLE__) && defined(__clang__))
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#pragma warning(push)
#pragma warning(disable: 4619) //#pragma warning: there is no warning number
#include <boost/signals2/signal.hpp>
#pragma warning(pop)

#if (defined(__APPLE__) && defined(__clang__))
#pragma clang diagnostic pop
#endif

namespace MR
{

/// This class wraps boost::signals2::signal adding copy constructor and copy assignment operator,
/// which do nothing, but allow putting this as a member in copyable classes
template<typename T>
struct Signal : boost::signals2::signal<T>
{
    Signal() noexcept = default;
    Signal( const Signal& ) noexcept : boost::signals2::signal<T>() {}
    Signal( Signal&& ) noexcept = default;
    Signal& operator =( const Signal& ) noexcept { return *this; }
    Signal& operator =( Signal&& ) noexcept = default;
    Signal& operator =( boost::signals2::signal<T>&& b ) noexcept { *static_cast<boost::signals2::signal<T>*>(this) = std::move( b ); return *this; }
};

} //namespace MR
