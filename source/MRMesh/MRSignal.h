#pragma once

#include "MRMeshFwd.h"

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
template<typename Signature,
         typename Combiner = boost::signals2::optional_last_value<typename boost::function_traits<Signature>::result_type>>
struct Signal : boost::signals2::signal<Signature, Combiner>
{
    using Parent = boost::signals2::signal<Signature, Combiner>;

    Signal() noexcept = default;
    Signal( const Signal& ) noexcept : Parent() {}
    Signal( Signal&& ) noexcept = default;
    Signal& operator =( const Signal& ) noexcept { return *this; }
    Signal& operator =( Signal&& ) noexcept = default;
    Signal& operator =( Parent&& b ) noexcept { *static_cast<Parent*>(this) = std::move( b ); return *this; }

    /// implementation of connect method in MRMesh shared library allows the caller shared library to be safely unloaded
    MRMESH_API boost::signals2::connection connect( const boost::function<Signature> & slot, boost::signals2::connect_position position = boost::signals2::at_back );
};

} //namespace MR
