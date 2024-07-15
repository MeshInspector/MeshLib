#pragma once

#include "MRPch/MRTBB.h"

namespace MR
{

namespace Parallel
{

struct CallSimply
{
    auto operator() ( auto && f, auto id ) const { return f( id ); }
    auto operator() ( auto && f, auto id, auto && subrange ) const { return f( id, subrange ); }
};

struct CallSimplyMaker
{
    auto operator() () const { return CallSimply{}; }
};

template<typename T>
struct CallWithTLS
{
    T & tls;
    auto operator() ( auto && f, auto id ) const { return f( id, tls ); }
    auto operator() ( auto && f, auto id, auto && subrange ) const { return f( id, subrange, tls ); }
};

template<typename L>
struct CallWithTLSMaker
{
    tbb::enumerable_thread_specific<L> & e;
    auto operator() () const { return CallWithTLS<L>{ e.local() }; }
};

} //namespace Parallel

} //namespace MR
