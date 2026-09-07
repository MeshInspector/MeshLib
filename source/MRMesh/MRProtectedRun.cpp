#include "MRProtectedRun.h"
#include <MRPch/MRSpdlog.h>
#include <boost/exception/diagnostic_information.hpp>

#ifdef _WIN32
#include <excpt.h>
#endif

namespace MR
{

namespace
{

bool protectedRun_( const std::function<void ()>& task, std::string & s )
{
#ifndef NDEBUG
    (void)s;
    task();
#else
    try
    {
        task();
    }
    catch ( const std::bad_alloc& badAllocE )
    {
        s = "Not enough memory for the requested operation: ";
        s += badAllocE.what();
        return false;
    }
    catch ( ... )
    {
        s = boost::current_exception_diagnostic_information();
        return false;
    }
#endif
    return true;
}

bool protectedRunEx_( const std::function<void ()>& task, std::string & s )
{
#if defined _WIN32 && defined NDEBUG
    __try
    {
        return protectedRun_( task, s );
    }
    __except ( EXCEPTION_EXECUTE_HANDLER )
    {
        s = "Unknown exception occurred";
        return false;
    }
#else
    return protectedRun_( task, s );
#endif
}

} // anonymous namespace

Expected<void> protectedRun( const std::function<void ()>& task )
{
    std::string s;
    if ( protectedRunEx_( task, s ) )
        return {};
    return unexpected( std::move( s ) );
}

std::function<void ()> protectedFunc( std::function<void ()> unprotectedFunc )
{
    return [f = std::move( unprotectedFunc )]
    {
        auto maybeOk = protectedRun( f );
        if ( !maybeOk )
            spdlog::error( "protectedFunc error: {}", maybeOk.error() );
    };
}

} //namespace MR
