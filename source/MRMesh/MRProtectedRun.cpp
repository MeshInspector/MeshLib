#include "MRProtectedRun.h"
#include <boost/exception/diagnostic_information.hpp>

namespace MR
{

namespace
{

bool protectedRun_( const std::function<void ()>& task, std::string & s )
{
#ifndef NDEBUG
    task();
#else
    try
    {
        task();
    }
    catch ( const std::bad_alloc& badAllocE )
    {
        s = std::string( badAllocE.what() );
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
#ifndef _WIN32
    return protectedRun( task, s );
#else
#ifndef NDEBUG
    task();
    (void)s;
    return true;
#else
    __try
    {
        return protectedRun_( task, s );
    }
    __except ( EXCEPTION_EXECUTE_HANDLER )
    {
        s = "Unknown exception occurred";
        return false;
    }
#endif
#endif
}

} // anonymous namespace

Expected<void> protectedRun( const std::function<void ()>& task )
{
    std::string s;
    if ( protectedRun_( task, s ) )
        return {};
    return unexpected( std::move( s ) );
}

Expected<void> protectedRunEx( const std::function<void ()>& task )
{
    std::string s;
    if ( protectedRunEx_( task, s ) )
        return {};
    return unexpected( std::move( s ) );
}

} //namespace MR
