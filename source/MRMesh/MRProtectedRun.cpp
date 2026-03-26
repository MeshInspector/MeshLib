#include "MRProtectedRun.h"
#include <boost/exception/diagnostic_information.hpp>

namespace MR
{

Expected<void> protectedRun( const std::function<void ()>& task )
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
        return unexpected( std::string( badAllocE.what() ) );
    }
    catch ( ... )
    {
        return unexpected( boost::current_exception_diagnostic_information() );
    }
#endif
    return {};
}

Expected<void> protectedRunEx1( const std::function<void ()>& task )
{
#ifndef _WIN32
    return protectedRun( task );
#else
#ifndef NDEBUG
    task();
    return {};
#else
    __try
    {
        ///*return*/(void) protectedRun( task );
        task();
    }
    __except ( EXCEPTION_EXECUTE_HANDLER )
    {
        //return unexpected( std::string( "Unknown exception occurred" ) );
    }
#endif
#endif
    return {};
}

} //namespace MR
