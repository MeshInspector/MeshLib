#include "MRI18n.h"
#ifndef MRVIEWER_NO_LOCALE
#include "MRLocale.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR::Locale
{

std::string translate( const char* msg )
{
    return boost::locale::translate( msg ).str( get() );
}

std::string translate( const char* context, const char* msg )
{
    return boost::locale::translate( context, msg ).str( get() );
}

std::string translate( const char* single, const char* plural, long long n )
{
    return boost::locale::translate( single, plural, n ).str( get() );
}

std::string translate( const char* context, const char* single, const char* plural, long long n )
{
    return boost::locale::translate( context, single, plural, n ).str( get() );
}

} // namespace MR::Locale
#endif
