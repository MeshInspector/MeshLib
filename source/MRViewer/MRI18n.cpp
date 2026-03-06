#include "MRI18n.h"
#ifndef MRVIEWER_NO_LOCALE
#include "MRLocale.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR::Locale
{

std::string translate( const char* msg, int domainId )
{
    return boost::locale::translate( msg ).str( get(), domainId );
}

std::string translate( const char* context, const char* msg, int domainId )
{
    return boost::locale::translate( context, msg ).str( get(), domainId );
}

std::string translate( const char* single, const char* plural, Int64 n, int domainId )
{
    return boost::locale::translate( single, plural, n ).str( get(), domainId );
}

std::string translate( const char* context, const char* single, const char* plural, Int64 n, int domainId )
{
    return boost::locale::translate( context, single, plural, n ).str( get(), domainId );
}

} // namespace MR::Locale
#endif
