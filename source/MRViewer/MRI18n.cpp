#include "MRI18n.h"
#ifndef MRVIEWER_NO_LOCALE
#include "MRLocale.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

namespace MR::Locale
{

std::string translate( const char* msg, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( msg );
    return boost::locale::translate( msg ).str( get(), domain.id );
}

std::string translate( const char* context, const char* msg, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( context, msg );
    return boost::locale::translate( context, msg ).str( get(), domain.id );
}

std::string translate( const char* single, const char* plural, Int64 n, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( single, plural, n );
    return boost::locale::translate( single, plural, n ).str( get(), domain.id );
}

std::string translate( const char* context, const char* single, const char* plural, Int64 n, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( context, single, plural, n );
    return boost::locale::translate( context, single, plural, n ).str( get(), domain.id );
}

} // namespace MR::Locale
#endif
