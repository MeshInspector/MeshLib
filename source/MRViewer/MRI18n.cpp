#include "MRI18n.h"
#include "MRLocale.h"

#ifndef MRVIEWER_NO_LOCALE
#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )
#endif

#include <cassert>
#include <cstring>

namespace MR::Locale
{

namespace
{

inline const char* asCStr( const std::string_view& sv )
{
    assert( std::strlen( sv.data() ) == sv.size() );
    return sv.data();
}

} // namespace

std::string translate( std::string_view msg, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
        return boost::locale::translate( asCStr( msg ) ).str( get(), domainId.get() );
#endif
    return translate_noop( asCStr( msg ) );
}

std::string translate( std::string_view context, std::string_view msg, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
        return boost::locale::translate( asCStr( context ), asCStr( msg ) ).str( get(), domainId.get() );
#endif
    return translate_noop( asCStr( context ), asCStr( msg ) );
}

std::string translate( std::string_view single, std::string_view plural, Int64 n, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
        return boost::locale::translate( asCStr( single ), asCStr( plural ), n ).str( get(), domainId.get() );
#endif
    return translate_noop( asCStr( single ), asCStr( plural ), n );
}

std::string translate( std::string_view context, std::string_view single, std::string_view plural, Int64 n, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
        return boost::locale::translate( asCStr( context ), asCStr( single ), asCStr( plural ), n ).str( get(), domainId.get() );
#endif
    return translate_noop( asCStr( context ), asCStr( single ), asCStr( plural ), n );
}

} // namespace MR::Locale
