#include "MRI18n.h"
#ifndef MRVIEWER_NO_LOCALE
#include "MRLocale.h"

#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )

#include <cassert>
#include <cstring>

namespace MR::Locale
{

namespace
{

inline const char* asCStr( std::string_view sv )
{
    assert( std::strlen( sv.data() ) == sv.size() );
    return sv.data();
}

} // namespace

std::string translate( std::string_view msg, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( asCStr( msg ) );
    return boost::locale::translate( asCStr( msg ) ).str( get(), domain.id );
}

std::string translate( std::string_view context, std::string_view msg, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( asCStr( context ), asCStr( msg ) );
    return boost::locale::translate( asCStr( context ), asCStr( msg ) ).str( get(), domain.id );
}

std::string translate( std::string_view single, std::string_view plural, Int64 n, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( asCStr( single ), asCStr( plural ), n );
    return boost::locale::translate( asCStr( single ), asCStr( plural ), n ).str( get(), domain.id );
}

std::string translate( std::string_view context, std::string_view single, std::string_view plural, Int64 n, Domain domain )
{
    if ( domain.id < 0 )
        return translate_noop( asCStr( context ), asCStr( single ), asCStr( plural ), n );
    return boost::locale::translate( asCStr( context ), asCStr( single ), asCStr( plural ), n ).str( get(), domain.id );
}

} // namespace MR::Locale
#endif
