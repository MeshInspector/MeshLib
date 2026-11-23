#include "MRI18n.h"
#include "MRLocale.h"

#include "MRMesh/MRTranslationFormat.h"
#include "MRPch/MRSpdlog.h"

#ifndef MRVIEWER_NO_LOCALE
#pragma warning( push )
#pragma warning( disable: 4619 ) // #pragma warning: there is no warning number 'N'
#include <boost/locale/message.hpp>
#pragma warning( pop )
#endif

#include <cassert>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace MR::Locale
{

namespace
{

inline const char* asCStr( const std::string_view& sv )
{
    assert( std::strlen( sv.data() ) == sv.size() );
    return sv.data();
}

#ifndef MRVIEWER_NO_LOCALE
/// returns false and reports once the translation that cannot replace the original string in fmt::format or printf call
bool isSafe( const std::string& translation, std::string_view msg, std::string_view plural = {} )
{
    if ( isFormatSafeTranslation( msg, translation ) || ( !plural.empty() && isFormatSafeTranslation( plural, translation ) ) )
        return true;
    static std::mutex mutex;
    static std::unordered_set<std::string> reported;
    std::unique_lock lock( mutex );
    if ( reported.insert( translation ).second )
        spdlog::warn( "Translation \"{}\" of \"{}\" is ignored: its format placeholders do not match the original", translation, msg );
    return false;
}
#endif

} // namespace

std::string translate( std::string_view msg, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
    {
        auto res = boost::locale::translate( asCStr( msg ) ).str( get(), domainId.get() );
        if ( isSafe( res, msg ) )
            return res;
    }
    if ( domainId == genericDomain )
    {
        for ( const auto genericDomainId : getGenericDomains() )
            if ( const auto str = translate( msg, genericDomainId ); str != translate_noop( asCStr( msg ) ) )
                return str;
    }
#else
    (void)domainId;
#endif
    return translate_noop( asCStr( msg ) );
}

std::string translate( std::string_view context, std::string_view msg, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
    {
        auto res = boost::locale::translate( asCStr( context ), asCStr( msg ) ).str( get(), domainId.get() );
        if ( isSafe( res, msg ) )
            return res;
    }
    if ( domainId == genericDomain )
    {
        for ( const auto genericDomainId : getGenericDomains() )
            if ( const auto str = translate( context, msg, genericDomainId ); str != translate_noop( asCStr( context ), asCStr( msg ) ) )
                return str;
    }
#else
    (void)domainId;
#endif
    return translate_noop( asCStr( context ), asCStr( msg ) );
}

std::string translate( std::string_view single, std::string_view plural, std::int64_t n, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
    {
        auto res = boost::locale::translate( asCStr( single ), asCStr( plural ), n ).str( get(), domainId.get() );
        if ( isSafe( res, single, plural ) )
            return res;
    }
    if ( domainId == genericDomain )
    {
        for ( const auto genericDomainId : getGenericDomains() )
            if ( const auto str = translate( single, plural, n, genericDomainId ); str != translate_noop( asCStr( single ), asCStr( plural ), n ) )
                return str;
    }
#else
    (void)domainId;
#endif
    return translate_noop( asCStr( single ), asCStr( plural ), n );
}

std::string translate( std::string_view context, std::string_view single, std::string_view plural, std::int64_t n, LocaleDomainId domainId )
{
#ifndef MRVIEWER_NO_LOCALE
    if ( domainId.valid() )
    {
        auto res = boost::locale::translate( asCStr( context ), asCStr( single ), asCStr( plural ), n ).str( get(), domainId.get() );
        if ( isSafe( res, single, plural ) )
            return res;
    }
    if ( domainId == genericDomain )
    {
        for ( const auto genericDomainId : getGenericDomains() )
            if ( const auto str = translate( context, single, plural, n, genericDomainId ); str != translate_noop( asCStr( context ), asCStr( single ), asCStr( plural ), n ) )
                return str;
    }
#else
    (void)domainId;
#endif
    return translate_noop( asCStr( context ), asCStr( single ), asCStr( plural ), n );
}

} // namespace MR::Locale
