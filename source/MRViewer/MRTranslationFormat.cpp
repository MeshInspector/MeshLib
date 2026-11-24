#include "MRTranslationFormat.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <tuple>
#include <vector>

namespace MR::Locale
{

namespace
{

/// one use of an argument in a format string of fmt::format
struct FmtField
{
    int index = -1;        ///< for {} and {N}
    std::string_view name; ///< for {name}
    std::string_view spec; ///< after the colon

    bool operator ==( const FmtField& ) const = default;
    bool operator <( const FmtField& r ) const { return std::tie( index, name, spec ) < std::tie( r.index, r.name, r.spec ); }
};

/// returns the fields of a format string of fmt::format, or nothing if the string is not a valid one
std::optional<std::vector<FmtField>> parseFmtFields( std::string_view s )
{
    std::vector<FmtField> fields;
    int numAuto = 0;
    bool manual = false;
    auto addField = [&]( std::string_view id )
    {
        FmtField f;
        if ( id.empty() )
        {
            if ( manual )
                return false;
            f.index = numAuto++;
        }
        else if ( std::isdigit( ( unsigned char )id[0] ) )
        {
            const auto res = std::from_chars( id.data(), id.data() + id.size(), f.index );
            if ( numAuto > 0 || res.ec != std::errc() || res.ptr != id.data() + id.size() )
                return false;
            manual = true;
        }
        else
        {
            const auto isNameChar = []( char c ) { return c == '_' || std::isalnum( ( unsigned char )c ); };
            if ( !std::all_of( id.begin(), id.end(), isNameChar ) )
                return false;
            f.name = id;
        }
        fields.push_back( f );
        return true;
    };

    for ( size_t i = 0; i < s.size(); ++i )
    {
        const bool doubled = i + 1 < s.size() && s[i + 1] == s[i];
        if ( s[i] == '}' )
        {
            if ( !doubled )
                return {};
            ++i;
            continue;
        }
        if ( s[i] != '{' )
            continue;
        if ( doubled )
        {
            ++i;
            continue;
        }
        const auto idEnd = s.find_first_of( ":}", i + 1 );
        if ( idEnd == std::string_view::npos || !addField( s.substr( i + 1, idEnd - i - 1 ) ) )
            return {};
        const auto fieldPos = fields.size() - 1;
        auto end = idEnd;
        if ( s[idEnd] == ':' )
        {
            // dynamic width and precision are separate fields
            for ( end = idEnd + 1; end < s.size() && s[end] != '}'; ++end )
            {
                if ( s[end] != '{' )
                    continue;
                const auto nestedEnd = s.find( '}', end + 1 );
                if ( nestedEnd == std::string_view::npos || !addField( s.substr( end + 1, nestedEnd - end - 1 ) ) )
                    return {};
                end = nestedEnd;
            }
            if ( end >= s.size() )
                return {};
            fields[fieldPos].spec = s.substr( idEnd + 1, end - idEnd - 1 );
        }
        i = end;
    }
    return fields;
}

/// returns the conversion specifications of a format string of printf except for %%,
/// or nothing if the string has a percent sign that does not start a usual specification
std::optional<std::vector<std::string_view>> parsePrintfSpecs( std::string_view s )
{
    std::vector<std::string_view> specs;
    for ( size_t i = 0; i < s.size(); ++i )
    {
        if ( s[i] != '%' )
            continue;
        const auto start = i++;
        if ( i < s.size() && s[i] == '%' )
            continue;
        const auto skip = [&]( std::string_view chars ) { i = std::min( s.find_first_not_of( chars, i ), s.size() ); };
        skip( "-+#0" ); // flags except for space, which is not written on purpose: "50% smaller"
        skip( "0123456789*" ); // width
        if ( i < s.size() && s[i] == '.' )
        {
            ++i;
            skip( "0123456789*" ); // precision
        }
        skip( "hljztL" ); // length
        if ( i >= s.size() || std::string_view( "diouxXcsfFeEgGaAp" ).find( s[i] ) == std::string_view::npos )
            return {};
        specs.push_back( s.substr( start, i + 1 - start ) );
    }
    return specs;
}

} // anonymous namespace

bool isFormatSafeTranslation( std::string_view original, std::string_view translation )
{
    if ( translation == original )
        return true;
    if ( original.find_first_of( "{}" ) != std::string_view::npos )
    {
        // the original with wrong braces is not a format string of fmt::format
        if ( auto origFields = parseFmtFields( original ) )
        {
            auto fields = parseFmtFields( translation );
            if ( !fields )
                return false;
            std::sort( origFields->begin(), origFields->end() );
            std::sort( fields->begin(), fields->end() );
            if ( *fields != *origFields )
                return false;
            // the original with fmt fields is not passed to printf, where its percent signs would need doubling
            if ( !fields->empty() )
                return true;
        }
    }
    if ( original.find( '%' ) == std::string_view::npos )
        return true;
    // the original with a single percent sign is not a format string of printf
    const auto origSpecs = parsePrintfSpecs( original );
    return !origSpecs || parsePrintfSpecs( translation ) == origSpecs;
}

} // namespace MR::Locale
