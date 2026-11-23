#include "MRTranslationFormat.h"
#include <algorithm>
#include <charconv>
#include <vector>

namespace MR::Locale
{

namespace
{

bool isDigit( char c )
{
    return c >= '0' && c <= '9';
}

/// one use of an argument in a format string of fmt::format
struct FmtField
{
    int index = -1;        ///< for {} and {N}
    std::string_view name; ///< for {name}
    std::string_view spec; ///< after the colon
    bool nested = false;   ///< dynamic width or precision inside the spec of another field
};

/// returns false if the string is not a valid format string of fmt::format
bool parseFmtFields( std::string_view s, std::vector<FmtField>& fields )
{
    int numAuto = 0;
    bool manual = false;
    auto addField = [&]( std::string_view id, bool nested )
    {
        FmtField f;
        f.nested = nested;
        if ( id.empty() )
        {
            if ( manual )
                return false;
            f.index = numAuto++;
        }
        else if ( isDigit( id[0] ) )
        {
            const auto res = std::from_chars( id.data(), id.data() + id.size(), f.index );
            if ( numAuto > 0 || res.ec != std::errc() || res.ptr != id.data() + id.size() )
                return false;
            manual = true;
        }
        else
        {
            const auto isNameChar = []( char c ) { return isDigit( c ) || c == '_' || ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ); };
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
                return false;
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
        if ( idEnd == std::string_view::npos || !addField( s.substr( i + 1, idEnd - i - 1 ), false ) )
            return false;
        const auto fieldPos = fields.size() - 1;
        auto end = idEnd;
        if ( s[idEnd] == ':' )
        {
            for ( end = idEnd + 1; end < s.size() && s[end] != '}'; ++end )
            {
                if ( s[end] != '{' )
                    continue;
                const auto nestedEnd = s.find( '}', end + 1 );
                if ( nestedEnd == std::string_view::npos || !addField( s.substr( end + 1, nestedEnd - end - 1 ), true ) )
                    return false;
                end = nestedEnd;
            }
            if ( end >= s.size() )
                return false;
            fields[fieldPos].spec = s.substr( idEnd + 1, end - idEnd - 1 );
        }
        i = end;
    }
    return true;
}

/// the type of an argument taken by printf
enum class PrintfArg
{
    Int,
    Long,
    LongLong,
    Size,
    Double,
    LongDouble,
    String,
    WideString,
    Pointer,
    Count
};

/// returns false if the string has a percent sign that is not a part of a conversion specification;
/// if \p strict then the specifications never written on purpose (with space flag) are not accepted too
bool parsePrintfArgs( std::string_view s, bool strict, std::vector<PrintfArg>& args )
{
    for ( size_t i = 0; i < s.size(); ++i )
    {
        if ( s[i] != '%' )
            continue;
        ++i;
        if ( i < s.size() && s[i] == '%' )
            continue;
        const auto isFlag = [strict]( char c ) { return c == '-' || c == '+' || c == '#' || c == '0' || ( !strict && c == ' ' ); };
        while ( i < s.size() && isFlag( s[i] ) )
            ++i;
        auto skipNumber = [&]
        {
            if ( i < s.size() && s[i] == '*' )
            {
                args.push_back( PrintfArg::Int );
                ++i;
            }
            else
            {
                while ( i < s.size() && isDigit( s[i] ) )
                    ++i;
            }
        };
        skipNumber();
        if ( i < s.size() && s[i] == '.' )
        {
            ++i;
            skipNumber();
        }
        auto intArg = PrintfArg::Int;
        bool isLong = false, isLongDouble = false;
        const auto has = [&]( std::string_view length )
        {
            if ( s.substr( i, length.size() ) != length )
                return false;
            i += length.size();
            return true;
        };
        if ( has( "hh" ) || has( "h" ) )
            intArg = PrintfArg::Int;
        else if ( has( "ll" ) || has( "j" ) )
            intArg = PrintfArg::LongLong;
        else if ( has( "l" ) )
        {
            intArg = PrintfArg::Long;
            isLong = true;
        }
        else if ( has( "z" ) || has( "t" ) )
            intArg = PrintfArg::Size;
        else if ( has( "L" ) )
            isLongDouble = true;
        if ( i >= s.size() )
            return false;
        switch ( s[i] )
        {
        case 'd': case 'i': case 'o': case 'u': case 'x': case 'X':
            args.push_back( intArg );
            break;
        case 'c':
            args.push_back( PrintfArg::Int );
            break;
        case 'e': case 'E': case 'f': case 'F': case 'g': case 'G': case 'a': case 'A':
            args.push_back( isLongDouble ? PrintfArg::LongDouble : PrintfArg::Double );
            break;
        case 's':
            args.push_back( isLong ? PrintfArg::WideString : PrintfArg::String );
            break;
        case 'p':
            args.push_back( PrintfArg::Pointer );
            break;
        case 'n':
            args.push_back( PrintfArg::Count );
            break;
        default:
            return false;
        }
    }
    return true;
}

bool isPrintfSafe( std::string_view original, std::string_view translation )
{
    std::vector<PrintfArg> origArgs, args;
    if ( original.find( '%' ) == std::string_view::npos || !parsePrintfArgs( original, true, origArgs ) )
        return true; // the original is not a format string
    if ( !parsePrintfArgs( translation, false, args ) )
        return false;
    // the last arguments may be not used
    return args.size() <= origArgs.size() && std::equal( args.begin(), args.end(), origArgs.begin() );
}

} // anonymous namespace

bool isFormatSafeTranslation( std::string_view original, std::string_view translation )
{
    if ( translation.find_first_of( "%{}" ) == std::string_view::npos || translation == original )
        return true;
    std::vector<FmtField> origFields, fields;
    // the original without braces or with wrong ones is not a format string
    if ( original.find_first_of( "{}" ) != std::string_view::npos && parseFmtFields( original, origFields ) )
    {
        if ( !parseFmtFields( translation, fields ) )
            return false;
        // every argument must be used as in the original; the empty spec suits any type
        const bool sameUse = std::all_of( fields.begin(), fields.end(), [&]( const FmtField& f )
        {
            return std::any_of( origFields.begin(), origFields.end(), [&]( const FmtField& o )
            {
                return o.index == f.index && o.name == f.name && o.nested == f.nested && ( f.spec.empty() || f.spec == o.spec );
            } );
        } );
        if ( !sameUse )
            return false;
        // the original with fmt fields is not passed to printf, where its percent signs would need doubling
        if ( !origFields.empty() )
            return true;
    }
    return isPrintfSafe( original, translation );
}

} // namespace MR::Locale
