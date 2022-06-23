#include "MRString.h"
#include <ranges>

namespace MR
{

size_t findSubstringCaseInsensitive( const std::string& string, const std::string& substring )
{
    auto iter = std::search( string.begin(), string.end(),
        substring.begin(), substring.end(),
        [] ( char ch1, char ch2 )
    {
        return std::toupper( ch1 ) == std::toupper( ch2 );
    } );
    if ( iter == string.end() )
        return std::string::npos;
    return std::distance( string.begin(), iter );
}

std::vector<std::string> split( const std::string& string, char delimiter )
{
    auto to_string = [] ( auto&& r ) -> std::string
    {
        const auto data = &*r.begin();
        const auto size = std::size_t( std::ranges::distance( r ) );

        return std::string{ data, size };
    };

    auto range = string |
        std::ranges::views::split( delimiter ) |
        std::ranges::views::transform( to_string );

    return { std::ranges::begin( range ), std::ranges::end( range ) };
}

}