#include "MRString.h"
#include <algorithm>

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

std::vector<std::string> split( const std::string& string, const std::string& delimiter )
{
    std::vector<std::string> res;
    size_t pos = 0;
    for ( ;;)
    {
        auto delimPos = string.find( delimiter, pos );
        res.push_back( string.substr( pos, delimPos - pos ) );
        if ( delimPos == std::string::npos )
            break;
        pos = delimPos + delimiter.size();
    }

    return res;
}

}