#include "MRString.h"
namespace MR
{

bool findSubstring( const std::string& string, const std::string& substring )
{
    return ( std::search( string.begin(), string.end(),
        substring.begin(), substring.end(),
        [] ( char ch1, char ch2 )
    {
        return std::toupper( ch1 ) == std::toupper( ch2 );
    } ) != string.end() );
}

}