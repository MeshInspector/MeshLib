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

int calcDamerauLevenshteinDistance( const std::string& stringA, const std::string& stringB,
    bool caseSensitive )
{
    std::vector<int> map( ( stringA.size() + 1 ) * ( stringB.size() + 1 ) );
    auto at = [&map, width = int( stringA.size() + 1 )] ( int i, int j )->int&
    {
        return map[i + j * width];
    };

    auto copm = [&] ( int i, int j )->bool
    {
        if ( caseSensitive )
            return stringA[i - 1] == stringB[j - 1];
        else
            return std::tolower( stringA[i - 1] ) == std::tolower( stringB[j - 1] );
    };

    for ( int i = 0; i < stringA.size() + 1; ++i )
    {
        for ( int j = 0; j < stringB.size() + 1; ++j )
        {
            if ( i == 0 || j == 0 )
                at( i, j ) = std::max( i, j );
            else if ( i > 1 && j > 1 && copm( i, j - 1 ) && copm( i - 1, j ) )
            {
                at( i, j ) = std::min( {
                    at( i - 1,j ) + 1,
                    at( i ,j - 1 ) + 1,
                    at( i - 1,j - 1 ) + ( copm( i, j ) ? 0 : 1 ),
                    at( i - 2,j - 2 ) + 1 } );
            }
            else
            {
                at( i, j ) = std::min( {
                    at( i - 1,j ) + 1,
                    at( i ,j - 1 ) + 1,
                    at( i - 1,j - 1 ) + ( copm( i, j ) ? 0 : 1 ) } );
            }
        }
    }
    return at( int( stringA.size() ), int( stringB.size() ) );
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

std::string replace( std::string target, std::string_view from, std::string_view to )
{
    std::string ret;
    bool first = true;
    split( target, from, [&]( std::string_view segment )
    {
        if ( first )
            first = false;
        else
            ret += to;
        ret += segment;
        return false;
    } );
    return ret;
}

void replaceInplace( std::string& target, std::string_view from, std::string_view to )
{
    target = replace( std::move( target ), from, to );
}

std::string_view trimRight( std::string_view str )
{
    auto l = str.size();
    while ( l > 0 && ( str[l-1] == ' ' || str[l-1] == '\t' || str[l-1] == '\n' || str[l-1] == '\r' ) )
        --l;
    return str.substr( 0, l );
}

}
