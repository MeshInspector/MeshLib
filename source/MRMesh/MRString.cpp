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

int calcDamerauLevenshteinDistance( const std::string& stringA, const std::string& stringB )
{
    std::vector<int> map( stringA.size() * stringB.size() );
    auto at = [&map,width = int( stringA.size() )] (int i, int j)->int&
    {
        return map[i + j * width];
    };

    for ( int i = 0; i < stringA.size(); ++i )
    {
        for ( int j = 0; j < stringB.size(); ++j )
        {
            if ( i == 0 || j == 0 )
                at( i, j ) = std::max( i, j );
            else if ( i > 1 && j > 1 && stringA[i] == stringB[j - 1] && stringA[i - 1] == stringB[j] )
            {
                at( i, j ) = std::min( { 
                    at( i - 1,j ) + 1,
                    at( i ,j - 1 ) + 1, 
                    at( i - 1,j - 1 ) + ( stringA[i] == stringB[j] ? 0 : 1 ),
                    at( i - 2,j - 2 ) + 1 } );
            }
            else
            {
                at( i, j ) = std::min( { 
                    at( i - 1,j ) + 1,
                    at( i ,j - 1 ) + 1,
                    at( i - 1,j - 1 ) + ( stringA[i] == stringB[j] ? 0 : 1 ) } );
            }
        }
    }
    return at( int( stringA.size() ) - 1, int( stringB.size() ) - 1 );
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