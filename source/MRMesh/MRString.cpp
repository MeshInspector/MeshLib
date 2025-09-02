#include "MRString.h"
#include <algorithm>
#include <limits>

namespace
{

enum class PatchType : int
{
    None = -1,
    Deletion,
    Insertion,
    Substitution,
    Transposition,
    Count
};

struct SumPatchWeight
{
    PatchType type{ PatchType::None };
    int prevI{ 0 };
    int prevJ{ 0 };
    int w{ 0 };
};

bool isAscii( char ch )
{
    return (unsigned char)ch <= 127;
}

} // namespace

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
    bool caseSensitive, int* outLeftRightAddition )
{
    std::vector<SumPatchWeight> map( ( stringA.size() + 1 ) * ( stringB.size() + 1 ) );

    auto at = [&map, width = int( stringA.size() + 1 )] ( int i, int j )->SumPatchWeight&
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
            {
                auto& sw = at( i, j );
                if ( i < j )
                {
                    sw.prevJ = j - 1;
                    sw.type = PatchType::Insertion;
                }
                else if ( j < i )
                {
                    sw.prevI = i - 1;
                    sw.type = PatchType::Deletion;
                }
                sw.w = std::max( i, j );
            }
            else
            {
                std::array<int, int( PatchType::Count )> candidates = {
                    std::numeric_limits<int>::max(), std::numeric_limits<int>::max() ,
                    std::numeric_limits<int>::max(), std::numeric_limits<int>::max() };
                candidates[int( PatchType::Deletion )] = at( i - 1, j ).w + 1; // deletion
                candidates[int( PatchType::Insertion )] = at( i, j - 1 ).w + 1; // insertion
                candidates[int( PatchType::Substitution )] = at( i - 1, j - 1 ).w + ( copm( i, j ) ? 0 : 1 ); // substitution
                if ( i > 1 && j > 1 && copm( i, j - 1 ) && copm( i - 1, j ) )
                    candidates[int( PatchType::Transposition )] = at( i - 2, j - 2 ).w + 1; // transposition

                auto minE = std::min_element( candidates.begin(), candidates.end() );
                auto& sw = at( i, j );
                sw.type = PatchType( std::distance( candidates.begin(), minE ) );
                sw.w = *minE;
                if ( sw.type == PatchType::Deletion )
                {
                    sw.prevI = i - 1;
                    sw.prevJ = j;
                }
                else if ( sw.type == PatchType::Insertion )
                {
                    sw.prevI = i;
                    sw.prevJ = j - 1;
                }
                else if ( sw.type == PatchType::Substitution )
                {
                    sw.prevI = i - 1;
                    sw.prevJ = j - 1;
                }
                else if ( sw.type == PatchType::Transposition )
                {
                    sw.prevI = i - 2;
                    sw.prevJ = j - 2;
                }
            }
        }
    }
    if ( outLeftRightAddition )
    {
        *outLeftRightAddition = 0;
        PatchType lastPatch = PatchType::Count;
        bool firstStrike = true;
        int i = int( stringA.size() );
        int j = int( stringB.size() );
        int currentInsertionStrike = 0;
        for ( ;; )
        {
            const auto& sw = at( i, j );
            lastPatch = sw.type;
            if ( lastPatch == PatchType::None )
                break;
            if ( lastPatch == PatchType::Insertion )
            {
                if ( firstStrike )
                    ( *outLeftRightAddition )++;
                else
                    ++currentInsertionStrike;
            }
            else
            {
                firstStrike = false;
                currentInsertionStrike = 0;
            }
            i = sw.prevI;
            j = sw.prevJ;
        }
        ( *outLeftRightAddition ) += currentInsertionStrike;
    }
    return at( int( stringA.size() ), int( stringB.size() ) ).w;
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

std::string_view trim( std::string_view str )
{
    return trimRight( trimLeft( str ) );
}

std::string_view trimLeft( std::string_view str )
{
    size_t pos = 0;
    while ( pos < str.size() && isAscii( str[pos] ) && std::isspace( str[pos] ) )
        ++pos;
    return str.substr( pos );
}

std::string_view trimRight( std::string_view str )
{
    auto l = str.size();
    while ( l > 0 && isAscii( str[l - 1] ) && std::isspace( str[l - 1] ) )
        --l;
    return str.substr( 0, l );
}

}
