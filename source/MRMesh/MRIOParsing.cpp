#include "MRIOParsing.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRString.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

#include <boost/spirit/home/x3.hpp>

// helper macro to make code cleaner
#define floatT real_parser<T>{}

namespace
{

using namespace MR;

constexpr int MaxErrorStringLen = 80;

template <typename T>
Expected<void> readFromStream( std::istream& in, T& out )
{
    const auto streamSize = getStreamSize( in );
    if ( !in )
        return unexpected( std::string( "File read error" ) );

    out.resize( streamSize );
    // important on Windows: in stream must be open in binary mode, otherwise next will fail
    in.read( out.data(), (ptrdiff_t)out.size() );
    if ( !in )
        return unexpected( std::string( "File read error" ) );

    return {};
}

} // namespace

namespace MR
{

std::vector<size_t> splitByLines( const char* data, size_t size )
{
    constexpr size_t blockSize = 4096;
    const auto blockCount = ( size_t )std::ceil( ( float )size / blockSize );

    constexpr size_t maxGroupCount = 256;
    const auto blocksPerGroup = ( size_t )std::ceil( ( float )blockCount / maxGroupCount );
    const auto groupSize = blockSize * blocksPerGroup;
    const auto groupCount = ( size_t )std::ceil( ( float )size / groupSize );
    assert( groupCount <= maxGroupCount );
    assert( groupSize * groupCount >= size );
    assert( groupSize * ( groupCount - 1 ) < size );

    std::vector<std::vector<size_t>> groups( groupCount );
    tbb::task_group taskGroup;
    for ( size_t gi = 0; gi < groupCount; gi++ )
    {
        taskGroup.run( [&, i = gi]
        {
            std::vector<size_t> group;
            const auto begin = i * groupSize;
            const auto end = std::min( ( i + 1 ) * groupSize, size );
            for ( auto ci = begin; ci < end; ci++ ) // support all 3: '\n', '\r\n' and '\r' line endings
                if ( data[ci] == '\n' || ( data[ci] == '\r' && ( ci + 1 == size || data[ci + 1] != '\n' ) ) )
                    group.emplace_back( ci + 1 );
            groups[i] = std::move( group );
        } );
    }
    taskGroup.wait();

    std::vector<size_t> newlines{ 0 };
    auto sum = newlines.size();
    std::vector<size_t> groupOffsets;
    for ( const auto& group : groups )
    {
        groupOffsets.emplace_back( sum );
        sum += group.size();
    }
    newlines.resize( sum );

    for ( size_t gi = 0; gi < groupCount; gi++ )
    {
        taskGroup.run( [&, i = gi]
        {
            const auto& group = groups[i];
            const auto offset = groupOffsets[i];
            for ( auto li = 0; li < group.size(); li++ )
                newlines[offset + li] = group[li];
        } );
    }
    taskGroup.wait();

    // add finish line
    if ( newlines.back() != size )
        newlines.emplace_back( size );

    return newlines;
}

std::streamoff getStreamSize( std::istream& in )
{
    const auto posStart = in.tellg();
    in.seekg( 0, std::ios::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    return posEnd - posStart;
}

Expected<std::string> readString( std::istream& in )
{
    MR_TIMER;
    std::string str;
    if ( auto result = readFromStream( in, str ); !result )
        return unexpected( std::move( result.error() ) );
    return str;
}

Expected<Buffer<char>> readCharBuffer( std::istream& in )
{
    Buffer<char> buf;
    if ( auto result = readFromStream( in, buf ); !result )
        return unexpected( std::move( result.error() ) );
    return buf;
}

template <typename T>
Expected<void> parseTextCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n, Color* c )
{
    using namespace boost::spirit::x3;

    int vi = 0;
    const auto coord = [&] ( auto& ctx ) { v[vi++] = _attr( ctx ); };
    int ni = 0;
    const auto normal = [&] ( auto& ctx ) { if ( n ) ( *n )[ni++] = _attr( ctx ); };
    int ci = 0;
    const auto color = [&] ( auto& ctx ) { if ( c ) ( *c )[ci++] = uint8_t( _attr( ctx ) ); };

    const auto spaceRule = ( ascii::space | ',' | ';' );
    const auto coordRule = ( floatT[coord] >> floatT[coord] >> floatT[coord] );
    const auto normalRule = ( floatT[normal] >> floatT[normal] >> floatT[normal] );
    const auto colorRule = ( floatT[color] >> floatT[color] >> floatT[color] );

    bool r;
    if ( c != nullptr )
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( coordRule >> -( normalRule >> -colorRule ) ),
            spaceRule
        );
    }
    else if ( n != nullptr )
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( coordRule >> -normalRule ),
            spaceRule
        );
    }
    else
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            coordRule,
            spaceRule
        );
    }
    if ( !r )
        return unexpected( "Failed to parse coord" );

    // explicitly set alpha value if color is present
    if ( c != nullptr && ci == 3 )
        c->a = 255;

    return {};
}

template <typename T>
Expected<void> parseObjCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* c )
{
    using namespace boost::spirit::x3;

    int i = 0;
    auto coord = [&] ( auto& ctx ) { v[i++] = _attr( ctx ); };
    int j = 0;
    auto color = [&] ( auto& ctx ) { if ( c ) ( *c )[j++] = _attr( ctx ); };

    bool r;
    if ( c != nullptr )
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> floatT[coord] >> floatT[coord] >> floatT[coord] >> -( floatT[color] >> floatT[color] >> floatT[color] ) ),
            ascii::space
        );
    }
    else
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> floatT[coord] >> floatT[coord] >> floatT[coord] ),
            ascii::space
        );
    }
    if ( !r )
        return unexpected( "Failed to parse vertex: " + std::string( trimRight( str.substr( 0, MaxErrorStringLen ) ) ) );

    return {};
}

template<typename T>
Expected<void> parsePtsCoordinate( const std::string_view& str, Vector3<T>& v, Color& c )
{
    using namespace boost::spirit::x3;

    int i = 0;
    auto coord = [&] ( auto& ctx ){ v[i++] = _attr( ctx ); };
    auto skip_pos = [&] ( auto& ){ i++;};
    auto col = [&] ( auto& ctx ) { ((uint8_t*)&c)[i++ -4] = _attr( ctx ); };

    using uint8_type = uint_parser<uint8_t>;
    constexpr uint8_type uint8_ = {};
    bool r = phrase_parse(
        str.begin(),
        str.end(),
        (
            floatT[coord] >> floatT[coord] >> floatT[coord] >>
            double_[skip_pos] >>
            uint8_[col] >> uint8_[col] >> uint8_[col] ),
        ascii::space
    );
    if ( !r )
        return unexpected( "Failed to parse vertex: " + std::string( trimRight( str.substr( 0, MaxErrorStringLen ) ) ) );

    return {};
}

template<typename T>
Expected<void> parseSingleNumber( const std::string_view& str, T& num )
{
    using namespace boost::spirit::x3;

    auto coord = [&] ( auto& ctx ) { num = _attr( ctx ); };

    bool r{};

    if constexpr ( std::is_same_v<T, int> )
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( int_parser<T>{}[coord] ),
            ascii::space
        );
    }
    else
    {
        r = phrase_parse(
            str.begin(),
            str.end(),
            ( floatT[coord] ),
            ascii::space
        );
    }

    if ( !r )
        return unexpected( "Failed to parse number" );

    return {};
}

Expected<void> parseFirstNum( const std::string_view& str, int& num )
{
    using namespace boost::spirit::x3;

    auto n = [&] ( auto& ctx )
    {
        num = _attr( ctx );
    };
    bool r = phrase_parse(
        str.begin(),
        str.end(),
        int_[n],
        ascii::space );

    if ( !r )
    {
        return unexpected( "Failed to parse face in OFF-file" );
    }

    return {};
}

Expected<void> parsePolygon( const std::string_view& str, VertId* vertId, int* numPoints)
{
    using namespace boost::spirit::x3;

    bool r = false;
    size_t vi = 0;
    auto v = [&] ( auto& ctx ) { vertId[vi++] = VertId( _attr( ctx ) ); };
    if ( numPoints )
    {
        auto n = [&] ( auto& ctx ) { *numPoints = _attr( ctx ); };
        r = phrase_parse(
            str.begin(),
            str.end(),
            int_[n] >> *int_[v],
            ascii::space );
    }
    else
    {
        auto n = [&] ( auto& ) { };
        r = phrase_parse(
            str.begin(),
            str.end(),
            int_[n] >> *int_[v],
            ascii::space );
    }

    if ( !r )
    {
        return unexpected( "Failed to parse face in OFF-file" );
    }

    return {};
}

template <typename T>
Expected<void> parseAscCoordinate( const std::string_view& str, Vector3<T>& v, Vector3<T>* n, Color* c )
{
    return parseTextCoordinate( str, v, n, c );
}

template Expected<void> parseSingleNumber<float>( const std::string_view& str, float& num );
template Expected<void> parseSingleNumber<int>( const std::string_view& str, int& num );

template Expected<void> parsePtsCoordinate<float>( const std::string_view& str, Vector3f& v, Color& c );
template Expected<void> parsePtsCoordinate<double>( const std::string_view& str, Vector3d& v, Color& c );

template Expected<void> parseTextCoordinate<float>( const std::string_view& str, Vector3f& v, Vector3f* n, Color* c );
template Expected<void> parseTextCoordinate<double>( const std::string_view& str, Vector3d& v, Vector3d* n, Color* c );

template Expected<void> parseObjCoordinate<float>( const std::string_view& str, Vector3f& v, Vector3f* c );
template Expected<void> parseObjCoordinate<double>( const std::string_view& str, Vector3d& v, Vector3d* c );

template Expected<void> parseAscCoordinate<float>( const std::string_view& str, Vector3f& v, Vector3f* n, Color* c );
template Expected<void> parseAscCoordinate<double>( const std::string_view& str, Vector3d& v, Vector3d* n, Color* c );

}
