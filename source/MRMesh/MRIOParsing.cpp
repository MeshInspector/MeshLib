#include "MRIOParsing.h"
#include "MRPch/MRTBB.h"
#include "MRVector3.h"


#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

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
            for ( auto ci = begin; ci < end; ci++ )
                if ( data[ci] == '\n' )
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

Expected<MR::Buffer<char>, std::string> readCharBuffer( std::istream& in )
{
    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Buffer<char> data( streamSize );
    // important on Windows: in stream must be open in binary mode, otherwise next will fail
    in.read( data.data(), ( ptrdiff_t )data.size() );
    if ( !in )
        return unexpected( std::string( "File read error" ) );

    return data;
}

VoidOrErrStr parseTextCoordinate( const std::string_view& str, Vector3f& v )
{
    using namespace boost::spirit::x3;

    int i = 0;
    auto coord = [&] ( auto& ctx ) { v[i++] = _attr( ctx ); };

    bool r = phrase_parse(
        str.begin(),
        str.end(),
        ( float_[coord] >> float_[coord] >> float_[coord] ),
        ascii::space | ascii::punct
    );
    if ( !r )
        return unexpected( "Failed to parse vertex" );

    return {};
}

VoidOrErrStr parseObjCoordinate( const std::string_view& str, Vector3f& v )
{
    using namespace boost::spirit::x3;

    int i = 0;
    auto coord = [&] ( auto& ctx ) { v[i++] = _attr( ctx ); };

    bool r = phrase_parse(
        str.begin(),
        str.end(),
        ( 'v' >> float_[coord] >> float_[coord] >> float_[coord] ),
        ascii::space
    );
    if ( !r )
        return unexpected( "Failed to parse vertex" );

    return {};
}

}