#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRPch/MRTBB.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

namespace
{
    using namespace MR;

    tl::expected<void, std::string> parseObjVertex( const std::string_view& str, Vector3f& v )
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
            return tl::make_unexpected( "Failed to parse vertex in OBJ-file" );

        return {};
    }

    struct ObjFace
    {
        std::vector<int> vertices;
        std::vector<int> textures;
        std::vector<int> normals;
    };

    tl::expected<void, std::string> parseObjFace( const std::string_view& str, ObjFace& f )
    {
        using namespace boost::spirit::x3;

        auto v = [&] ( auto& ctx ) { f.vertices.emplace_back( _attr( ctx ) ); };
        auto vt = [&] ( auto& ctx ) { f.textures.emplace_back( _attr( ctx ) ); };
        auto vn = [&] ( auto& ctx ) { f.normals.emplace_back( _attr( ctx ) ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            // NOTE: actions are not being reverted after backtracking
            // https://github.com/boostorg/spirit/issues/378
            ( 'f' >> *( int_[v] >> -( '/' >> ( ( int_[vt] >> -( '/' >> int_[vn] ) ) | ( '/' >> int_[vn] ) ) ) ) ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse face in OBJ-file" );

        if ( f.vertices.empty() )
            return tl::make_unexpected( "Invalid face vertex count in OBJ-file" );
        if ( !f.textures.empty() && f.textures.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face texture count in OBJ-file" );
        if ( !f.normals.empty() && f.normals.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face normal count in OBJ-file" );
        return {};
    }
}

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Buffer<char> data( streamSize );
    in.read( data.data(), (ptrdiff_t)data.size() );
    if ( !in )
        return tl::make_unexpected( std::string( "OBJ-format read error" ) );

    if ( !callback( 0.25f ) )
        return tl::make_unexpected( "Loading canceled" );
    // TODO: redefine callback

    return fromSceneObjFile( data.data(), data.size(), combineAllObjects, callback );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    Triangulation t;

    auto finishObject = [&]() 
    {
        MR_NAMED_TIMER( "finish object" )
        if ( !t.empty() )
        {
            res.emplace_back();
            res.back().name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : t )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            for ( auto & vs : t )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            res.back().mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), t );
            t.clear();
        }
        currentObjName.clear();
    };

    Timer timer( "split by lines" );
    std::vector<size_t> newlines{ 0 };
    {
        constexpr size_t blockSize = 4096;
        const auto blockCount = (size_t)std::ceil( (float)size / blockSize );
        constexpr size_t maxGroupCount = 256;
        const auto blocksPerGroup = (size_t)std::ceil( (float)blockCount / maxGroupCount );
        const auto groupSize = blockSize * blocksPerGroup;
        const auto groupCount = (size_t)std::ceil( (float)size / groupSize );
        assert( groupCount <= maxGroupCount );
        assert( groupSize * groupCount >= size );
        assert( groupSize * ( groupCount - 1 ) < size );

        std::vector<std::vector<size_t>> groups( groupCount );
        tbb::task_group taskGroup;
        for ( size_t gi = 0; gi < groupCount; gi++ )
        {
            taskGroup.run( [&, i = gi]
            {
                auto& group = groups[i];
                const auto begin = i * groupSize;
                const auto end = std::min( ( i + 1 ) * groupSize, size );
                for ( auto ci = begin; ci < end; ci++ )
                    if ( data[ci] == '\n' )
                        group.emplace_back( ci + 1 );
            } );
        }
        taskGroup.wait();

        size_t sum = 1;
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
    }
    // add finish line
    if ( newlines.back() != size )
        newlines.emplace_back( size );
    const auto lineCount = newlines.size() - 1;

    if ( !callback( 0.40f ) )
        return tl::make_unexpected( "Loading canceled" );

    timer.restart( "group element lines" );
    enum Element
    {
        Unknown,
        Vertex,
        Face,
        Object,
    };
    struct ElementGroup
    {
        Element element;
        size_t begin;
        size_t end;
    };
    std::vector<ElementGroup> groups{ { Unknown, 0, 0 } }; // emplace stub initial group
    for ( size_t li = 0; li < lineCount; ++li )
    {
        auto* line = data + newlines[li];

        Element element = Unknown;
        if ( line[0] == 'v' && line[1] != 'n' /*normals*/ && line[1] != 't' /*texture coordinates*/ )
        {
            element = Vertex;
        }
        else if ( line[0] == 'f' )
        {
            element = Face;
        }
        else if ( line[0] == 'o' )
        {
            element = Object;
        }
        // TODO: multi-line elements

        if ( element != groups.back().element )
        {
            groups.back().end = li;
            groups.push_back( { element, li, 0 } );
        }
    }
    groups.back().end = lineCount;

    if ( !callback( 0.50f ) )
        return tl::make_unexpected( "Loading canceled" );

    auto parseVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        const auto offset = points.size();
        points.resize( points.size() + ( end - begin ) );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3f v;
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjVertex( line, v );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                points[offset + ( li - begin )] = v;
            }
        }, ctx );
    };

    auto parseFaces = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        tbb::task_group_context ctx;
        tbb::enumerable_thread_specific<Triangulation> trisPerThread;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& tris = trisPerThread.local();

            ObjFace f;
            // usually a face has 3 or 4 vertices
            for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                elements->reserve( 4 );

            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                    elements->clear();

                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjFace( line, f );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }

                auto& vs = f.vertices;
                for ( auto& v : vs )
                {
                    if ( v < 0 )
                        v += (int)points.size() + 1;

                    if ( v <= 0 )
                    {
                        if ( ctx.cancel_group_execution() )
                            parseError = "Too negative vertex ID in OBJ-file";
                        return;
                    }
                }
                if ( vs.size() < 3 )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = "Face with less than 3 vertices in OBJ-file";
                    return;
                }

                // TODO: make smarter triangulation based on point coordinates
                for ( int j = 1; j + 1 < vs.size(); ++j )
                    tris.push_back( { VertId( vs[0]-1 ), VertId( vs[j]-1 ), VertId( vs[j+1]-1 ) } );
            }
        }, ctx );

        if ( !parseError.empty() )
            return;

        size_t trisSize = 0;
        for ( auto& tris : trisPerThread )
            trisSize += tris.size();
        t.reserve( t.size() + trisSize );
        for ( auto& tris : trisPerThread )
            t.vec_.insert( t.vec_.end(), tris.vec_.begin(), tris.vec_.end() );
    };

    auto parseObject = [&] ( size_t, size_t end, std::string& )
    {
        if ( combineAllObjects )
            return;

        // finish previous object
        finishObject();

        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        currentObjName = line.substr( 1, std::string_view::npos );
        boost::trim( currentObjName );
    };

    timer.restart( "parse groups" );
    for ( const auto& group : groups )
    {
        std::string parseError;
        switch ( group.element )
        {
        case Unknown:
            break;
        case Vertex:
            parseVertices( group.begin, group.end, parseError );
            break;
        case Face:
            parseFaces( group.begin, group.end, parseError );
            break;
        case Object:
            parseObject( group.begin, group.end, parseError );
            break;
        }
        if ( !parseError.empty() )
            return tl::make_unexpected( parseError );

        if ( !callback( 0.50f + 0.50f * ( (float)group.end / (float)lineCount ) ) )
            return tl::make_unexpected( "Loading canceled" );
    }

    finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
