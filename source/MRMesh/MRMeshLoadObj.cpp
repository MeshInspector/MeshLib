#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRPch/MRTBB.h"

#include <boost/spirit/home/x3.hpp>

namespace
{
    using namespace MR;

    tl::expected<Vector3f, std::string> parseObjVertex( const std::string_view& str )
    {
        using namespace boost::spirit::x3;

        Vector3f v;
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

        return v;
    }

    struct ObjFace
    {
        std::vector<int> vertices;
        std::vector<int> textures;
        std::vector<int> normals;
    };

    tl::expected<ObjFace, std::string> parseObjFace( const std::string_view& str )
    {
        using namespace boost::spirit::x3;

        ObjFace f;
        for ( auto ia : { &f.vertices, &f.textures, &f.normals } )
            ia->reserve( 4 );
        auto v = [&] ( auto& ctx ) { f.vertices.emplace_back( _attr( ctx ) ); };
        auto vt = [&] ( auto& ctx ) { f.textures.emplace_back( _attr( ctx ) ); };
        auto vn = [&] ( auto& ctx ) { f.normals.emplace_back( _attr( ctx ) ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            ( 'f' >> *( int_[v] >> -( ( '/' >> int_[vt] ) | ( '/' >> int_[vt] >> '/' >> int_[vn] ) | ( "//" >> int_[vn] ) ) ) ),
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
        return f;
    }
}

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
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

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Timer timer( "read data" );
    Buffer<char> data( streamSize );
    in.read( data.data(), (ptrdiff_t)data.size() );
    if ( !in )
        return tl::make_unexpected( std::string( "OBJ-format read error" ) );

    if ( !callback( 0.25f ) )
        return tl::make_unexpected( "Loading cancelled" );

    timer.restart( "split by lines" );
    std::vector<size_t> newlines{ 0 };
    tbb::enumerable_thread_specific<std::vector<size_t>> newlinesPerThread;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, data.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        bool exists = false;
        auto& newlines = newlinesPerThread.local( exists );
        // blocks shall not intersect each other
        assert( !exists );
        for ( auto ci = range.begin(); ci < range.end(); ci++ )
        {
            if ( data[ci] == '\n' )
                newlines.emplace_back( ci + 1 );
        }
    }, tbb::static_partitioner() );
    std::vector<std::vector<size_t>> newlinesBlocks;
    size_t newlinesSize = 1;
    for ( auto&& newlinesBlock : newlinesPerThread )
    {
        assert( !newlinesBlock.empty() );
        newlinesSize += newlinesBlock.size();
        newlinesBlocks.emplace_back( std::move( newlinesBlock ) );
    }
    std::sort( newlinesBlocks.begin(), newlinesBlocks.end(), [] ( const auto& lhs, const auto& rhs )
    {
        return lhs.front() < rhs.front();
    } );
    newlines.reserve( newlinesSize );
    for ( auto&& newlinesBlock : newlinesBlocks )
        newlines.insert( newlines.end(), newlinesBlock.begin(), newlinesBlock.end() );
    // add finish line
    if ( newlines.back() != data.size() )
        newlines.emplace_back( data.size() );

    if ( !callback( 0.40f ) )
        return tl::make_unexpected( "Loading cancelled" );

    timer.restart( "find element lines" );
    struct Object
    {
        std::string name;
        std::vector<size_t> faceLines;
    };
    std::vector<size_t> vertexLines;
    std::vector<Object> objects( 1 ); // emplace default object
    for ( size_t li = 0; li + 1 < newlines.size(); ++li )
    {
        auto* line = data.data() + newlines[li];
        if ( line[0] == 'v' && line[1] != 'n' /*normals*/ && line[1] != 't' /*texture coordinates*/ )
        {
            vertexLines.emplace_back( li );
        }
        else if ( line[0] == 'f' )
        {
            objects.back().faceLines.emplace_back( li );
        }
        else if ( line[0] == 'o' )
        {
            auto object = objects.emplace_back();

            const auto lineWidth = newlines[li + 1] - 1 - newlines[li + 0];
            size_t trimShift = 1;
            while ( line[trimShift] == ' ' )
                trimShift++;
            object.name = std::string_view( line + trimShift, lineWidth - trimShift );
        }
    }

    timer.restart( "parse vertex lines" );
    points.resize( vertexLines.size() );
    std::string parseError;
    tbb::task_group_context ctx;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, vertexLines.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( auto vi = range.begin(); vi < range.end(); vi++ )
        {
            const auto li = vertexLines[vi];
            std::string_view line( data.data() + newlines[li], newlines[li + 1] - 1 - newlines[li + 0] );
            auto v = parseObjVertex( line );
            if ( !v.has_value() )
            {
                if ( ctx.cancel_group_execution() )
                    parseError = v.error();
                return;
            }
            points[vi] = *v;
        }
    }, ctx );
    if ( !parseError.empty() )
        return tl::make_unexpected( parseError );

    if ( !callback( 0.50f ) )
        return tl::make_unexpected( "Loading cancelled" );

    timer.restart( "parse face lines" );
    size_t faceTotal = 0;
    for ( auto& object : objects )
        faceTotal += object.faceLines.size();
    size_t faceProcessed = 0;
    for ( auto& object : objects )
    {
        tbb::enumerable_thread_specific<Triangulation> trisPerThread;
        parseError.clear();
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, object.faceLines.size() ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& tris = trisPerThread.local();
            for ( auto fi = range.begin(); fi < range.end(); fi++ )
            {
                const auto li = object.faceLines[fi];
                std::string_view line( data.data() + newlines[li], newlines[li + 1] - 1 - newlines[li + 0] );
                auto f = parseObjFace( line );
                if ( !f.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = f.error();
                    return;
                }

                auto& vs = f->vertices;
                for ( auto& v : vs )
                {
                    if ( v < 0 )
                    {
                        v += (int)points.size() + 1;
                        if ( v <= 0 )
                        {
                            if ( ctx.cancel_group_execution() )
                                parseError = "Too negative vertex ID in OBJ-file";
                            return;
                        }
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
            return tl::make_unexpected( parseError );

        size_t size = 0;
        for ( auto& tris : trisPerThread )
            size += tris.size();
        t.reserve( t.size() + size );
        for ( auto& tris : trisPerThread )
            t.vec_.insert( t.vec_.end(), tris.vec_.begin(), tris.vec_.end() );

        if ( !combineAllObjects )
        {
            currentObjName = object.name;
            finishObject();
        }

        faceProcessed += object.faceLines.size();
        if ( !callback( 0.50f + 0.50f * ( (float)faceProcessed / (float)faceTotal ) ) )
            return tl::make_unexpected( "Loading cancelled" );
    }

    if ( combineAllObjects )
        finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
