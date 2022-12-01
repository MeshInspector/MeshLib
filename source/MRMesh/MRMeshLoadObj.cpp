#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRPch/MRTBB.h"

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi.hpp>

namespace
{
    using namespace MR;

    tl::expected<Vector3f, std::string> parse_obj_vertex( const std::string_view& str )
    {
        namespace qi = boost::spirit::qi;
        namespace ascii = boost::spirit::ascii;

        using boost::phoenix::ref;

        Vector3f v;
        bool r = qi::phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> qi::float_[ref( v.x ) = qi::_1] >> qi::float_[ref( v.y ) = qi::_1] >> qi::float_[ref( v.z ) = qi::_1] ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse vertex" );
        return v;
    }

    struct index_accumulator
    {
        std::vector<int> indices;

        index_accumulator& operator +=( int v )
        {
            indices.emplace_back( v );
            return *this;
        }
    };

    struct obj_face_indices
    {
        index_accumulator vertices;
        index_accumulator textures;
        index_accumulator normals;
    };

    tl::expected<obj_face_indices, std::string> parse_obj_face( const std::string_view& str )
    {
        namespace qi = boost::spirit::qi;
        namespace ascii = boost::spirit::ascii;

        using boost::phoenix::ref;

        obj_face_indices vs;
        for ( auto ia : { &vs.vertices, &vs.textures, &vs.normals } )
            ia->indices.reserve( 4 );
        bool r = qi::phrase_parse(
            str.begin(),
            str.end(),
            (
                'f' >>
                *( qi::int_[ref( vs.vertices ) += qi::_1]
                    >> -(
                        ( '/' >> qi::int_[ref( vs.textures ) += qi::_1] )
                        |
                        ( '/' >> qi::int_[ref( vs.textures ) += qi::_1] >> '/' >> qi::int_[ref( vs.normals ) += qi::_1] )
                        |
                        ( "//" >> qi::int_[ref( vs.normals ) += qi::_1] )
                    )
                )
            ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse face" );
        // TODO: checks
        return vs;
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

    timer.restart( "split by lines" );
    std::vector<size_t> newlines{ 0 };
    for ( size_t ci = 0; ci < data.size(); ci++ )
        if ( data[ci] == '\n' )
            newlines.emplace_back( ci + 1 );
    // add finish line
    if ( newlines.back() != data.size() )
        newlines.emplace_back( data.size() );

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
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, vertexLines.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( auto vi = range.begin(); vi < range.end(); vi++ )
        {
            const auto li = vertexLines[vi];
            std::string_view line( data.data() + newlines[li], newlines[li + 1] - 1 - newlines[li + 0] );
            auto v = parse_obj_vertex( line );
            //if ( !v.has_value() )
            //    return tl::make_unexpected( v.error() );
            points[vi] = *v;
        }
    } );

    timer.restart( "parse face lines" );
    for ( auto& object : objects )
    {
        tbb::enumerable_thread_specific<Triangulation> trisPerThread;
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, object.faceLines.size() ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& tris = trisPerThread.local();
            for ( auto fi = range.begin(); fi < range.end(); fi++ )
            {
                const auto li = object.faceLines[fi];
                std::string_view line( data.data() + newlines[li], newlines[li + 1] - 1 - newlines[li + 0] );
                auto f = parse_obj_face( line );
                //if ( !f.has_value() )
                //    return tl::make_unexpected( f.error() );

                auto& vs = f->vertices.indices;
                for ( auto& v : vs )
                {
                    if ( v < 0 )
                    {
                        v += (int)points.size() + 1;
                        //if ( v <= 0 )
                        //    return tl::make_unexpected( std::string( "Too negative vertex ID in OBJ-file" ) );
                    }
                }
                //if ( vs.size() < 3 )
                //    return tl::make_unexpected( std::string( "Face with less than 3 vertices in OBJ-file" ) );

                // TODO: make smarter triangulation based on point coordinates
                for ( int j = 1; j + 1 < vs.size(); ++j )
                    tris.push_back( { VertId( vs[0]-1 ), VertId( vs[j]-1 ), VertId( vs[j+1]-1 ) } );
            }
        } );

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
    }

    if ( combineAllObjects )
        finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
