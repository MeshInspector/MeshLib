#include "MRPly.h"
#include "miniply.h"
#include "MRVector3.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRTimer.h"

namespace MR
{

Expected<VertCoords> loadPly( std::istream& in, const PlyLoadParams& params )
{
    MR_TIMER;

    const auto posStart = in.tellg();
    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false;

    std::vector<unsigned char> colorsBuffer;
    VertCoords res;
    const auto posEnd = reader.get_end_pos();
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0; reader.has_element(); reader.next_element(), ++i )
    {
        if ( reader.element_is(miniply::kPLYVertexElement) && reader.load_element() )
        {
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                Timer t( "extractPoints" );
                res.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.data() );
                gotVerts = true;
            }
            if ( params.normals && reader.find_normal( indecies ) )
            {
                Timer t( "extractNormals" );
                params.normals->resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, params.normals->data() );
            }
            if ( params.colors && reader.find_color( indecies ) )
            {
                Timer t( "extractColors" );
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !reportProgress( params.callback, progress ) )
                return unexpectedOperationCanceled();
            continue;
        }

        const auto posLast = in.tellg();
        if ( params.tris && reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indecies) )
        {
            bool polys = reader.requires_triangulation( indecies[0] );
            if ( polys && !gotVerts )
                return unexpected( std::string( "PLY file open: need vertex positions to triangulate faces" ) );

            Triangulation tris;
            if (polys)
            {
                Timer t( "extractTriangles" );
                auto numIndices = reader.num_triangles( indecies[0] );
                tris.resize( numIndices );
                reader.extract_triangles( indecies[0], &res.front().x, (std::uint32_t)res.size(), miniply::PLYPropertyType::Int, &tris.front() );
            }
            else
            {
                Timer t( "extractTriples" );
                auto numIndices = reader.num_rows();
                tris.resize( numIndices );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, &tris.front() );
            }
            const auto posCurrent = in.tellg();
            // suppose  that reading is 10% of progress and building mesh is 90% of progress
            if ( !reportProgress( params.callback, ( float( posLast ) + ( posCurrent - posLast ) * 0.1f - posStart ) / streamSize ) )
                return unexpectedOperationCanceled();
            *params.tris = std::move( tris );
        }

        if ( params.edges && reader.element_is( "edge" ) && reader.load_element() && reader.find_properties( indecies, 2, "vertex1", "vertex2" ) )
        {
            auto numEdges = reader.num_rows();
            Edges es( numEdges );
            static_assert( sizeof( es.front() ) == 8 );
            if ( reader.extract_properties( indecies, 2, miniply::PLYPropertyType::Int, es.data() ) )
                *params.edges = std::move( es );
        }
    }

    if ( !reader.valid() )
        return unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
         return unexpected( std::string( "PLY file does not contain vertices" ) );

    if ( params.colors && !colorsBuffer.empty() )
    {
        params.colors->resize( res.size() );
        for ( VertId i{ 0 }; i < res.size(); ++i )
        {
            int ind = 3 * i;
            ( *params.colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }

    return res;
}

} //namespace MR
