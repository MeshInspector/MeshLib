#include "MRPly.h"
#include "miniply.h"
#include "MRVector3.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRImageLoad.h"
#include "MRMeshTexture.h"
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
                Timer t( "extractVertColors" );
                std::vector<unsigned char> colorsBuffer( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
                params.colors->resize( numVerts );
                for ( VertId v{ 0 }; v < res.size(); ++v )
                {
                    int ind = 3 * v;
                    ( *params.colors )[v] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
                }
            }
            if ( params.uvCoords && reader.find_texcoord( indecies ) )
            {
                Timer t( "extractUVs" );
                params.uvCoords->resize( numVerts );
                reader.extract_properties( indecies, 2, miniply::PLYPropertyType::Float, params.uvCoords->data() );
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
                reader.extract_triangles( indecies[0], &res.front().x, (std::uint32_t)res.size(), miniply::PLYPropertyType::Int, tris.data() );
            }
            else
            {
                Timer t( "extractTriples" );
                auto numIndices = reader.num_rows();
                tris.resize( numIndices );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, tris.data() );
            }

            if ( params.faceColors && reader.find_color( indecies ) )
            {
                Timer t( "extractFaceColors" );
                std::vector<unsigned char> colorsBuffer( 3 * tris.size() );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
                params.faceColors->resize( tris.size() );
                for ( FaceId f{ 0 }; f < tris.size(); ++f )
                {
                    int ind = 3 * f;
                    ( *params.faceColors )[f] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
                }
            }
            if ( !res.empty() && !tris.empty() && params.uvCoords && reader.find_properties( indecies, 1, "texcoord" ) )
            {
                Timer t( "extractFaceUVs" );
                // the number of float-values in the property
                const auto propSize = reader.sum_of_list_counts( indecies[0] );
                // round upward to allocate not smaller amount of space
                const auto uvSize = ( propSize + 1 ) / 2;
                std::vector<UVCoord> cornerUVs( uvSize );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Float, cornerUVs.data() );

                // convert per-corner UVs into per-vertex UVs by keeping the last value only
                params.uvCoords->resize( res.size() );
                const auto endTri = std::min( tris.size(), cornerUVs.size() / 3 );
                for ( FaceId tri( 0 ); tri < endTri; ++tri )
                {
                    for ( int ic = 0; ic < 3; ++ic )
                    {
                        auto v = tris[tri][ic];
                        if ( v < params.uvCoords->size() )
                            (*params.uvCoords)[v] = cornerUVs[ size_t( tri ) * 3 + ic ];
                    }
                }
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

    if ( params.texture )
    {
        for ( const auto& comment : reader.comments() )
        {
            if ( !comment.starts_with( "TextureFile" ) )
                continue;
            int n = 11;
            while ( n < comment.size() && ( comment[n] == ' ' || comment[n] == '\t' ) )
                ++n;
            const auto texFile = params.dir / asU8String( comment.substr( n ) );
            std::error_code ec;
            if ( !is_regular_file( texFile, ec ) )
                break;
            if ( auto image = ImageLoad::fromAnySupportedFormat( texFile ) )
            {
                params.texture->resolution = std::move( image->resolution );
                params.texture->pixels = std::move( image->pixels );
                params.texture->filter = FilterType::Linear;
                params.texture->wrap = WrapType::Clamp;
            }
            break;
        }
    }

    return res;
}

} //namespace MR
