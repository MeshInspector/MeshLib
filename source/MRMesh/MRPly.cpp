#include "MRPly.h"
#include "miniply.h"
#include "MRVector3.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRImageLoad.h"
#include "MRMeshTexture.h"
#include "MRTelemetry.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

/// meaningless or case-specific comments to skip
bool ignoreComment( const std::string & comment )
{
    return comment == "File generated"
        || ( comment.starts_with( "Created 20" ) && comment.size() > 12 && comment[12] == '-' ) // e.g. "Created 2025-12-19T22:09:05"
        || comment.starts_with( "scalex " )
        || comment.starts_with( "scaley " )
        || comment.starts_with( "scalez " )
        || comment.starts_with( "shiftx " )
        || comment.starts_with( "shifty " )
        || comment.starts_with( "shiftz " )
        || comment.starts_with( "minx " )
        || comment.starts_with( "miny " )
        || comment.starts_with( "minz " )
        || comment.starts_with( "maxx " )
        || comment.starts_with( "maxy " )
        || comment.starts_with( "maxz " )
        || comment.starts_with( "offsetx " )
        || comment.starts_with( "offsety " )
        || comment.starts_with( "offsetz " )
        ;
}

} // anonymous namespace

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

    std::string signalString = "PLY";
    for ( int i = 0; reader.has_element(); reader.next_element(), ++i )
    {
        if ( reader.element_is(miniply::kPLYVertexElement) && reader.load_element() )
        {
            signalString += " V";
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                Timer t( "extractPoints" );
                signalString += 'P';
                res.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.data() );
                gotVerts = true;
            }
            if ( reader.find_normal( indecies ) )
            {
                signalString += 'N';
                if ( params.normals )
                {
                    Timer t( "extractNormals" );
                    params.normals->resize( numVerts );
                    reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, params.normals->data() );
                }
            }
            if ( reader.find_color( indecies ) )
            {
                signalString += 'C';
                if ( params.colors )
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
            }
            if ( reader.find_texcoord( indecies ) )
            {
                signalString += "UV";
                if ( params.uvCoords )
                {
                    Timer t( "extractUVs" );
                    params.uvCoords->resize( numVerts );
                    reader.extract_properties( indecies, 2, miniply::PLYPropertyType::Float, params.uvCoords->data() );
                }
            }

            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !reportProgress( params.callback, progress ) )
                return unexpectedOperationCanceled();
            continue;
        }

        const auto posLast = in.tellg();
        if ( reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indecies) )
        {
            const size_t numRows( reader.num_rows() );
            const bool polys = reader.requires_triangulation( indecies[0] );
            signalString += polys ? " POLY" : " TRI";
            if ( params.tris )
            {
                Triangulation tris;
                if (polys)
                {
                    Timer t( "extractTriangles" );
                    if ( !gotVerts )
                    {
                        signalString += " ENOVC";
                        if ( params.telemetrySignal )
                            TelemetrySignal( signalString );
                        return unexpected( std::string( "PLY file open: need vertex positions to triangulate faces" ) );
                    }
                    auto numIndices = reader.num_triangles( indecies[0] );
                    tris.resize( numIndices );
                    reader.extract_triangles( indecies[0], &res.front().x, (std::uint32_t)res.size(), miniply::PLYPropertyType::Int, tris.data() );
                }
                else
                {
                    Timer t( "extractTriples" );
                    tris.resize( numRows );
                    reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, tris.data() );
                }
                *params.tris = std::move( tris );
            }

            if ( reader.find_color( indecies ) )
            {
                signalString += 'C';
                if ( params.faceColors )
                {
                    Timer t( "extractFaceColors" );
                    std::vector<unsigned char> colorsBuffer( 3 * numRows );
                    reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
                    params.faceColors->resize( numRows );
                    for ( FaceId f{ 0 }; f < numRows; ++f )
                    {
                        int ind = 3 * f;
                        ( *params.faceColors )[f] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
                    }
                }
            }
            if ( reader.find_properties( indecies, 1, "texcoord" ) )
            {
                signalString += "UV";
                if ( params.triCornerUvCoords )
                {
                    Timer t( "extractFaceUVs" );
                    // the number of float-values in the property
                    const auto propSize = reader.sum_of_list_counts( indecies[0] );
                    // round upward to allocate not smaller amount of space
                    static_assert( sizeof( params.triCornerUvCoords->front() ) == 6 * sizeof( float ) );
                    params.triCornerUvCoords->resize( ( propSize + 5 ) / 6 );
                    reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Float, params.triCornerUvCoords->data() );
                }
            }

            const auto posCurrent = in.tellg();
            // suppose  that reading is 10% of progress and building mesh is 90% of progress
            if ( !reportProgress( params.callback, ( float( posLast ) + ( posCurrent - posLast ) * 0.1f - posStart ) / streamSize ) )
                return unexpectedOperationCanceled();
        }

        if ( reader.element_is( "edge" ) && reader.load_element() && reader.find_properties( indecies, 2, "vertex1", "vertex2" ) )
        {
            signalString += " EDGE";
            if ( params.edges )
            {
                auto numEdges = reader.num_rows();
                Edges es( numEdges );
                static_assert( sizeof( es.front() ) == 8 );
                if ( reader.extract_properties( indecies, 2, miniply::PLYPropertyType::Int, es.data() ) )
                    *params.edges = std::move( es );
            }
        }
    }

    if ( !reader.valid() )
    {
        signalString += " EREAD";
        if ( params.telemetrySignal )
            TelemetrySignal( signalString );
        return unexpected( std::string( "PLY file read or parse error" ) );
    }

    if ( !gotVerts )
    {
        signalString += " ENOVC";
        if ( params.telemetrySignal )
            TelemetrySignal( signalString );
        return unexpected( std::string( "PLY file does not contain vertices" ) );
    }

    auto textureToLoad = params.texture;
    for ( const auto& comment : reader.comments() )
    {
        if ( !comment.starts_with( "TextureFile" ) )
        {
            if ( params.telemetrySignal && !ignoreComment( comment ) )
                TelemetrySignal( "PLY comment " + comment );
            continue;
        }
        signalString += " TEX";
        int n = 11;
        while ( n < comment.size() && ( comment[n] == ' ' || comment[n] == '\t' ) )
            ++n;
        const auto texFile = params.dir / asU8String( comment.substr( n ) );
        std::error_code ec;
        if ( !is_regular_file( texFile, ec ) )
        {
            signalString += '-';
            continue;
        }
        if ( textureToLoad )
        {
            if ( auto image = ImageLoad::fromAnySupportedFormat( texFile ) )
            {
                textureToLoad->resolution = std::move( image->resolution );
                textureToLoad->pixels = std::move( image->pixels );
                textureToLoad->filter = FilterType::Linear;
                textureToLoad->wrap = WrapType::Clamp;
            }
            textureToLoad = nullptr; // load the first texture only
        }
    }

    if ( params.telemetrySignal )
        TelemetrySignal( signalString );
    return res;
}

} //namespace MR
