#include "MRMeshLoadStep.h"

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"

#pragma warning( push )
#pragma warning( disable: 5054 )
#pragma warning( disable: 5220 )
#pragma warning( disable: 5267 )
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#pragma warning( pop )

namespace MR::MeshLoad
{

Expected<Mesh, std::string> fromStep( const std::filesystem::path& path, Vector<Color, VertId>* colors, ProgressCallback callback )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    return fromStep( in, colors, callback );
}

Expected<Mesh, std::string> fromStep( std::istream& in, Vector<Color, VertId>*, ProgressCallback callback )
{
    MR_TIMER

    // NOTE: OpenCASCADE STEP reader is NOT thread-safe
    static std::mutex mutex;
    std::unique_lock lock( mutex );

    STEPControl_Reader reader;
    {
        MR_NAMED_TIMER( "STEP reader: read stream" )
        reader.ReadStream( "STEP file", in );
    }
    {
        MR_NAMED_TIMER( "STEP reader: transfer roots" )
        reader.TransferRoots();
    }
    const auto shape = reader.OneShape();
    //for ( auto i = 1; i <= reader.NbShapes(); ++i )
    //    shape = reader.Shape( i );

    BRepMesh_IncrementalMesh incMesh( shape, 0.1 );
    const auto& meshShape = incMesh.Shape();

    struct FeatureData
    {
        Handle( Poly_Triangulation ) triangulation;
        TopLoc_Location location;
        TopAbs_Orientation orientation;

        FeatureData( Handle( Poly_Triangulation )&& triangulation, TopLoc_Location&& location, TopAbs_Orientation orientation )
            : triangulation( std::move( triangulation ) )
            , location( std::move( location ) )
            , orientation( orientation )
        {}
    };
    std::deque<FeatureData> features;
    size_t totalVertexCount = 0, totalFaceCount = 0;
    for ( auto explorer = TopExp_Explorer( meshShape, TopAbs_FACE ); explorer.More(); explorer.Next() )
    {
        const auto& face = explorer.Current();

        TopLoc_Location location;
        auto triangulation = BRep_Tool::Triangulation( TopoDS::Face( face ), location );
        if ( triangulation.IsNull() )
            continue;

        totalVertexCount += triangulation->NbNodes();
        totalFaceCount += triangulation->NbTriangles();
        features.emplace_back( std::move( triangulation ), std::move( location ), face.Orientation() );
    }

    std::vector<Vector3f> points;
    points.reserve( totalVertexCount );

    std::vector<Triangle3f> t;
    t.reserve( totalFaceCount );

    std::vector<FaceBitSet> parts;
    parts.reserve( features.size() );

    for ( const auto& feature : features )
    {
        // vertices
        const auto vertexCount = feature.triangulation->NbNodes();
        const auto vertexOffset = points.size();

        const auto& xf = feature.location.Transformation();
        for ( auto i = 1; i <= vertexCount; ++i )
        {
            auto point = feature.triangulation->Node( i );
            point.Transform( xf );
            points.emplace_back( point.X(), point.Y(), point.Z() );
        }

        // faces
        const auto faceCount = feature.triangulation->NbTriangles();
        const auto faceOffset = t.size();

        const auto reversed = ( feature.orientation == TopAbs_REVERSED );
        for ( auto i = 1; i <= faceCount; ++i )
        {
            const auto& tri = feature.triangulation->Triangle( i );

            std::array<int, 3> vs { -1, -1, -1 };
            tri.Get( vs[0], vs[1], vs[2] );
            if ( reversed )
                std::swap( vs[1], vs[2] );
            for ( auto& v : vs )
                v += (int)vertexOffset - 1;

            t.emplace_back( Triangle3f {
                points[vs[0]],
                points[vs[1]],
                points[vs[2]],
            } );
        }

        // parts
        FaceBitSet region;
        region.resize( t.size(), false );
        region.set( FaceId( faceOffset ), faceCount, true );
        parts.emplace_back( std::move( region ) );
    }

    return MeshBuilder::fromPointTriples( t );
}

MR_ADD_MESH_LOADER( IOFilter( "STEP files (*.step)", "*.step" ), fromStep )

} // namespace MR::MeshLoad