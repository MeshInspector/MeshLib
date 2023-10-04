#ifdef _WIN32
#include "MRMeshLoad.h"

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRTimer.h"

#include "MRPch/MRSpdlog.h"

#pragma warning( push )
#pragma warning( disable: 5054 )
#pragma warning( disable: 5220 )
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( disable: 5267 ) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Message.hxx>
#include <opencascade/Message_Printer.hxx>
#include <opencascade/Message_PrinterOStream.hxx>
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#pragma warning( pop )

namespace
{

class SpdlogPrinter final : public Message_Printer
{
private:
    void send( const TCollection_AsciiString& string, const Message_Gravity gravity ) const override
    {
        auto level = spdlog::level::trace;
        switch ( gravity )
        {
            case Message_Trace:
                level = spdlog::level::trace;
                break;
            case Message_Info:
                level = spdlog::level::info;
                break;
            case Message_Warning:
                level = spdlog::level::warn;
                break;
            case Message_Alarm:
                level = spdlog::level::err;
                break;
            case Message_Fail:
                level = spdlog::level::critical;
                break;
        }

        spdlog::log( level, "OpenCASCADE: {}", string.ToCString() );
    }
};

class MessageHandler
{
public:
    MessageHandler()
        : printer_( new SpdlogPrinter )
    {
        auto& messenger = Message::DefaultMessenger();
        // remove default stdout output
        messenger->RemovePrinters( STANDARD_TYPE( Message_PrinterOStream ) );
        // add spdlog output
        messenger->AddPrinter( Handle( Message_Printer)( printer_ ) );
    }

private:
    SpdlogPrinter* printer_;
};

}

namespace MR::MeshLoad
{

Expected<Mesh, std::string> fromStep( const std::filesystem::path& path, VertColors* colors, ProgressCallback callback )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    return addFileNameInError( fromStep( in, colors, callback ), path );
}

Expected<Mesh, std::string> fromStep( std::istream& in, VertColors*, ProgressCallback callback )
{
    MR_TIMER

    static MessageHandler handler;

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

    return Mesh::fromPointTriples( t, true );
}

MR_ADD_MESH_LOADER( IOFilter( "STEP files (*.step)", "*.step" ), fromStep )

} // namespace MR::MeshLoad
#endif