#include "MRMeshLoadStep.h"
#ifndef MRMESH_NO_OPENCASCADE
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRObjectMesh.h"
#include "MRObjectsAccess.h"
#include "MRParallelFor.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRMeshLoadSettings.h"

#include "MRPch/MRSpdlog.h"
#include "MRPch/MRSuppressWarning.h"

#if !defined( __GNUC__ ) || defined( __clang__ ) || __GNUC__ >= 11
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
#endif
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-enum-enum-conversion", 5054 )
#pragma warning( push )
#pragma warning( disable: 5220 ) // a non-static data member with a volatile qualified type no longer implies that compiler generated copy/move constructors and copy/move assignment operators are not trivial
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( disable: 5267 ) // definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <Message.hxx>
#include <Message_Printer.hxx>
#include <Message_PrinterOStream.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Version.hxx>
#include <StepData_Protocol.hxx>
#include <StepData_StepModel.hxx>
#include <StepData_StepWriter.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#pragma warning( pop )
MR_SUPPRESS_WARNING_POP
#if !defined( __GNUC__ ) || defined( __clang__ ) || __GNUC__ >= 11
MR_SUPPRESS_WARNING_POP
#endif

namespace
{

using namespace MR;

// updated BRepMesh API support
// https://dev.opencascade.org/doc/overview/html/occt__upgrade.html#upgrade_740_changed_api_of_brepmesh
#define MODERN_BREPMESH_SUPPORTED ( ( ( OCC_VERSION_MAJOR > 7 ) || ( OCC_VERSION_MAJOR == 7 && OCC_VERSION_MINOR >= 4 ) ) )
// updated Message interface support
// https://dev.opencascade.org/doc/overview/html/occt__upgrade.html#upgrade_750_message_messenger
// https://dev.opencascade.org/doc/overview/html/occt__upgrade.html#upgrade_750_message_printer
#define MODERN_MESSAGE_SUPPORTED ( ( OCC_VERSION_MAJOR > 7 ) || ( OCC_VERSION_MAJOR == 7 && OCC_VERSION_MINOR >= 5 ) )
// reading STEP data from streams support
// https://www.opencascade.com/open-cascade-technology-7-5-0-released/
#define STEP_READSTREAM_SUPPORTED ( ( ( OCC_VERSION_MAJOR > 7 ) || ( OCC_VERSION_MAJOR == 7 && OCC_VERSION_MINOR >= 5 ) ) )
// reading STEP data fixed
#define STEP_READER_FIXED ( ( ( OCC_VERSION_MAJOR > 7 ) || ( OCC_VERSION_MAJOR == 7 && OCC_VERSION_MINOR >= 7 ) ) )

#if MODERN_MESSAGE_SUPPORTED
/// spdlog adaptor for OpenCASCADE logging system
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

/// replace default OpenCASCADE log output with redirect to spdlog
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
        // NOTE: OpenCASCADE takes the ownership of the printer, don't delete it manually
        messenger->AddPrinter( Handle( Message_Printer )( printer_ ) );
    }

private:
    SpdlogPrinter* printer_;
};

MessageHandler messageHandler;
#else
#pragma message( "Log redirecting is currently unsupported for OpenCASCADE versions prior to 7.4" )
#endif

struct FeatureData
{
    Handle( Poly_Triangulation ) triangulation;
    TopLoc_Location location;
    TopAbs_Orientation orientation;

    FeatureData( Handle( Poly_Triangulation )&& triangulation, TopLoc_Location&& location, TopAbs_Orientation orientation )
        : triangulation( std::move( triangulation ) )
        , location( std::move( location ) )
        , orientation( orientation )
    {
        //
    }
};

std::vector<Triangle3f> loadSolid( const TopoDS_Shape& solid )
{
    MR_TIMER

    // TODO: expose parameters
#if MODERN_BREPMESH_SUPPORTED
    IMeshTools_Parameters parameters;
#else
    BRepMesh_FastDiscret::Parameters parameters;
#endif
    parameters.Angle = 0.1;
    parameters.Deflection = 0.5;
    parameters.Relative = false;
    parameters.InParallel = true;

    BRepMesh_IncrementalMesh incMesh( solid, parameters );
    const auto& meshShape = incMesh.Shape();

    std::deque<FeatureData> features;
    size_t totalVertexCount = 0, totalFaceCount = 0;
    for ( auto faceExp = TopExp_Explorer( meshShape, TopAbs_FACE ); faceExp.More(); faceExp.Next() )
    {
        const auto& face = faceExp.Current();

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

    std::vector<Triangle3f> triples;
    triples.reserve( totalFaceCount );

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
        const auto faceOffset = triples.size();

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

            triples.emplace_back( Triangle3f {
                points[vs[0]],
                points[vs[1]],
                points[vs[2]],
            } );
        }

        // parts
        // TODO: return mesh features
        FaceBitSet region;
        region.resize( triples.size(), false );
        region.set( FaceId( faceOffset ), faceCount, true );
        parts.emplace_back( std::move( region ) );
    }

    return triples;
}

#if STEP_READER_FIXED
// read STEP file without any work-arounds
VoidOrErrStr readStepData( STEPControl_Reader& reader, std::istream& in, const ProgressCallback& )
{
    auto ret = reader.ReadStream( "STEP file", in );
    if ( ret != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );
    return {};
}
#elif STEP_READSTREAM_SUPPORTED
// read STEP file with the 'save-load' work-around
// some STEP models are loaded broken in OpenCASCADE prior to 7.7
VoidOrErrStr readStepData( STEPControl_Reader& reader, std::istream& in, const ProgressCallback& cb )
{
    STEPControl_Reader auxReader;
    auto ret = auxReader.ReadStream( "STEP file", in );
    if ( ret != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );

    if ( !reportProgress( cb, 0.50f ) )
        return unexpected( std::string( "Loading canceled" ) );

    const auto model = auxReader.StepModel();
    const auto protocol = Handle( StepData_Protocol )::DownCast( model->Protocol() );

    StepData_StepWriter sw( model );
    sw.SendModel( protocol );

    std::stringstream buffer;
    if ( !sw.Print( buffer ) )
        return unexpected( "Failed to repair STEP model" );

    if ( !reportProgress( cb, 0.65f ) )
        return unexpected( std::string( "Loading canceled" ) );

    buffer.seekp( 0, std::ios::beg );
    ret = reader.ReadStream( "STEP file", buffer );
    if ( ret != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );

    return {};
}
#else
// read STEP file with the 'save-load' work-around
// loading from a file stream is not supported in OpenCASCADE prior to 7.5
std::filesystem::path getStepTemporaryDirectory()
{
    const auto path = std::filesystem::temp_directory_path() / "MeshLib_MeshLoadStep";
    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        std::filesystem::create_directory( path, ec );
    return path;
}

VoidOrErrStr readStepData( STEPControl_Reader& reader, const std::filesystem::path& path, const ProgressCallback& cb )
{
    STEPControl_Reader auxReader;
    auto ret = auxReader.ReadFile( path.c_str() );
    if ( ret != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );

    if ( !reportProgress( cb, 0.50f ) )
        return unexpected( std::string( "Loading canceled" ) );

    const auto model = auxReader.StepModel();
    const auto protocol = Handle( StepData_Protocol )::DownCast( model->Protocol() );

    StepData_StepWriter sw( model );
    sw.SendModel( protocol );

    const auto auxFilePath = getStepTemporaryDirectory() / "auxFile.step";
    {
        // NOTE: opening the file in binary mode and passing to the StepData_StepWriter object lead to segfault
        std::ofstream ofs( auxFilePath );
        if ( !sw.Print( ofs ) )
            return unexpected( "Failed to repair STEP model" );
    }

    if ( !reportProgress( cb, 0.65f ) )
        return unexpected( std::string( "Loading canceled" ) );

    ret = reader.ReadFile( auxFilePath.c_str() );
    if ( ret != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );

    std::filesystem::remove( auxFilePath );

    return {};
}
#endif

Expected<std::shared_ptr<Object>, std::string> stepModelToScene( STEPControl_Reader& reader, const MeshLoadSettings&, const ProgressCallback& cb )
{
    MR_TIMER

    const auto cb1 = subprogress( cb, 0.30f, 0.74f );
    const auto rootCount = reader.NbRootsForTransfer();
    for ( auto i = 1; i <= rootCount; ++i )
    {
        reader.TransferRoot( i );
        if ( !reportProgress( cb1, ( float )i / ( float )rootCount ) )
            return unexpected( std::string( "Loading canceled" ) );
    }
    if ( !reportProgress( cb, 0.9f ) )
        return unexpected( std::string( "Loading canceled" ) );

    std::deque<TopoDS_Shape> shapes;
    for ( auto i = 1; i <= reader.NbShapes(); ++i )
        shapes.emplace_back( reader.Shape( i ) );

    // TODO: preserve shape-solid hierarchy? (not sure about actual shape count in real models)
    std::deque<TopoDS_Shape> solids;
    for ( const auto& shape : shapes )
    {
        size_t solidCount = 0;
        for ( auto explorer = TopExp_Explorer( shape, TopAbs_SOLID ); explorer.More(); explorer.Next() )
        {
            solids.emplace_back( explorer.Current() );
            ++solidCount;
        }
        // import the whole shape if it doesn't consist of solids
        if ( solidCount == 0 )
            solids.emplace_back( shape );
    }
    shapes.clear();

    if ( solids.empty() )
    {
        auto rootObject = std::make_shared<Object>();
        rootObject->select( true );
        return rootObject;
    }
    else if ( solids.size() == 1 )
    {
        const auto triples = loadSolid( solids.front() );
        auto mesh = Mesh::fromPointTriples( triples, true );

        auto rootObject = std::make_shared<ObjectMesh>();
        rootObject->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
        rootObject->select( true );
        return rootObject;
    }
    else
    {
        auto cb2 = subprogress( cb, 0.90f, 1.0f );
        tbb::task_arena taskArena;
        std::vector<std::shared_ptr<Object>> children( solids.size() );
        taskArena.execute( [&]
        {
            tbb::task_group taskGroup;
            std::vector<std::vector<Triangle3f>> triples( solids.size() );
            for ( auto i = 0; i < solids.size(); ++i )
            {
                triples[i] = loadSolid( solids[i] );
                taskGroup.run( [i, &triples, &children]
                {
                    auto mesh = Mesh::fromPointTriples( triples[i], true );

                    auto child = std::make_shared<ObjectMesh>();
                    child->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
                    child->setName( fmt::format( "Solid{}", i + 1 ) );
                    child->select( true );
                    children[i] = std::move( child );
                } );
            }
            taskGroup.wait();
        } );

        auto rootObject = std::make_shared<Object>();
        rootObject->select( true );
        for ( auto& child : children )
            rootObject->addChild( std::move( child ), true );

        auto sceneObject = std::make_shared<Object>();
        sceneObject->addChild( rootObject );
        return sceneObject;
    }
}

std::mutex cOpenCascadeMutex = {};
#if !STEP_READSTREAM_SUPPORTED
std::mutex cOpenCascadeTempFileMutex = {};
#endif

}

#if STEP_READSTREAM_SUPPORTED
namespace MR::MeshLoad
{

Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    auto result = fromSceneStepFile( in, settings );
    if ( !result )
        return addFileNameInError( result, path );

    if ( auto& obj = *result )
    {
        const auto defaultName = utf8string( path.stem() );
        if ( obj->name().empty() )
            obj->setName( defaultName );
        for ( auto& child : getAllObjectsInTree( *obj ) )
            if ( child->name().empty() )
                child->setName( defaultName );
    }

    return result;
}

Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings )
{
    MR_TIMER

    // NOTE: OpenCASCADE STEP reader is NOT thread-safe
    std::unique_lock lock( cOpenCascadeMutex );

    const auto cb1 = subprogress( settings.callback, 0.00f, 0.90f );
    STEPControl_Reader reader;
    const auto ret = readStepData( reader, in, cb1 );
    if ( !ret.has_value() )
        return unexpected( ret.error() );

    const auto cb2 = subprogress( settings.callback, 0.90f, 1.00f );
    return stepModelToScene( reader, settings, cb2 );
}

} // namespace MR::MeshLoad
#else
namespace MR::MeshLoad
{

Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings )
{
    std::unique_lock lock( cOpenCascadeTempFileMutex );
    const auto tempFileName = getStepTemporaryDirectory() / "tempFile.step";
    {
        std::ofstream ofs( tempFileName, std::ios::binary );
        if ( !ofs )
            return unexpected( std::string( "Cannot open buffer file" ) );
        ofs << in.rdbuf();
    }
    return fromSceneStepFile( tempFileName, settings );
}

Expected<std::shared_ptr<Object>, std::string> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings )
{
    MR_TIMER

    // NOTE: OpenCASCADE STEP reader is NOT thread-safe
    std::unique_lock lock( cOpenCascadeMutex );

    const auto cb1 = subprogress( settings.callback, 0.00f, 0.90f );
    STEPControl_Reader reader;
    const auto ret = readStepData( reader, path, cb1 );
    if ( !ret.has_value() )
        return unexpected( ret.error() );

    const auto cb2 = subprogress( settings.callback, 0.90f, 1.00f );
    return stepModelToScene( reader, settings, cb2 );
}

} // namespace MR::MeshLoad
#endif
#endif
