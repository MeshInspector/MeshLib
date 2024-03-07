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

MR_SUPPRESS_WARNING_PUSH
MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )
MR_SUPPRESS_WARNING( "-Wpedantic", 4996 )
#if !defined( __GNUC__ ) || defined( __clang__ ) || __GNUC__ >= 11
MR_SUPPRESS_WARNING( "-Wdeprecated-enum-enum-conversion", 5054 )
#endif
#ifdef _MSC_VER
#pragma warning( disable: 4266 ) // no override available for virtual member function from base '...'; function is hidden
#pragma warning( disable: 5220 ) // a non-static data member with a volatile qualified type no longer implies that compiler generated copy/move constructors and copy/move assignment operators are not trivial
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning( disable: 5267 ) // definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
// FIXME: include dir with vcpkg
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/Message.hxx>
#include <opencascade/Message_Printer.hxx>
#include <opencascade/Message_PrinterOStream.hxx>
#include <opencascade/Standard_Version.hxx>
#include <opencascade/STEPCAFControl_Reader.hxx>
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/StepData_Protocol.hxx>
#include <opencascade/StepData_StepModel.hxx>
#include <opencascade/StepData_StepWriter.hxx>
#include <opencascade/TDataStd_Name.hxx>
#include <opencascade/TDF_ChildIterator.hxx>
#include <opencascade/TDocStd_Document.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/XCAFDoc_DocumentTool.hxx>
#include <opencascade/XCAFDoc_ShapeTool.hxx>
#else
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Message.hxx>
#include <Message_Printer.hxx>
#include <Message_PrinterOStream.hxx>
#include <Standard_Version.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPControl_Reader.hxx>
#include <StepData_Protocol.hxx>
#include <StepData_StepModel.hxx>
#include <StepData_StepWriter.hxx>
#include <TDataStd_Name.hxx>
#include <TDF_ChildIterator.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#endif
MR_SUPPRESS_WARNING_POP

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

std::vector<Triangle3f> loadShape( const TopoDS_Shape& shape, bool resetXf = false )
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

    BRepMesh_IncrementalMesh incMesh( shape, parameters );
    auto meshShape = incMesh.Shape();
    if ( resetXf )
        meshShape.Location( TopLoc_Location() );

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
        const auto triples = loadShape( solids.front() );
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
                triples[i] = loadShape( solids[i] );
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

std::string readName( const TDF_Label& label )
{
    if ( label.IsNull() )
        return {};

    Handle( TDataStd_Name ) name;
    if ( label.FindAttribute( TDataStd_Name::GetID(), name ) != Standard_True )
        return {};

    const auto& str = name->Get();
    std::string result( str.LengthOfCString(), '\0' );
    auto* resultCStr = result.data();
    str.ToUTF8CString( resultCStr );

    return result;
}

Vector3d toVector( const gp_XYZ& xyz )
{
    return { xyz.X(), xyz.Y(), xyz.Z() };
}

AffineXf3d toXf( const gp_Trsf& transformation )
{
    AffineXf3d xf;
    const auto xfA = transformation.VectorialPart();
    xf.A = Matrix3d::fromColumns(
        toVector( xfA.Column( 1 ) ),
        toVector( xfA.Column( 2 ) ),
        toVector( xfA.Column( 3 ) )
    );
    xf.b = toVector( transformation.TranslationPart() );
    return xf;
}

std::mutex cOpenCascadeMutex = {};
#if !STEP_READSTREAM_SUPPORTED
std::mutex cOpenCascadeTempFileMutex = {};
#endif

} // namespace

namespace MR::MeshLoad
{

#if STEP_READSTREAM_SUPPORTED
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
#else
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
#endif

Expected<std::shared_ptr<Object>> fromSceneStepFile2( const std::filesystem::path& path, const MeshLoadSettings& settings )
{
    MR_TIMER

    std::unique_lock lock( cOpenCascadeMutex );

    STEPCAFControl_Reader reader;
    reader.SetColorMode( true );
    reader.SetNameMode( true );

    const TCollection_AsciiString pathStr( path.c_str() );
    if ( reader.ReadFile( pathStr.ToCString() ) != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );

    reportProgress( settings.callback, 0.25f );

    Handle( TDocStd_Document ) document = new TDocStd_Document( "MDTV-CAF" );
    if ( reader.Transfer( document ) != Standard_True )
        return unexpected( "Failed to read STEP model" );

    auto shapeTool = XCAFDoc_DocumentTool::ShapeTool( document->Main() );
    auto colorTool = XCAFDoc_DocumentTool::ColorTool( document->Main() );

    reportProgress( settings.callback, 0.85f );

    auto root = std::make_shared<Object>();
    root->select( true );

    std::stack<std::shared_ptr<Object>> objStack;
    objStack.push( root );

    struct TriangulationTask
    {
        TopoDS_Shape shape;
        std::shared_ptr<ObjectMesh> mesh;
        std::vector<Triangle3f> triples;
        Mesh part;

        TriangulationTask( TopoDS_Shape shape, std::shared_ptr<ObjectMesh> mesh )
            : shape( shape )
            , mesh( mesh )
        {}
    };
    std::vector<TriangulationTask> triangulationTasks;

    std::function<void ( const TDF_Label& label )> readShape;
    readShape = [&] ( const TDF_Label& label )
    {
        const auto shape = shapeTool->GetShape( label );
        const auto name = readName( label );

        const auto& location = shape.Location();
        const auto xf = AffineXf3f( toXf( location.Transformation() ) );

# if 0
        std::ostringstream oss;
        if ( shapeTool->IsShape( label ) )
            oss << " shape";
        if ( shapeTool->IsSimpleShape( label ) )
            oss << " simple-shape";
        if ( shapeTool->IsReference( label ) )
            oss << " reference";
        if ( shapeTool->IsAssembly( label ) )
            oss << " assembly";
        if ( shapeTool->IsComponent( label ) )
            oss << " component";
        if ( shapeTool->IsCompound( label ) )
            oss << " compound";
        //if ( shapeTool->IsSubShape( label ) )
        //    oss << " subshape";
        if ( !shapeTool->IsSubShape( label ) )
            spdlog::info( "{}{}:{}", std::string( 2 * objStack.size(), ' ' ), name, oss.str() );
# endif

        if ( shapeTool->IsAssembly( label ) )
        {
            auto obj = std::make_shared<Object>();
            obj->setName( name );
            obj->setXf( xf );
            obj->select( true );
            objStack.top()->addChild( obj );

            objStack.push( obj );
            for ( TDF_ChildIterator it( label ); it.More(); it.Next() )
                readShape( it.Value() );
            objStack.pop();
        }
        else if ( shapeTool->IsReference( label ) )
        {
            auto objMesh = std::make_shared<ObjectMesh>();
            objMesh->setName( name );
            objMesh->setMesh( std::make_shared<Mesh>() );
            objMesh->setXf( xf );
            objMesh->select( true );
            objStack.top()->addChild( objMesh );

            triangulationTasks.emplace_back( shape, objMesh );

            objStack.push( objMesh );
            TDF_Label ref;
            [[maybe_unused]] auto res = shapeTool->GetReferredShape( label, ref );
            assert( res );
            for ( TDF_ChildIterator it( ref ); it.More(); it.Next() )
                readShape( it.Value() );
            objStack.pop();
        }
        else
        {
            if ( shapeTool->IsSubShape( label ) )
            {
                auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( objStack.top() );
                assert( objMesh );
                assert( objMesh->mesh() );
                triangulationTasks.emplace_back( shape, objMesh );
            }
            else
            {
                auto objMesh = std::make_shared<ObjectMesh>();
                objMesh->setName( name );
                objMesh->setMesh( std::make_shared<Mesh>() );
                objMesh->setXf( xf );
                objMesh->select( true );
                objStack.top()->addChild( objMesh );

                triangulationTasks.emplace_back( shape, objMesh );
            }
        }
    };

    TDF_LabelSequence labels;
    shapeTool->GetFreeShapes( labels );
    for ( auto i = 1; i <= labels.Length(); ++i )
        readShape( labels.Value( i ) );

    assert( objStack.size() == 1 );
    assert( objStack.top() == root );

    tbb::task_arena taskArena;
    taskArena.execute( [&]
    {
        tbb::task_group taskGroup;
        for ( auto i = 0; i < triangulationTasks.size(); ++i )
        {
            auto& task = triangulationTasks[i];
            task.triples = loadShape( task.shape, true );
            taskGroup.run( [i, &triangulationTasks]
            {
                auto& task = triangulationTasks[i];
                task.part = Mesh::fromPointTriples( task.triples, true );
                task.triples.clear();
            } );
        }
        taskGroup.wait();
    } );

    reportProgress( settings.callback, 0.95f );

    for ( auto& task : triangulationTasks )
    {
        task.mesh->varMesh()->addPart( task.part );
        task.part = Mesh();
    }

    return root;
}

} // namespace MR::MeshLoad
#endif
