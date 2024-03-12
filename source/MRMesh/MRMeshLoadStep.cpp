#include "MRMeshLoadStep.h"
#ifndef MRMESH_NO_OPENCASCADE
#include "MRFinally.h"
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
#include <opencascade/Quantity_ColorRGBA.hxx>
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
#include <opencascade/XCAFDoc_ColorTool.hxx>
#include <opencascade/XCAFDoc_DocumentTool.hxx>
#include <opencascade/XCAFDoc_ShapeTool.hxx>
#else
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Message.hxx>
#include <Message_Printer.hxx>
#include <Message_PrinterOStream.hxx>
#include <Quantity_ColorRGBA.hxx>
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
#include <XCAFDoc_ColorTool.hxx>
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
// updated progress indication API support
// https://dev.opencascade.org/doc/overview/html/occt__upgrade.html#upgrade_750_ProgressIndicator
#define MODERN_PROGRESS_INDICATION_SUPPORTED ( ( OCC_VERSION_MAJOR > 7 ) || ( OCC_VERSION_MAJOR == 7 && OCC_VERSION_MINOR >= 5 ) )
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

#if MODERN_PROGRESS_INDICATION_SUPPORTED
class ProgressIndicator final : public Message_ProgressIndicator
{
public:
    explicit ProgressIndicator( ProgressCallback callback )
        : callback_( std::move( callback ) )
    {}

    Standard_Boolean UserBreak() override { return interrupted_; }

    void Show( const Message_ProgressScope&, const Standard_Boolean ) override
    {
        interrupted_ = !reportProgress( callback_, (float)GetPosition() );
    }

private:
    ProgressCallback callback_;
    bool interrupted_{ false };
};
#else
#pragma message( "Progress indication is currently unsupported for OpenCASCADE versions prior to 7.4" )
#endif

Color toColor( const Quantity_ColorRGBA& rgba )
{
    const auto& rgb = rgba.GetRGB();
    return { (float)rgb.Red(), (float)rgb.Green(), (float)rgb.Blue(), rgba.Alpha() };
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

struct StepLoader
{
public:
    [[nodiscard]] std::shared_ptr<Object> rootObject() const
    {
        return rootObj_;
    }

    void loadModelStructure( STEPControl_Reader& reader, [[maybe_unused]] ProgressCallback callback )
    {
        MR_TIMER

        {
            MR_NAMED_TIMER( "transfer roots" )

#if MODERN_PROGRESS_INDICATION_SUPPORTED
            ProgressIndicator progress( subprogress( callback, 0.00f, 0.80f ) );
            reader.TransferRoots( progress.Start() );
#else
            reader.TransferRoots();
#endif
        }

        std::deque<TopoDS_Shape> solids;
        for ( auto shi = 1; shi <= reader.NbShapes(); ++shi )
        {
            const auto shape = reader.Shape( shi );

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

        rootObj_ = std::make_shared<Object>();
        rootObj_->select( true );

        auto solidIndex = 1;
        for ( const auto& shape : solids )
        {
            const auto& location = shape.Location();
            const auto xf = AffineXf3f( toXf( location.Transformation() ) );

            auto objMesh = std::make_shared<ObjectMesh>();
            objMesh->setName( fmt::format( "Solid{}", solidIndex++ ) );
            objMesh->setMesh( std::make_shared<Mesh>() );
            objMesh->setXf( xf );
            objMesh->select( true );
            rootObj_->addChild( objMesh );

            meshTriangulationContexts_.emplace_back( shape, objMesh, std::nullopt, std::nullopt );
        }

        if ( solids.size() > 1 )
        {
            // create a fictional 'assembly' object
            auto assemblyObj = rootObj_;
            rootObj_ = std::make_shared<Object>();
            rootObj_->select( true );
            rootObj_->addChild( std::move( assemblyObj ) );
        }
    }

    /// load object structure without actual geometry data
    void loadModelStructure( const Handle( TDocStd_Document )& document )
    {
        MR_TIMER

        const auto shapeTool = XCAFDoc_DocumentTool::ShapeTool( document->Main() );
        colorTool_ = XCAFDoc_DocumentTool::ColorTool( document->Main() );

        rootObj_ = std::make_shared<Object>();
        rootObj_->select( true );
        objStack_.push( rootObj_ );

        TDF_LabelSequence shapes, colors;
        shapeTool->GetFreeShapes( shapes );
        colorTool_->GetColors( colors );

        for ( TDF_LabelSequence::Iterator it( shapes ); it.More(); it.Next() )
            readLabel_( it.Value() );

        colorTool_.Nullify();

        assert( objStack_.size() == 1 );
        assert( objStack_.top() == rootObj_ );
    }

    /// load and triangulate meshes
    void loadMeshes()
    {
        MR_TIMER

        tbb::task_arena taskArena;
        taskArena.execute( [&]
        {
            tbb::task_group taskGroup;
            for ( auto& ctx : meshTriangulationContexts_ )
            {
                ctx.shape = triangulateShape_( ctx.shape );
                // reset transformation as it already is loaded
                ctx.shape.Location( TopLoc_Location() );
                ctx.triples = loadShape_( ctx.shape );

                taskGroup.run( [&ctx]
                {
                    ctx.part = Mesh::fromPointTriples( ctx.triples, true );
                    ctx.triples.clear();
                } );
            }
            taskGroup.wait();
        } );

        for ( auto& ctx : meshTriangulationContexts_ )
        {
            auto& mesh = *ctx.mesh->varMesh();

            if ( ctx.faceColor )
            {
                FaceMap fmap;
                mesh.addPart( ctx.part, &fmap );

                auto faceColors = ctx.mesh->getFacesColorMap();
                faceColors.resize( mesh.topology.lastValidFace() + 1, Color::gray() );
                for ( const auto of : ctx.part.topology.getValidFaces() )
                    faceColors[fmap[of]] = *ctx.faceColor;
                ctx.mesh->setColoringType( ColoringType::FacesColorMap );
                ctx.mesh->setFacesColorMap( std::move( faceColors ) );
            }
            else
            {
                mesh.addPart( ctx.part );
            }

            ctx.part = Mesh();
        }
    }

private:
    void readLabel_( const TDF_Label& label )
    {
        using ShapeTool = XCAFDoc_ShapeTool;

        const auto shape = ShapeTool::GetShape( label );
        const auto name = readName_( label );

        const auto& location = shape.Location();
        const auto xf = AffineXf3f( toXf( location.Transformation() ) );

        if ( ShapeTool::IsAssembly( label ) )
        {
            auto obj = std::make_shared<Object>();
            obj->setName( name );
            obj->setXf( xf );
            obj->select( true );
            objStack_.top()->addChild( obj );

            iterateLabel_( label, obj );
        }
        else if ( ShapeTool::IsSubShape( label ) )
        {
            auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( objStack_.top() );
            assert( objMesh );
            assert( objMesh->mesh() );

            std::optional<Color> faceColor, edgeColor;
            Quantity_ColorRGBA color;
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorGen, color ) )
                faceColor = toColor( color );
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorSurf, color ) )
                faceColor = toColor( color );
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorCurv, color ) )
                edgeColor = toColor( color );

            meshTriangulationContexts_.emplace_back( shape, objMesh, faceColor, edgeColor );
        }
        else
        {
            auto objMesh = std::make_shared<ObjectMesh>();
            objMesh->setName( name );
            objMesh->setMesh( std::make_shared<Mesh>() );
            objMesh->setXf( xf );
            objMesh->select( true );
            objStack_.top()->addChild( objMesh );

            std::optional<Color> faceColor, edgeColor;
            Quantity_ColorRGBA color;
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorGen, color ) )
                faceColor = toColor( color );
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorSurf, color ) )
                faceColor = toColor( color );
            if ( colorTool_->GetColor( shape, XCAFDoc_ColorCurv, color ) )
                edgeColor = toColor( color );

            meshTriangulationContexts_.emplace_back( shape, objMesh, faceColor, edgeColor );

            if ( ShapeTool::IsReference( label ) )
            {
                TDF_Label ref;
                [[maybe_unused]] const auto isRef = ShapeTool::GetReferredShape( label, ref );
                assert( isRef );

                iterateLabel_( ref, objMesh );
            }
        }
    }

    void iterateLabel_( const TDF_Label& label, const std::shared_ptr<Object>& obj )
    {
        objStack_.push( obj );
        for ( TDF_ChildIterator it( label ); it.More(); it.Next() )
            readLabel_( it.Value() );
        objStack_.pop();
    }

private:
    static std::string readName_( const TDF_Label& label )
    {
        assert( !label.IsNull() );

        Handle( TDataStd_Name ) name;
        if ( label.FindAttribute( TDataStd_Name::GetID(), name ) != Standard_True )
            return {};

        const auto& str = name->Get();
        std::string result( str.LengthOfCString(), '\0' );
        auto* resultCStr = result.data();
        str.ToUTF8CString( resultCStr );

        return result;
    }

    static TopoDS_Shape triangulateShape_( const TopoDS_Shape& shape )
    {
        MR_TIMER

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
        return incMesh.Shape();
    }

    static std::vector<Triangle3f> loadShape_( const TopoDS_Shape& shape )
    {
        MR_TIMER

        struct FeatureData
        {
            TopAbs_Orientation orientation;
            Handle( Poly_Triangulation ) triangulation;
            TopLoc_Location location;

            explicit FeatureData( const TopoDS_Shape& face )
                : orientation( face.Orientation() )
            {
                triangulation = BRep_Tool::Triangulation( TopoDS::Face( face ), location );
            }
        };

        std::deque<FeatureData> features;
        for ( auto faceExp = TopExp_Explorer( shape, TopAbs_FACE ); faceExp.More(); faceExp.Next() )
            features.emplace_back( faceExp.Current() );

        size_t totalVertexCount = 0, totalFaceCount = 0;
        for ( const auto& feature : features )
        {
            if ( feature.triangulation.IsNull() )
                continue;

            totalVertexCount += feature.triangulation->NbNodes();
            totalFaceCount += feature.triangulation->NbTriangles();
        }

        std::vector<Vector3f> points;
        points.reserve( totalVertexCount );

        std::vector<Triangle3f> triples;
        triples.reserve( totalFaceCount );

        std::vector<FaceBitSet> parts;
        parts.reserve( features.size() );

        for ( const auto& feature : features )
        {
            if ( feature.triangulation.IsNull() )
                continue;

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

private:
    Handle( XCAFDoc_ColorTool ) colorTool_;

    std::shared_ptr<Object> rootObj_;
    std::stack<std::shared_ptr<Object>> objStack_;

    struct MeshTriangulationContext
    {
        TopoDS_Shape shape;
        std::shared_ptr<ObjectMesh> mesh;
        std::optional<Color> faceColor;
        std::optional<Color> edgeColor;
        std::vector<Triangle3f> triples;
        Mesh part;

        MeshTriangulationContext( TopoDS_Shape shape, std::shared_ptr<ObjectMesh> mesh, std::optional<Color> faceColor, std::optional<Color> edgeColor )
            : shape( std::move( shape ) )
            , mesh( std::move( mesh ) )
            , faceColor( std::move( faceColor ) )
            , edgeColor( std::move( edgeColor ) )
        {}
    };
    std::deque<MeshTriangulationContext> meshTriangulationContexts_;
};

#if !STEP_READSTREAM_SUPPORTED
std::filesystem::path getStepTemporaryDirectory()
{
    const auto path = std::filesystem::temp_directory_path() / "MeshLib_MeshLoadStep";
    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        std::filesystem::create_directory( path, ec );
    return path;
}

std::mutex cOpenCascadeTempFileMutex = {};
#endif

VoidOrErrStr readFromFile( STEPControl_Reader& reader, const std::filesystem::path& path )
{
    MR_TIMER

    const TCollection_AsciiString pathStr( path.c_str() );
    if ( reader.ReadFile( pathStr.ToCString() ) != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );
    return {};
}

VoidOrErrStr readFromStream( STEPControl_Reader& reader, std::istream& in )
{
    MR_TIMER

#if STEP_READSTREAM_SUPPORTED
    if ( reader.ReadStream( "STEP file", in ) != IFSelect_RetDone )
        return unexpected( "Failed to read STEP model" );
    return {};
#else
    std::unique_lock lock( cOpenCascadeTempFileMutex );

    const auto tempFilePath = getStepTemporaryDirectory() / "tempFile.step";
    std::error_code ec;
    MR_FINALLY {
        std::filesystem::remove( tempFilePath, ec );
    };

    {
        std::ofstream ofs( tempFilePath, std::ios::binary );
        if ( !ofs )
            return unexpected( std::string( "Cannot open buffer file" ) );

        ofs << in.rdbuf();
    }

    return readFromFile( reader, tempFilePath );
#endif
}

#if !STEP_READER_FIXED
VoidOrErrStr repairStepFile( STEPControl_Reader& reader )
{
    const auto model = reader.StepModel();
    const auto protocol = Handle( StepData_Protocol )::DownCast( model->Protocol() );

    StepData_StepWriter sw( model );
    sw.SendModel( protocol );

#if STEP_READSTREAM_SUPPORTED
    std::stringstream buffer;
    if ( !sw.Print( buffer ) )
        return unexpected( "Failed to repair STEP model" );

    buffer.seekp( 0, std::ios::beg );

    reader = STEPControl_Reader();
    return readFromStream( reader, buffer );
#else
    std::unique_lock lock( cOpenCascadeTempFileMutex );

    const auto auxFilePath = getStepTemporaryDirectory() / "auxFile.step";
    std::error_code ec;
    MR_FINALLY {
        std::filesystem::remove( auxFilePath, ec );
    };

    {
        // NOTE: opening the file in binary mode and passing to the StepData_StepWriter object lead to segfault
        std::ofstream ofs( auxFilePath );
        if ( !ofs )
            return unexpected( std::string( "Cannot open buffer file" ) );

        if ( !sw.Print( ofs ) )
            return unexpected( "Failed to repair STEP model" );
    }

    reader = STEPControl_Reader();
    return readFromFile( reader, auxFilePath );
#endif
}
#endif

std::mutex cOpenCascadeMutex = {};

Expected<Mesh> fromStepImpl( std::function<VoidOrErrStr ( STEPControl_Reader& )> readFunc, const MeshLoadSettings& settings )
{
    MR_TIMER

    std::unique_lock lock( cOpenCascadeMutex );

    STEPControl_Reader reader;
    {
        auto res = readFunc( reader );
        if ( !res )
            return unexpected( std::move( res.error() ) );
    }

    if ( !reportProgress( settings.callback, 0.50f ) )
        return unexpectedOperationCanceled();

    StepLoader loader;
    loader.loadModelStructure( reader, subprogress( settings.callback, 0.50f, 1.00f ) );
    loader.loadMeshes();

    Mesh result;
    for ( auto& objMesh : getAllObjectsInTree<ObjectMesh>( loader.rootObject().get() ) )
    {
        auto& mesh = objMesh->varMesh();
        mesh->transform( objMesh->worldXf() );
        result.addPart( *mesh );
    }
    return result;
}

Expected<std::shared_ptr<Object>> fromSceneStepFileImpl( std::function<VoidOrErrStr ( STEPControl_Reader& )> readFunc, const MeshLoadSettings& settings )
{
    MR_TIMER

    std::unique_lock lock( cOpenCascadeMutex );

    STEPCAFControl_Reader reader;
    {
        auto res = readFunc( reader.ChangeReader() );
        if ( !res )
            return unexpected( std::move( res.error() ) );
    }

    if ( !reportProgress( settings.callback, 0.25f ) )
        return unexpectedOperationCanceled();

    Handle( TDocStd_Document ) document = new TDocStd_Document( "MDTV-CAF" );
    {
        MR_NAMED_TIMER( "transfer data" )

        reader.SetNameMode( true );
        reader.SetColorMode( true );
#if MODERN_PROGRESS_INDICATION_SUPPORTED
        ProgressIndicator progress( subprogress( settings.callback, 0.25f, 0.85f ) );
        if ( reader.Transfer( document, progress.Start() ) != Standard_True )
#else
        if ( reader.Transfer( document ) != Standard_True )
#endif
            return unexpected( "Failed to read STEP model" );
    }

    if ( !reportProgress( settings.callback, 0.85f ) )
        return unexpectedOperationCanceled();

    StepLoader loader;
    loader.loadModelStructure( document );
    loader.loadMeshes();

    if ( !reportProgress( settings.callback, 1.00f ) )
        return unexpectedOperationCanceled();

    return loader.rootObject();
}

} // namespace

namespace MR::MeshLoad
{

Expected<Mesh> fromStep( const std::filesystem::path& path, const MeshLoadSettings& settings )
{
    return fromStepImpl( [&] ( STEPControl_Reader& reader )
    {
        return readFromFile( reader, path )
#if !STEP_READER_FIXED
        .and_then( [&] { return repairStepFile( reader ); } )
#endif
        ;
    }, settings );
}

Expected<Mesh> fromStep( std::istream& in, const MeshLoadSettings& settings )
{
    return fromStepImpl( [&] ( STEPControl_Reader& reader )
    {
        return readFromStream( reader, in )
#if !STEP_READER_FIXED
        .and_then( [&] { return repairStepFile( reader ); } )
#endif
        ;
    }, settings );
}

Expected<std::shared_ptr<Object>> fromSceneStepFile( const std::filesystem::path& path, const MeshLoadSettings& settings )
{
    return fromSceneStepFileImpl( [&] ( STEPControl_Reader& reader )
    {
        return readFromFile( reader, path )
#if !STEP_READER_FIXED
        .and_then( [&] { return repairStepFile( reader ); } )
#endif
        ;
    }, settings );
}

Expected<std::shared_ptr<Object>> fromSceneStepFile( std::istream& in, const MeshLoadSettings& settings )
{
    return fromSceneStepFileImpl( [&] ( STEPControl_Reader& reader )
    {
        return readFromStream( reader, in )
#if !STEP_READER_FIXED
        .and_then( [&] { return repairStepFile( reader ); } )
#endif
        ;
    }, settings );
}

} // namespace MR::MeshLoad

#endif // MRMESH_NO_OPENCASCADE
