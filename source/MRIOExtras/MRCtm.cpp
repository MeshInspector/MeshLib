#include "MRCtm.h"
#ifndef MRIOEXTRAS_NO_CTM

#include <MRMesh/MRColor.h>
#include <MRMesh/MRFinally.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRIOParsing.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRProgressReadWrite.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRTimer.h>

// MSBuild compatibility
#if __has_include(<openctm.h>)
#include <openctm.h>
#else
#include <OpenCTM/openctm.h>
#endif

#include <fstream>

namespace MR
{

namespace
{

class NormalXfMatrix
{
public:
    /// given transformation of points, prepares matrix for transformation of their normals
    explicit NormalXfMatrix( const AffineXf3d * xf )
    {
        if ( xf )
        {
            normXf_ = xf->A.inverse().transposed();
            pNormXf_ = &normXf_;
        }
    }
    operator const Matrix3d *() const { return pNormXf_; }

private:
    Matrix3d normXf_;
    const Matrix3d * pNormXf_ = nullptr;
};

class Saver
{
public:
    Saver( CTMcontext context, std::ostream& outStream, const ProgressCallback & cb )
        : context_( context ), stream_( outStream ), cb_( cb ) {}
    Expected<void> run();

private:
    CTMcontext context_;
    std::ostream& stream_;
    const ProgressCallback & cb_;
    float lastProgress_ = 0;
};

Expected<void> Saver::run()
{
    MR_TIMER;
    if ( !stream_ )
        return unexpected( std::string( "Bad stream before CTM-encoding" ) );
    ctmSaveCustom( context_, []( const void* buf, CTMuint size, void* data ) -> CTMuint
    {
        Saver& self = *reinterpret_cast< Saver* >( data );
        writeByBlocks( self.stream_, (const char*) buf, size );
        if ( !self.stream_ || !reportProgress( self.cb_, self.lastProgress_ ) )
            return 0; // stop
        return size;
    },
    []( size_t pos, size_t total, void* data )
    {
        assert( pos < total );
        Saver& self = *reinterpret_cast< Saver* >( data );
        self.lastProgress_ = float( pos ) / total;
        return reportProgress( self.cb_, self.lastProgress_ ) ? 0 : 1;
    },
    this );

    if ( !reportProgress( cb_, 1.0f ) )
        return unexpectedOperationCanceled();
    if ( !stream_ )
        return unexpected( std::string( "Error writing in stream during CTM-encoding" ) );
    if ( auto err = ctmGetError( context_ ); err != CTM_NONE )
        return unexpected( "Error " + std::to_string( err ) + " during CTM-encoding" );
    return {};
}

} // anonymous namespace

namespace MeshLoad
{

Expected<Mesh> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromCtm( in, settings ), file );
}

Expected<Mesh> fromCtm( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    auto context = ctmNewContext( CTM_IMPORT );
    MR_FINALLY { ctmFreeContext( context ); };

    struct LoadData
    {
        std::function<bool( float )> callbackFn{};
        std::istream* stream;
        bool wasCanceled{ false };
    } loadData;
    loadData.stream = &in;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    if ( settings.callback )
    {
        loadData.callbackFn = [callback = settings.callback, posStart, streamSize, &in] ( float )
        {
            float progress = float( in.tellg() - posStart ) / float( streamSize );
            return callback( progress );
        };
    }

    ctmLoadCustom( context, []( void * buf, CTMuint size, void * data )
    {
        LoadData& loadData = *reinterpret_cast<LoadData*>( data );
        auto& stream = *loadData.stream;
        auto pos = stream.tellg();
        loadData.wasCanceled |= !readByBlocks( stream, ( char* )buf, size, loadData.callbackFn, 1u << 12 );
        if ( loadData.wasCanceled )
            return 0u;
        return (CTMuint)( stream.tellg() - pos );
    }, &loadData );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto triCount  = ctmGetInteger( context, CTM_TRIANGLE_COUNT );
    auto vertices  = ctmGetFloatArray( context, CTM_VERTICES );
    auto indices   = ctmGetIntegerArray( context, CTM_INDICES );
    if ( loadData.wasCanceled )
        return unexpectedOperationCanceled();
    if ( ctmGetError(context) != CTM_NONE )
        return unexpected( "Error reading CTM format" );

    // even if we save false triangle (0,0,0) in MG2 format, it can be open as triangle (i,i,i)
    if ( triCount == 1 && indices[0] == indices[1] && indices[0] == indices[2] )
    {
        // CTM file is representing points, but it was written with the library requiring the presence of at least one triangle
        triCount = 0;
    }

    if ( settings.colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            settings.colors->resize( vertCount );
            for ( VertId i{ 0 }; CTMuint( i ) < vertCount; ++i )
            {
                auto j = 4 * i;
                (*settings.colors)[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    if ( settings.normals && ctmGetInteger( context, CTM_HAS_NORMALS ) == CTM_TRUE )
    {
        auto normals = ctmGetFloatArray( context, CTM_NORMALS );
        settings.normals->resize( vertCount );
        for ( VertId i{0}; i < (int) vertCount; ++i )
            (*settings.normals)[i] = Vector3f( normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] );
    }

    Mesh mesh;
    mesh.points.resize( vertCount );
    for ( VertId i{0}; i < (int)vertCount; ++i )
        mesh.points[i] = Vector3f( vertices[3*i], vertices[3*i+1], vertices[3*i+2] );

    Triangulation t;
    t.reserve( triCount );
    for ( FaceId i{0}; i < (int)triCount; ++i )
        t.push_back( { VertId( (int)indices[3*i] ), VertId( (int)indices[3*i+1] ), VertId( (int)indices[3*i+2] ) } );

    mesh.topology = MeshBuilder::fromTriangles( t, { .skippedFaceCount = settings.skippedFaceCount } );
    if ( mesh.topology.lastValidVert() + 1 > mesh.points.size() )
        return unexpected( "vertex id is larger than total point coordinates" );
    return mesh;
}

MR_ADD_MESH_LOADER( IOFilter( "Compact triangle-based mesh (.ctm)", "*.ctm" ), fromCtm )

} // namespace MeshLoad

namespace MeshSave
{

Expected<void> toCtm( const Mesh & mesh, const std::filesystem::path & file, const CtmSaveOptions& options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( mesh, out, options );
}

Expected<void> toCtm( const Mesh & mesh, std::ostream & out, const CtmSaveOptions& options )
{
    MR_TIMER;

    class ScopedCtmConext
    {
        CTMcontext context_ = ctmNewContext( CTM_EXPORT );
    public:
        ~ScopedCtmConext() { ctmFreeContext( context_ ); }
        operator CTMcontext() { return context_; }
    } context;

    ctmFileComment( context, options.comment );
    switch ( options.meshCompression )
    {
    default:
        assert( false );
        [[fallthrough]];
    case CtmSaveOptions::MeshCompression::None:
        ctmCompressionMethod( context, CTM_METHOD_RAW );
        break;
    case CtmSaveOptions::MeshCompression::Lossless:
        ctmCompressionMethod( context, CTM_METHOD_MG1 );
        break;
    case CtmSaveOptions::MeshCompression::Lossy:
        ctmCompressionMethod( context, CTM_METHOD_MG2 );
        ctmVertexPrecision( context, options.vertexPrecision );
        break;
    }
    ctmRearrangeTriangles( context, options.packPrimitives ? 1 : 0 );
    ctmCompressionLevel( context, options.compressionLevel );

    const VertRenumber vertRenumber( mesh.topology.getValidVerts(), options.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = mesh.topology.lastValidVert();

    std::vector<CTMuint> aIndices;
    const auto fLast = mesh.topology.lastValidFace();
    const auto numSaveFaces = options.packPrimitives ? mesh.topology.numValidFaces() : int( fLast + 1 );
    aIndices.reserve( numSaveFaces * 3 );
    for ( FaceId f{0}; f <= fLast; ++f )
    {
        if ( mesh.topology.hasFace( f ) )
        {
            VertId v[3];
            mesh.topology.getTriVerts( f, v );
            for ( int i = 0; i < 3; ++i )
                aIndices.push_back( vertRenumber( v[i] ) );
        }
        else if ( !options.packPrimitives )
        {
            for ( int i = 0; i < 3; ++i )
                aIndices.push_back( 0 );
        }
    }
    assert( aIndices.size() == numSaveFaces * 3 );

    CTMuint aVertexCount = numPoints;
    VertCoords buf;
    const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), options.xf, buf, &vertRenumber );
    ctmDefineMesh( context,
        (const CTMfloat *)xfVerts.data(), aVertexCount,
        aIndices.data(), numSaveFaces, nullptr );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( options.colors )
    {
        colors4f.reserve( aVertexCount );
        for ( VertId i{ 0 }; i <= lastVertId; ++i )
        {
            if ( options.onlyValidPoints && !mesh.topology.hasVert( i ) )
                continue;
            colors4f.push_back( Vector4f( ( *options.colors )[i] ) );
        }
        assert( colors4f.size() == aVertexCount );

        ctmAddAttribMap( context, (const CTMfloat*) colors4f.data(), "Color" );
    }

    if ( ctmGetError(context) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format" );

    return Saver( context, out, options.progress ).run();
}

Expected<void> toCtm( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toCtm( mesh, file, CtmSaveOptions { settings } );
}

Expected<void> toCtm( const Mesh& mesh, std::ostream& out, const SaveSettings& settings )
{
    return toCtm( mesh, out, CtmSaveOptions { settings } );
}

MR_ADD_MESH_SAVER( IOFilter( "CTM (.ctm)", "*.ctm" ), toCtm, { .storesVertexColors = true } )

} // namespace MeshSave

namespace PointsLoad
{

Expected<MR::PointCloud> fromCtm( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromCtm( in, settings ), file );
}

Expected<MR::PointCloud> fromCtm( std::istream& in, const PointsLoadSettings& settings )
{
    MR_TIMER;

    auto context = ctmNewContext( CTM_IMPORT );
    MR_FINALLY { ctmFreeContext( context ); };

    struct LoadData
    {
        std::function<bool( float )> callbackFn{};
        std::istream* stream;
        bool wasCanceled{ false };
    } loadData;
    loadData.stream = &in;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    if ( settings.callback )
    {
        loadData.callbackFn = [cb = settings.callback, posStart, streamSize, &in]( float )
        {
            float progress = float( in.tellg() - posStart ) / float( streamSize );
            return cb( progress );
        };
    }

    ctmLoadCustom( context, []( void* buf, CTMuint size, void* data )
    {
        LoadData& loadData = *reinterpret_cast< LoadData* >( data );
        auto& stream = *loadData.stream;
        auto pos = stream.tellg();
        loadData.wasCanceled |= !readByBlocks( stream, (char*)buf, size, loadData.callbackFn, 1u << 12 );
        if ( loadData.wasCanceled )
            return 0u;
        return ( CTMuint )( stream.tellg() - pos );
    }, & loadData );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto vertices = ctmGetFloatArray( context, CTM_VERTICES );
    if ( loadData.wasCanceled )
        return unexpectedOperationCanceled();
    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error reading CTM format" );

    if ( settings.colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            settings.colors->resize( vertCount );
            for ( VertId i{ 0 }; CTMuint( i ) < vertCount; ++i )
            {
                auto j = 4 * i;
                ( *settings.colors )[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    PointCloud points;
    points.points.resize( vertCount );
    points.validPoints.resize( vertCount, true );
    for ( VertId i{0}; i < (int) vertCount; ++i )
        points.points[i] = Vector3f( vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2] );

    if ( ctmGetInteger( context, CTM_HAS_NORMALS ) == CTM_TRUE )
    {
        auto normals = ctmGetFloatArray( context, CTM_NORMALS );
        points.normals.resize( vertCount );
        for ( VertId i{0}; i < (int) vertCount; ++i )
            points.normals[i] = Vector3f( normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] );
    }

    return points;
}

MR_ADD_POINTS_LOADER( IOFilter( "CTM (.ctm)", "*.ctm" ), fromCtm )

} // namespace PointsLoad

namespace PointsSave
{

Expected<void> toCtm( const PointCloud& points, const std::filesystem::path& file, const CtmSavePointsOptions& options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( points, out, options );
}

Expected<void> toCtm( const PointCloud& cloud, std::ostream& out, const CtmSavePointsOptions& options )
{
    MR_TIMER;

    if ( (  options.onlyValidPoints && !cloud.validPoints.any() ) ||
         ( !options.onlyValidPoints && cloud.points.empty() ) )
        return unexpected( "Cannot save empty point cloud in CTM format" );
    // the only fake triangle with point #0 in all 3 vertices
    std::vector<CTMuint> aIndices{ 0,0,0 };

    auto context = ctmNewContext( CTM_EXPORT );
    MR_FINALLY { ctmFreeContext( context ); };

    ctmFileComment( context, options.comment );
    ctmCompressionMethod( context, CTM_METHOD_MG1 );
    ctmCompressionLevel( context, options.compressionLevel );

    const bool saveNormals = cloud.hasNormals();
    CTMuint aVertexCount = CTMuint( options.onlyValidPoints ? cloud.validPoints.count() : cloud.points.size() );

    VertCoords points;
    VertNormals normals;
    NormalXfMatrix normXf( options.xf );
    if ( options.onlyValidPoints || options.xf )
    {
        if ( options.onlyValidPoints )
        {
            points.reserve( aVertexCount );
            for ( auto v : cloud.validPoints )
                points.push_back( applyFloat( options.xf, cloud.points[v] ) );
        }
        else
        {
            transformPoints( cloud.points, cloud.validPoints, options.xf, points );
            assert( cloud.points.size() == points.size() );
        }

        if ( saveNormals )
        {
            if ( options.onlyValidPoints )
            {
                normals.reserve( aVertexCount );
                for ( auto v : cloud.validPoints )
                    normals.push_back( applyFloat( normXf, cloud.normals[v] ) );
            }
            else
            {
                transformNormals( cloud.normals, cloud.validPoints, normXf, normals );
                assert( cloud.normals.size() == normals.size() );
            }
        }

        ctmDefineMesh( context,
                       ( const CTMfloat* )points.data(), aVertexCount,
                       aIndices.data(), 1, saveNormals ? ( const CTMfloat* )normals.data() : nullptr );
    }
    else
    {
        ctmDefineMesh( context,
                       ( const CTMfloat* )cloud.points.data(), aVertexCount,
                       aIndices.data(), 1, saveNormals ? ( const CTMfloat* )cloud.normals.data() : nullptr );
    }

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( options.colors && options.colors->size() >= cloud.points.size() )
    {
        colors4f.reserve( aVertexCount );
        for ( auto v = 0_v; v < cloud.points.size(); ++v )
        {
            if ( options.onlyValidPoints && !cloud.validPoints.test( v ) )
                continue;
            colors4f.push_back( Vector4f{ ( *options.colors )[v] } );
        }
        ctmAddAttribMap( context, ( const CTMfloat* )colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format" );

    return Saver( context, out, options.progress ).run();
}

Expected<void> toCtm( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toCtm( points, file, CtmSavePointsOptions{ settings } );
}

Expected<void> toCtm( const PointCloud& points, std::ostream& out, const SaveSettings& settings )
{
    return toCtm( points, out, CtmSavePointsOptions{ settings } );
}

MR_ADD_POINTS_SAVER( IOFilter( "CTM (.ctm)", "*.ctm" ), toCtm )

} // namespace PointsSave

} // namespace MR
#endif
