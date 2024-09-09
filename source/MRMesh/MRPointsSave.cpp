#include "MRPointsSave.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPch/MRFmt.h"
#include <fstream>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

namespace MR
{

namespace PointsSave
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

} //anonymous namespace

VoidOrErrStr toAsc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsc( points, out, settings );
}

VoidOrErrStr toAsc( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    MR_TIMER
    const bool saveNormals = cloud.points.size() <= cloud.normals.size();
    const size_t totalPoints = settings.saveValidOnly ? cloud.validPoints.count() : cloud.points.size();
    size_t numSaved = 0;

    NormalXfMatrix normXf( settings.xf );
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.saveValidOnly && !cloud.validPoints.test( v ) )
            continue;
        auto saveVertex = [&]( auto && p )
        {
            out << fmt::format( "{} {} {}", p.x, p.y, p.z );
        };
        auto saveNormal = [&]( auto && n )
        {
            out << fmt::format( " {} {} {}", n.x, n.y, n.z );
        };
        if ( settings.xf )
        {
            saveVertex( applyDouble( settings.xf, cloud.points[v] ) );
            if ( saveNormals )
                saveNormal( applyDouble( normXf, cloud.normals[v] ) );
        }
        else
        {
            saveVertex( cloud.points[v] );
            if ( saveNormals )
                saveNormal( cloud.normals[v] );
        }
        out << '\n';
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in ASC-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, settings );
}

VoidOrErrStr toPly( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    MR_TIMER
    const size_t totalPoints = settings.saveValidOnly ? cloud.validPoints.count() : cloud.points.size();

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << totalPoints << "\nproperty float x\nproperty float y\nproperty float z\n";

    const bool saveNormals = cloud.points.size() <= cloud.normals.size();
    if ( saveNormals )
        out << "property float nx\nproperty float ny\nproperty float nz\n";

    if ( settings.colors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";

    static_assert( sizeof( cloud.points.front() ) == 12, "wrong size of Vector3f" );
#pragma pack(push, 1)
    struct PlyColor
    {
        unsigned char r = 0, g = 0, b = 0;
    };
#pragma pack(pop)
    static_assert( sizeof( PlyColor ) == 3, "check your padding" );

    NormalXfMatrix normXf( settings.xf );
    size_t numSaved = 0;
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.saveValidOnly && !cloud.validPoints.test( v ) )
            continue;
        const Vector3f p = applyFloat( settings.xf, cloud.points[v] );
        out.write( ( const char* )&p, 12 );
        if ( saveNormals )
        {
            const Vector3f n = applyFloat( normXf, cloud.normals[v] );
            out.write( ( const char* )&n, 12 );
        }
        if ( settings.colors )
        {
            const auto c = ( *settings.colors )[v];
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

#ifndef MRMESH_NO_OPENCTM
VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const CtmSavePointsOptions& options )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( points, out, options );
}

VoidOrErrStr toCtm( const PointCloud& cloud, std::ostream& out, const CtmSavePointsOptions& options )
{
    MR_TIMER

    if ( (  options.saveValidOnly && !cloud.validPoints.any() ) ||
         ( !options.saveValidOnly && cloud.points.empty() ) )
        return unexpected( "Cannot save empty point cloud in CTM format" );
    // the only fake triangle with point #0 in all 3 vertices
    std::vector<CTMuint> aIndices{ 0,0,0 };

    class ScopedCtmConext
    {
        CTMcontext context_ = ctmNewContext( CTM_EXPORT );
    public:
        ~ScopedCtmConext()
        {
            ctmFreeContext( context_ );
        }
        operator CTMcontext()
        {
            return context_;
        }
    } context;

    ctmFileComment( context, options.comment );
    ctmCompressionMethod( context, CTM_METHOD_MG1 );
    ctmCompressionLevel( context, options.compressionLevel );

    const bool saveNormals = cloud.points.size() <= cloud.normals.size();
    CTMuint aVertexCount = CTMuint( options.saveValidOnly ? cloud.validPoints.count() : cloud.points.size() );

    VertCoords points;
    VertNormals normals;
    NormalXfMatrix normXf( options.xf );
    if ( options.saveValidOnly || options.xf )
    {
        if ( options.saveValidOnly )
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
            if ( options.saveValidOnly )
            {
                normals.reserve( aVertexCount );
                for ( auto v : cloud.validPoints )
                    normals.push_back( applyFloat( normXf, cloud.normals[v] ) );
            }
        }
        else
        {
            transformNormals( cloud.normals, cloud.validPoints, normXf, normals );
            assert( cloud.normals.size() == normals.size() );
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

    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( options.colors && options.colors->size() >= cloud.points.size() )
    {
        colors4f.reserve( aVertexCount );
        for ( auto v = 0_v; v < cloud.points.size(); ++v )
        {
            if ( options.saveValidOnly && !cloud.validPoints.test( v ) )
                continue;
            colors4f.push_back( Vector4f{ ( *options.colors )[v] } );
        }
        ctmAddAttribMap( context, ( const CTMfloat* )colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error encoding in CTM-format colors" );

    struct SaveData
    {
        std::function<bool( float )> callbackFn{};
        std::ostream* stream;
        size_t sum{ 0 };
        size_t blockSize{ 0 };
        size_t maxSize{ 0 };
        bool wasCanceled{ false };
    } saveData;
    if ( options.progress )
    {
        saveData.callbackFn = [callback = options.progress, &saveData] ( float progress )
        {
            // calculate full progress in partial-linear scale (we don't know compressed size and it less than real size)
            // conversion rules:
            // step 1) range (0, rangeBefore) is converted in range (0, rangeAfter)
            // step 2) moving on to new ranges: (rangeBefore, 1) and (rangeAfter, 1)
            // step 3) go to step 1)
            const float rangeBefore = 0.2f;
            const float rangeAfter = 0.7f;
            progress = ( saveData.sum + progress * saveData.blockSize ) / saveData.maxSize;
            float newProgress = 0.f;
            for ( ; newProgress < 98.5f; )
            {
                if ( progress < rangeBefore )
                {
                    newProgress += progress / rangeBefore * rangeAfter * ( 1 - newProgress );
                    break;
                }
                else
                {
                    progress = ( progress - rangeBefore ) / ( 1 - rangeBefore );
                    newProgress += ( 1 - newProgress ) * rangeAfter;
                }
            }
            return callback( newProgress );
        };
    }
    saveData.stream = &out;
    saveData.maxSize = aVertexCount * sizeof( Vector3f ) + cloud.normals.size() * sizeof( Vector3f ) + 150; // 150 - reserve for some ctm specific data
    ctmSaveCustom( context, [] ( const void* buf, CTMuint size, void* data )
    {
        SaveData& saveData = *reinterpret_cast< SaveData* >( data );
        std::ostream& outStream = *saveData.stream;
        saveData.blockSize = size;

        saveData.wasCanceled |= !MR::writeByBlocks( outStream, (const char*) buf, size, saveData.callbackFn, 1u << 12 );
        saveData.sum += size;
        if ( saveData.wasCanceled )
            return 0u;

        return outStream.good() ? size : 0;
    }, &saveData );

    if ( saveData.wasCanceled )
        return unexpected( std::string( "Saving canceled" ) );
    if ( !out || ctmGetError( context ) != CTM_NONE )
        return unexpected( std::string( "Error saving in CTM-format" ) );

    reportProgress( options.progress, 1.f );
    return {};
}

VoidOrErrStr toCtm( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toCtm( points, file, CtmSavePointsOptions{ settings } );
}

VoidOrErrStr toCtm( const PointCloud& points, std::ostream& out, const SaveSettings& settings )
{
    return toCtm( points, out, CtmSavePointsOptions{ settings } );
}
#endif

VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getPointsSaver( ext );
    if ( !saver.fileSave )
        return unexpected( std::string( "unsupported file extension" ) );

    return saver.fileSave( points, file, settings );
}
VoidOrErrStr toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const SaveSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getPointsSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    return saver.streamSave( points, out, settings );
}

MR_ADD_POINTS_SAVER( IOFilter( "ASC (.asc)",        "*.asc" ), toAsc )
MR_ADD_POINTS_SAVER( IOFilter( "PLY (.ply)",        "*.ply" ), toPly )
#ifndef MRMESH_NO_OPENCTM
MR_ADD_POINTS_SAVER( IOFilter( "CTM (.ctm)",        "*.ctm" ), toCtm )
#endif

} // namespace PointsSave

} // namespace MR
