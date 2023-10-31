#include "MRPointsSave.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPch/MRSpdlog.h"
#include <fstream>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

#if _MSC_VER <= 1929 // Visual Studio 2019
#pragma warning( disable : 4866 ) //compiler may not enforce left-to-right evaluation order for call to 'std::operator<<<char,std::char_traits<char>,std::allocator<char> >'
#endif

namespace MR
{

namespace PointsSave
{
const IOFilters Filters =
{
    {"ASCII (.asc)",      "*.asc"},
    {"PLY (.ply)",        "*.ply"},
#ifndef MRMESH_NO_OPENCTM
    {"CTM (.ctm)",        "*.ctm"},
#endif
};

VoidOrErrStr toAsc( const PointCloud& points, const std::filesystem::path& file, const Settings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsc( points, out, settings );
}

VoidOrErrStr toAsc( const PointCloud& cloud, std::ostream& out, const Settings& settings )
{
    MR_TIMER
    const bool saveNormals = cloud.points.size() <= cloud.normals.size();
    const size_t totalPoints = settings.saveValidOnly ? cloud.validPoints.count() : cloud.points.size();
    size_t numSaved = 0;
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.saveValidOnly && !cloud.validPoints.test( v ) )
            continue;
        const auto & p = cloud.points[v];
        out << fmt::format( "{} {} {}", p.x, p.y, p.z );
        if ( saveNormals )
        {
            const auto & n = cloud.normals[v];
            out << fmt::format( " {} {} {}", n.x, n.y, n.z );
        }
        out << '\n';
        ++numSaved;
        if ( settings.callback && !( numSaved & 0x3FF ) && !settings.callback( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in ASC-format" ) );

    reportProgress( settings.callback, 1.f );
    return {};
}

VoidOrErrStr toPly( const PointCloud& points, const std::filesystem::path& file, const Settings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, settings );
}

VoidOrErrStr toPly( const PointCloud& cloud, std::ostream& out, const Settings& settings )
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

    size_t numSaved = 0;
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.saveValidOnly && !cloud.validPoints.test( v ) )
            continue;
        out.write( ( const char* )&cloud.points[v], 12 );
        if ( saveNormals )
            out.write( ( const char* )&cloud.normals[v], 12 );
        if ( settings.colors )
        {
            const auto c = ( *settings.colors )[v];
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        ++numSaved;
        if ( settings.callback && !( numSaved & 0x3FF ) && !settings.callback( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.callback, 1.f );
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

    std::vector<CTMuint> aIndices{ 0,0,0 };
    std::vector<Vector3f> validPoints, validNormals;
    if ( options.saveValidOnly )
    {
        validPoints.reserve( aVertexCount );
        for ( auto v : cloud.validPoints )
            validPoints.push_back( cloud.points[v] );
        if ( saveNormals )
        {
            validNormals.reserve( aVertexCount );
            for ( auto v : cloud.validPoints )
                validNormals.push_back( cloud.normals[v] );
        }
        ctmDefineMesh( context,
            ( const CTMfloat* )validPoints.data(), aVertexCount,
            aIndices.data(), 1, saveNormals ? ( const CTMfloat* )validNormals.data() : nullptr );
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
    if ( options.callback )
    {
        saveData.callbackFn = [callback = options.callback, &saveData] ( float progress )
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

    reportProgress( options.callback, 1.f );
    return {};
}
#endif

VoidOrErrStr toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const Settings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".asc" )
        res = MR::PointsSave::toAsc( points, file, settings );
    if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, file, settings );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, file, { settings } );
#endif
    return res;
}
VoidOrErrStr toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const Settings& settings )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".asc" )
        res = MR::PointsSave::toAsc( points, out, settings );
    else if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, out, settings );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, out, { settings } );
#endif
    return res;
}

} // namespace PointsSave

} // namespace MR
