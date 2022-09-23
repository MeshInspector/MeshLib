#include "MRPointsSave.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include <fstream>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

namespace MR
{

namespace PointsSave
{
const IOFilters Filters =
{
    {"PLY (.ply)",        "*.ply"},
#ifndef MRMESH_NO_OPENCTM
    {"CTM (.ctm)",        "*.ctm"},
#endif
};

tl::expected<void, std::string> toPly( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr*/, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, colors, callback );
}

tl::expected<void, std::string> toPly( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors /*= nullptr*/, ProgressCallback callback )
{
    MR_TIMER;

    size_t numVertices = points.points.size();

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( colors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";

    if ( !colors )
    {
        // write vertices
        static_assert( sizeof( points.points.front() ) == 12, "wrong size of Vector3f" );
        const bool cancel = !MR::writeByBlocks( out, (const char*) points.points.data(), points.points.size() * sizeof( Vector3f ), callback );
        if ( cancel )
            return tl::make_unexpected( std::string( "Saving canceled" ) );
    }
    else
    {
        // write triangles
#pragma pack(push, 1)
        struct PlyColoredVert
        {
            Vector3f p;
            unsigned char r = 0, g = 0, b = 0;
        };
#pragma pack(pop)
        static_assert( sizeof( PlyColoredVert ) == 15, "check your padding" );

        PlyColoredVert cVert;
        for ( int v = 0; v < numVertices; ++v )
        {
            cVert.p = points.points[VertId( v )];
            const auto& c = ( *colors )[VertId( v )];
            cVert.r = c.r; cVert.g = c.g; cVert.b = c.b;
            out.write( ( const char* )&cVert, 15 );
            if ( callback && !( v & 0x3FF ) && !callback( float( v ) / numVertices ) )
                return tl::make_unexpected( std::string( "Saving canceled" ) );
        }
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PLY-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

#ifndef MRMESH_NO_OPENCTM
tl::expected<void, std::string> toCtm( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( points, out, colors, options, callback );
}

tl::expected<void, std::string> toCtm( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/, ProgressCallback callback )
{
    MR_TIMER;

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

    const CTMfloat* normalsPtr = points.normals.empty() ? nullptr : ( const CTMfloat* )points.normals.data();
    CTMuint aVertexCount = CTMuint( points.points.size() );

    std::vector<CTMuint> aIndices{ 0,0,0 };

    ctmDefineMesh( context,
        ( const CTMfloat* )points.points.data(), aVertexCount,
        aIndices.data(), 1, normalsPtr );

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( colors && colors->size() == points.points.size() )
    {
        colors4f.resize( colors->size() );
        for ( int i = 0; i < colors4f.size(); ++i )
            colors4f[i] = Vector4f( ( *colors )[VertId{ i }] );

        ctmAddAttribMap( context, ( const CTMfloat* )colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format colors" );

    struct SaveData
    {
        std::function<bool( float )> callbackFn{};
        std::ostream* stream;
        size_t sum{ 0 };
        size_t blockSize{ 0 };
        size_t maxSize{ 0 };
        bool wasCanceled{ false };
    } saveData;
    if ( callback )
    {
        saveData.callbackFn = [callback, &saveData] ( float progress )
        {
            // calculate full progress in partical-linear scale (we don't know compressed size and it less than real size)
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
    saveData.maxSize = points.points.size() * sizeof( Vector3f ) + points.normals.size() * sizeof( Vector3f ) + 150; // 150 - reserve for some ctm specific data
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
        return tl::make_unexpected( std::string( "Saving canceled" ) );
    if ( !out || ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( std::string( "Error saving in CTM-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}
#endif

tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr */,
                                                      ProgressCallback callback )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, file, colors, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, file, colors, {}, callback );
#endif
    return res;
}
tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const Vector<Color, VertId>* colors /*= nullptr */,
                                                      ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, out, colors, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, out, colors, {}, callback );
#endif
    return res;
}

}
}
