#include "MRPointsSave.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "OpenCTM/openctm.h"
#include "MRStreamOperators.h"
#include <fstream>

namespace MR
{

namespace PointsSave
{
const IOFilters Filters =
{
    {"PLY (.ply)",        "*.ply"},
    {"CTM (.ctm)",        "*.ctm"},
    {"PTS (.pts)",        "*.pts"}
};

tl::expected<void, std::string> toPly( const PointCloud& points, const std::filesystem::path& file, const std::vector<Color>* colors /*= nullptr*/ )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, colors );
}

tl::expected<void, std::string> toPly( const PointCloud& points, std::ostream& out, const std::vector<Color>* colors /*= nullptr*/ )
{
    MR_TIMER;

    size_t numVertices = points.points.size();

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshRUs\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( colors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";

    if ( !colors )
    {
        // write vertices
        static_assert( sizeof( points.points.front() ) == 12, "wrong size of Vector3f" );
        out.write( (const char*) &points.points.front().x, 12 * numVertices );
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
            out.write( (const char*) &cVert, 15 );
        }
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PLY-format" ) );

    return {};
}

tl::expected<void, std::string> toCtm( const PointCloud& points, const std::filesystem::path& file, const std::vector<Color>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/ )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( points, out, colors, options );
}

tl::expected<void, std::string> toCtm( const PointCloud& points, std::ostream& out, const std::vector<Color>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/ )
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

    const CTMfloat* normalsPtr = points.normals.empty() ? nullptr : (const CTMfloat*) points.normals.data();
    CTMuint aVertexCount = CTMuint( points.points.size() );

    std::vector<CTMuint> aIndices{0,0,0};

    ctmDefineMesh( context,
        (const CTMfloat*) points.points.data(), aVertexCount,
        aIndices.data(), 1, normalsPtr );

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( colors && colors->size() == points.points.size() )
    {
        colors4f.resize( colors->size() );
        for ( int i = 0; i < colors4f.size(); ++i )
            colors4f[i] = Vector4f( ( *colors )[i] );

        ctmAddAttribMap( context, (const CTMfloat*) colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format colors" );

    ctmSaveCustom( context, []( const void* buf, CTMuint size, void* data )
    {
        std::ostream& s = *reinterpret_cast<std::ostream*>( data );
        s.write( (const char*) buf, size );
        return s.good() ? size : 0;
    }, &out );

    if ( !out || ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( std::string( "Error saving in CTM-format" ) );

    return {};

}

tl::expected<void, std::string> toPts( const PointCloud& points, const std::filesystem::path& file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( points, out );
}

tl::expected<void, std::string> toPts( const PointCloud& points, std::ostream& out )
{
    out << "BEGIN_Polyline\n";
    for ( auto v : points.validPoints )
        out << points.points[v] << "\n";
    out << "END_Polyline\n";
    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const std::vector<Color>* colors /*= nullptr */ )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".ply" )
        res = MR::PointsSave::toPly( points, file, colors );
    else if ( ext == u8".ctm" )
        res = MR::PointsSave::toCtm( points, file, colors );
    else if ( ext == u8".pts" )
        res = MR::PointsSave::toPts( points, file );
    return res;
}
tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const std::vector<Color>* colors /*= nullptr */ )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, out, colors );
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, out, colors );
    else if ( ext == ".pts" )
        res = MR::PointsSave::toPts( points, out );
    return res;
}

}
}
