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

Expected<void> toXyz( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toXyz( points, out, settings );
}

Expected<void> toXyz( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    MR_TIMER;
    const size_t totalPoints = settings.onlyValidPoints ? cloud.validPoints.count() : cloud.points.size();
    size_t numSaved = 0;

    NormalXfMatrix normXf( settings.xf );
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.onlyValidPoints && !cloud.validPoints.test( v ) )
            continue;
        auto saveVertex = [&]( auto && p )
        {
            out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, cloud.points[v] ) );
        else
            saveVertex( cloud.points[v] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Stream write error" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toXyzn( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toXyzn( points, out, settings );
}

Expected<void> toXyzn( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    MR_TIMER;
    if ( !cloud.hasNormals() )
        return unexpected( std::string( "Point cloud does not have normal data" ) );
    const size_t totalPoints = settings.onlyValidPoints ? cloud.validPoints.count() : cloud.points.size();
    size_t numSaved = 0;

    NormalXfMatrix normXf( settings.xf );
    for ( auto v = 0_v; v < cloud.points.size(); ++v )
    {
        if ( settings.onlyValidPoints && !cloud.validPoints.test( v ) )
            continue;
        auto saveVertex = [&]( auto && p, auto && n )
        {
            out << fmt::format( "{} {} {} {} {} {}\n", p.x, p.y, p.z, n.x, n.y, n.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, cloud.points[v] ), applyDouble( normXf, cloud.normals[v] ) );
        else
            saveVertex( cloud.points[v], cloud.normals[v] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / totalPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Stream write error" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toAsc( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsc( points, out, settings );
}

Expected<void> toAsc( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    if ( cloud.hasNormals() )
        return toXyzn( cloud, out, settings );
    else
        return toXyz( cloud, out, settings );
}

Expected<void> toPly( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, settings );
}

Expected<void> toPly( const PointCloud& cloud, std::ostream& out, const SaveSettings& settings )
{
    MR_TIMER;
    const size_t totalPoints = settings.onlyValidPoints ? cloud.validPoints.count() : cloud.points.size();

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
        if ( settings.onlyValidPoints && !cloud.validPoints.test( v ) )
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

Expected<void> toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const SaveSettings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getPointsSaver( ext );
    if ( !saver.fileSave )
        return unexpectedUnsupportedFileExtension();

    return saver.fileSave( points, file, settings );
}
Expected<void> toAnySupportedFormat( const PointCloud& points, const std::string& extension, std::ostream& out, const SaveSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getPointsSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    return saver.streamSave( points, out, settings );
}

MR_ADD_POINTS_SAVER( IOFilter( "XYZ (.xyz)", "*.xyz" ), toXyz )
MR_ADD_POINTS_SAVER( IOFilter( "XYZN (.xyzn)", "*.xyzn" ), toXyzn )
MR_ADD_POINTS_SAVER( IOFilter( "ASC (.asc)", "*.asc" ), toAsc )
MR_ADD_POINTS_SAVER( IOFilter( "PLY (.ply)", "*.ply" ), toPly )

} // namespace PointsSave

} // namespace MR
