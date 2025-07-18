#include "MRLinesSave.h"
#include "MRIOFormatsRegistry.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPch/MRFmt.h"
#include <fstream>

namespace MR
{

namespace LinesSave
{

Expected<void> toMrLines( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrLines( polyline, out, settings );
}

Expected<void> toMrLines( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    MR_TIMER;
    polyline.topology.write( out );

    // write points
    const std::uint32_t type = 3; //3d points
    out.write( (const char*)&type, 4 );
    auto numPoints = (std::uint32_t)( polyline.topology.lastValidVert() + 1 );
    out.write( ( const char* )&numPoints, 4 );

    VertCoords buf;
    const auto & xfVerts = transformPoints( polyline.points, polyline.topology.getValidVerts(), settings.xf, buf );
    if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numPoints * sizeof( Vector3f ), settings.progress ) )
        return unexpectedOperationCanceled();

    if ( !out )
        return unexpected( std::string( "Error saving in MrLines-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toPts( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( polyline, out, settings );
}

Expected<void> toPts( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    float pointsNum{ 0.f };
    auto contours = polyline.contours();
    for ( const auto& contour : contours )
        pointsNum += contour.size();

    int pointIndex = 0;
    for ( const auto& contour : contours )
    {
        out << "BEGIN_Polyline\n";
        for ( auto v : contour )
        {
            auto saveVertex = [&]( auto && p )
            {
                out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
            };
            if ( settings.xf )
                saveVertex( applyDouble( settings.xf, v ) );
            else
                saveVertex( v );
            ++pointIndex;
            if ( settings.progress && !( pointIndex & 0x3FF ) && !settings.progress( float( pointIndex ) / pointsNum ) )
                return unexpectedOperationCanceled();
        }
        out << "END_Polyline\n";
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PTS-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toDxf( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toDxf( polyline, out, settings );
}

Expected<void> toDxf( const Polyline3& polyline, std::ostream& out, const SaveSettings & settings )
{
    out << "0\nSECTION\n";
    out << "2\nENTITIES\n";

    float pointsNum{ 0.f };
    auto contours = polyline.contours();
    for ( const auto& contour : contours )
        pointsNum += contour.size();

    int pointIndex = 0;
    for ( const auto& contour : contours )
    {
        out << "0\nPOLYLINE\n";
        out << "8\n0\n";
        out << "66\n1\n";
        int flags = 8;
        if ( contour[0] == contour.back() )
            flags += 1;
        out << "70\n" << flags << "\n";
        for ( auto v : contour )
        {
            auto saveVertex = [&]( auto && p )
            {
                out << fmt::format(
                    "0\nVERTEX\n"
                    "8\n0\n"
                    "70\n32\n"
                    "10\n{}\n"
                    "20\n{}\n"
                    "30\n{}\n",
                    p.x, p.y, p.z );
            };
            if ( settings.xf )
                saveVertex( applyDouble( settings.xf, v ) );
            else
                saveVertex( v );
            ++pointIndex;
            if ( settings.progress && !( pointIndex & 0x3FF ) && !settings.progress( float( pointIndex ) / pointsNum ) )
                return unexpectedOperationCanceled();
        }
        out << "0\nSEQEND\n";
    }

    out << "0\nENDSEC\n";
    out << "0\nEOF\n";

    if ( !out )
        return unexpected( std::string( "Error saving in DXF-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toPly( const Polyline3& polyline, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( polyline, out, settings );
}

Expected<void> toPly( const Polyline3& polyline, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    const VertRenumber vertRenumber( polyline.topology.getValidVerts(), settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = polyline.topology.lastValidVert();
    const bool saveColors = settings.colors && settings.colors->size() > lastVertId;

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numPoints << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    const auto ueLast = polyline.topology.lastNotLoneUndirectedEdge();
    const auto numSaveEdges = settings.packPrimitives ? polyline.topology.computeNotLoneUndirectedEdges() : int( ueLast + 1 );
    out <<  "element edge " << numSaveEdges << "\nproperty int vertex1\nproperty int vertex2\nend_header\n";

    static_assert( sizeof( Vector3f ) == 12, "wrong size of Vector3f" );
#pragma pack(push, 1)
    struct PlyColor
    {
        unsigned char r = 0, g = 0, b = 0;
    };
#pragma pack(pop)
    static_assert( sizeof( PlyColor ) == 3, "check your padding" );

    // write vertices
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !polyline.topology.hasVert( i ) )
            continue;
        const Vector3f p = applyFloat( settings.xf, polyline.points[i] );
        out.write( ( const char* )&p, 12 );
        if ( settings.colors )
        {
            const auto c = ( *settings.colors )[i];
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }

    // write edges
    #pragma pack(push, 1)
    struct PlyEdge
    {
        int v1 = 0, v2 = 0;
    };
    #pragma pack(pop)
    static_assert( sizeof( PlyEdge ) == 8, "check your padding" );

    PlyEdge edge;
    int savedEdges = 0;
    for ( auto ue = 0_ue; ue <= ueLast; ++ue )
    {
        if ( !polyline.topology.isLoneEdge( ue ) )
        {
            edge.v1 = polyline.topology.org( ue );
            edge.v2 = polyline.topology.dest( ue );
        }
        else if ( !settings.packPrimitives )
            edge = {};
        else
            continue;
        out.write( (const char *)&edge, sizeof( PlyEdge ) );
        ++savedEdges;
        if ( settings.progress && !( savedEdges & 0x3FF ) && !settings.progress( float( savedEdges ) / numSaveEdges * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getLinesSaver( ext );
    if ( !saver.fileSave )
        return unexpectedUnsupportedFileExtension();

    return saver.fileSave( polyline, file, settings );
}

Expected<void> toAnySupportedFormat( const Polyline3& polyline, const std::string& extension, std::ostream& out, const SaveSettings & settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getLinesSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    return saver.streamSave( polyline, out, settings );
}

MR_ADD_LINES_SAVER_WITH_PRIORITY( IOFilter( "MrLines (.mrlines)", "*.mrlines" ), toMrLines, -1 )
MR_ADD_LINES_SAVER( IOFilter( "PTS (.pts)", "*.pts" ), toPts )
MR_ADD_LINES_SAVER( IOFilter( "PLY (.ply)", "*.ply" ), toPly )
MR_ADD_LINES_SAVER( IOFilter( "Drawing exchange format (.dxf)", "*.dxf" ), toDxf )

} //namespace LinesSave

} //namespace MR
