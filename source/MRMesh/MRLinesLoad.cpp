#include "MRLinesLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOParsing.h"
#include "MRPolyline.h"
#include "MRMatrix3.h"
#include "MRVector2.h"
#include "MRConstants.h"
#include "MRString.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRPly.h"
#include "MRTimer.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace MR
{

namespace LinesLoad
{

Expected<Polyline3> fromMrLines( const std::filesystem::path & file, const LinesLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromMrLines( in, settings ), file );
}

Expected<Polyline3> fromMrLines( std::istream & in, const LinesLoadSettings& settings )
{
    MR_TIMER;

    Polyline3 polyline;
    if ( !polyline.topology.read( in ) )
        return unexpected( std::string( "Error reading topology from lines-file" ) );

    // read points
    std::uint32_t type = 0;
    in.read( (char*)&type, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the type of points from lines-file" ) );
    if ( type != 3 )
        return unexpected( std::string( "Unsupported point type in lines-file" ) );
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of points from lines-file" ) );
    polyline.points.resize( numPoints );
    readByBlocks( in, (char*)polyline.points.data(), polyline.points.size() * sizeof(Vector3f), settings.callback );
    if ( !in )
        return unexpected( std::string( "Error reading  points from lines-file" ) );

    return polyline;
}

Expected<MR::Polyline3> fromPts( const std::filesystem::path& file, const LinesLoadSettings& settings /*= {} */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPts( in, settings ), file );
}

Expected<MR::Polyline3> fromPts( std::istream& in, const LinesLoadSettings& settings /*= {} */ )
{
    std::string line;
    int pointCount = 0;

    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    bool isPolylineBlock{ false };

    Polyline3 polyline;
    std::vector<Vector3f> points;
    while ( std::getline( in, line ) )
    {
        line.erase( std::find_if( line.rbegin(), line.rend(), [] ( unsigned char ch )
        {
            return !std::isspace( ch );
        } ).base(), line.end() );
        if ( !isPolylineBlock )
        {
            if ( line != "BEGIN_Polyline" )
                return unexpected( stringUnsupportedFileFormat() + " for .PTS file with polylines" );
            else
            {
                isPolylineBlock = true;
                continue;
            }
        }
        else if ( line == "END_Polyline" )
        {
            isPolylineBlock = false;
            if ( points.empty() )
                continue;

            polyline.addFromPoints( points.data(), points.size() );
            points.clear();

            continue;
        }

        std::istringstream iss( line );
        Vector3f point;
        if ( !( iss >> point ) )
            return unexpected( stringUnsupportedFileFormat() + " for .PTS file with polylines" );
        points.push_back( point );
        ++pointCount;

        if ( settings.callback && !( pointCount & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / float( streamSize );
            if ( !settings.callback( progress ) )
                return unexpectedOperationCanceled();
        }
    }
    if ( isPolylineBlock )
        return unexpected( stringUnsupportedFileFormat() + " for .PTS file with polylines" );

    return polyline;
}

Expected<Polyline3> fromPly( const std::filesystem::path& file, const LinesLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, settings ), file );
}

Expected<Polyline3> fromPly( std::istream& in, const LinesLoadSettings& settings )
{
    MR_TIMER;

    std::optional<Edges> edges;
    PlyLoadParams params =
    {
        .edges = &edges,
        .colors = settings.colors,
        // suppose that reading is 10% of progress and building polyline is 90% of progress
        .callback = subprogress( settings.callback, 0.0f, 0.1f ),
        .telemetrySignal = settings.telemetrySignal
    };
    auto maybePoints = loadPly( in, params );
    if ( !maybePoints )
        return unexpected( std::move( maybePoints.error() ) );

    Polyline3 res;
    res.points = std::move( *maybePoints );

    if ( edges )
    {
        res.topology.vertResize( res.points.size() );
        res.topology.makeEdges( *edges );
    }

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

namespace
{

/// rotation from Object Coordinate System of a DXF entity with given extrusion direction to the world (Arbitrary Axis Algorithm)
Matrix3f dxfOcsToWorld( const Vector3f& extrusion )
{
    if ( extrusion.lengthSq() <= 0 )
        return {};
    const auto az = extrusion.normalized();
    constexpr float cLimit = 1.0f / 64;
    const auto ax = ( std::abs( az.x ) < cLimit && std::abs( az.y ) < cLimit ? cross( Vector3f::plusY(), az ) : cross( Vector3f::plusZ(), az ) ).normalized();
    return Matrix3f::fromColumns( ax, cross( az, ax ).normalized(), az );
}

/// appends the points of an arc in the plane z = center.z, starting after the angle a0 and spanning the signed angle sweep (radians)
void appendDxfArc( Contour3f& c, const Vector3f& center, float radius, float a0, float sweep )
{
    constexpr float cMaxStep = PI_F / 36; // 5 degrees
    const int n = std::max( 1, int( std::ceil( std::abs( sweep ) / cMaxStep ) ) );
    for ( int i = 1; i <= n; ++i )
    {
        const float a = a0 + sweep * float( i ) / float( n );
        c.push_back( center + radius * Vector3f( std::cos( a ), std::sin( a ), 0 ) );
    }
}

/// appends the arc from p0 (already in the contour) to p1, where bulge is the tangent of 1/4 of the arc's angle
void appendDxfBulge( Contour3f& c, const Vector3f& p0, const Vector3f& p1, float bulge )
{
    const Vector2f a( p0.x, p0.y ), b( p1.x, p1.y );
    const auto d = b - a;
    const float len = d.length();
    if ( bulge != 0 && len > 0 )
    {
        const float angle = 4 * std::atan( bulge );
        // the center is to the left of the chord for a counter-clockwise arc
        const auto center = 0.5f * ( a + b ) + Vector2f( -d.y, d.x ) * ( 0.5f / std::tan( angle / 2 ) );
        appendDxfArc( c, Vector3f( center.x, center.y, p0.z ), ( a - center ).length(), std::atan2( a.y - center.y, a.x - center.x ), angle );
        c.back() = p1;
    }
    else
        c.push_back( p1 );
}

/// one entity of DXF file with the values of its group codes used by lines
struct DxfEntity
{
    std::string type;
    Contour3f pts;              ///< 10,20,30 (and 11,21,31 of LINE)
    std::vector<float> bulges;  ///< 42, one per point
    int flags = 0;              ///< 70
    float elevation = 0;        ///< 38
    float radius = 0;           ///< 40
    float startAngle = 0;       ///< 50, degrees
    float endAngle = 360;       ///< 51, degrees
    Vector3f extrusion{ 0, 0, 1 }; ///< 210,220,230

    DxfEntity() : DxfEntity( std::string_view{} ) {}
    explicit DxfEntity( std::string_view t ) : type( t )
    {
        if ( type != "LWPOLYLINE" )
        {
            pts.resize( type == "LINE" ? 2 : 1 );
            bulges.resize( pts.size() );
        }
    }

    bool isSupported() const { return type == "LINE" || type == "LWPOLYLINE" || type == "POLYLINE" || type == "VERTEX" || type == "CIRCLE" || type == "ARC"; }

    /// returns false if the value cannot be parsed
    bool setValue( int code, std::string_view value )
    {
        if ( code == 70 )
            return parseSingleNumber<int>( value, flags ).has_value();
        float v = 0;
        auto parse = [&] { return parseSingleNumber<float>( value, v ).has_value(); };
        const bool lw = type == "LWPOLYLINE";
        switch ( code )
        {
        case 10:
            if ( !parse() )
                return false;
            if ( lw )
            {
                pts.emplace_back().x = v;
                bulges.push_back( 0 );
            }
            else
                pts[0].x = v;
            return true;
        case 20:
            if ( !parse() )
                return false;
            if ( lw && !pts.empty() )
                pts.back().y = v;
            else if ( !lw )
                pts[0].y = v;
            return true;
        case 30:
            if ( !parse() )
                return false;
            if ( !lw )
                pts[0].z = v;
            return true;
        case 11:
        case 21:
        case 31:
            if ( type != "LINE" )
                return true;
            if ( !parse() )
                return false;
            pts[1][code / 10 - 1] = v;
            return true;
        case 38:
            return parseSingleNumber<float>( value, elevation ).has_value();
        case 40:
            return parseSingleNumber<float>( value, radius ).has_value();
        case 42:
            if ( !parse() )
                return false;
            if ( !bulges.empty() )
                bulges.back() = v;
            return true;
        case 50:
            return parseSingleNumber<float>( value, startAngle ).has_value();
        case 51:
            return parseSingleNumber<float>( value, endAngle ).has_value();
        case 210:
        case 220:
        case 230:
            return parseSingleNumber<float>( value, extrusion[code / 10 - 21] ).has_value();
        default:
            return true;
        }
    }
};

} // anonymous namespace

Expected<Polyline3> fromDxf( const std::filesystem::path& file, const LinesLoadSettings& settings )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromDxf( in, settings ), file );
}

Expected<Polyline3> fromDxf( std::istream& in, const LinesLoadSettings& settings )
{
    MR_TIMER;
    const auto posStart = in.tellg();
    const auto streamSize = getStreamSize( in );

    Contours3f contours;

    /// adds the polyline given in Object Coordinate System (if ocs is not null), replacing bulges with arcs
    auto addPolyline = [&]( const Contour3f& pts, const std::vector<float>& bulges, bool closed, const Matrix3f* ocs )
    {
        auto n = pts.size();
        if ( closed && n > 2 && pts.back() == pts.front() )
            --n; // the first point is repeated in the end
        if ( n < 2 )
            return;
        Contour3f c;
        c.reserve( n + 1 );
        c.push_back( pts[0] );
        for ( size_t i = 0; i + 1 < n + size_t( closed ); ++i )
            appendDxfBulge( c, pts[i], pts[( i + 1 ) % n], ocs ? bulges[i] : 0 );
        if ( ocs )
            for ( auto& p : c )
                p = *ocs * p;
        contours.push_back( std::move( c ) );
    };

    // POLYLINE entity is followed by its VERTEX entities and SEQEND
    struct
    {
        bool active = false;
        bool skip = false;
        DxfEntity header;
        Contour3f pts;
        std::vector<float> bulges;
    } poly;

    auto finishPoly = [&]
    {
        if ( poly.active && !poly.skip )
        {
            const bool is3d = ( poly.header.flags & 8 ) != 0;
            const auto ocs = dxfOcsToWorld( poly.header.extrusion );
            addPolyline( poly.pts, poly.bulges, ( poly.header.flags & 1 ) != 0, is3d ? nullptr : &ocs );
        }
        poly = {};
    };

    auto finishEntity = [&]( const DxfEntity& e )
    {
        if ( e.type == "VERTEX" )
        {
            // 16 is a control point of spline frame
            if ( poly.active && !( e.flags & 16 ) )
            {
                poly.pts.push_back( e.pts[0] );
                poly.bulges.push_back( e.bulges[0] );
            }
            return;
        }
        finishPoly();
        if ( e.type == "LINE" )
            contours.push_back( e.pts );
        else if ( e.type == "LWPOLYLINE" )
        {
            auto pts = e.pts;
            for ( auto& p : pts )
                p.z = e.elevation;
            const auto ocs = dxfOcsToWorld( e.extrusion );
            addPolyline( pts, e.bulges, ( e.flags & 1 ) != 0, &ocs );
        }
        else if ( e.type == "POLYLINE" )
        {
            poly.active = true;
            poly.skip = ( e.flags & ( 16 | 64 ) ) != 0; // polygon mesh or polyface mesh
            poly.header = e;
        }
        else if ( e.type == "CIRCLE" || e.type == "ARC" )
        {
            const bool circle = e.type == "CIRCLE";
            const float a0 = circle ? 0 : e.startAngle * PI_F / 180;
            float sweep = circle ? 2 * PI_F : ( e.endAngle - e.startAngle ) * PI_F / 180;
            while ( sweep <= 0 )
                sweep += 2 * PI_F;
            Contour3f c;
            c.push_back( e.pts[0] + e.radius * Vector3f( std::cos( a0 ), std::sin( a0 ), 0 ) );
            appendDxfArc( c, e.pts[0], e.radius, a0, sweep );
            if ( circle )
                c.back() = c.front();
            const auto ocs = dxfOcsToWorld( e.extrusion );
            for ( auto& p : c )
                p = ocs * p;
            contours.push_back( std::move( c ) );
        }
    };

    // entities of other sections (e.g. BLOCKS) are not parts of the drawing
    bool inSection = false;
    std::string section;
    auto entitiesAllowed = [&] { return !inSection || section == "ENTITIES"; };

    DxfEntity entity;
    std::string codeStr, valueStr;
    for ( int i = 0; std::getline( in, codeStr ); ++i )
    {
        if ( i % 1024 == 0 && !reportProgress( settings.callback, float( in.tellg() - posStart ) / float( streamSize ) ) )
            return unexpectedOperationCanceled();

        std::string_view codeView = codeStr;
        if ( i == 0 && hasBom( codeView ) )
            codeView = codeView.substr( 3 );
        if ( trim( codeView ).empty() )
            continue;
        int code = 0;
        if ( !parseSingleNumber<int>( codeView, code ) || !std::getline( in, valueStr ) )
            return unexpected( "File is corrupted" );
        const auto value = trim( valueStr );

        if ( code == 0 )
        {
            if ( entitiesAllowed() && entity.isSupported() )
                finishEntity( entity );
            else
                finishPoly();
            entity = DxfEntity( value );
            if ( value == "SECTION" )
            {
                inSection = true;
                section.clear();
            }
            else if ( value == "ENDSEC" )
                inSection = false;
        }
        else if ( code == 2 && entity.type == "SECTION" )
            section = value;
        else if ( entity.isSupported() && entitiesAllowed() && !entity.setValue( code, value ) )
            return unexpected( "File is corrupted" );
    }
    if ( entitiesAllowed() && entity.isSupported() )
        finishEntity( entity );
    finishPoly();

    if ( !reportProgress( settings.callback, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( contours.empty() )
        return unexpected( stringUnsupportedFileFormat() + ": no lines are found in DXF file" ); // points can be there

    return Polyline3( contours );
}

Expected<Polyline3> fromAnySupportedFormat( const std::filesystem::path& file, const LinesLoadSettings& settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto loader = getLinesLoader( ext );
    if ( !loader.fileLoad )
        return unexpectedUnsupportedFileExtension();

    return loader.fileLoad( file, settings );
}

Expected<MR::Polyline3> fromAnySupportedFormat( std::istream& in, const std::string& extension, const LinesLoadSettings& settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto loader = getLinesLoader( ext );
    if ( !loader.streamLoad )
        return unexpected( std::string( "unsupported stream extension" ) );

    return loader.streamLoad( in, settings );
}

MR_ADD_LINES_LOADER_WITH_PRIORITY( IOFilter( "MrLines (.mrlines)", "*.mrlines" ), fromMrLines, -1 )
MR_ADD_LINES_LOADER( IOFilter( "PTS (.pts)",         "*.pts" ),     fromPts )
MR_ADD_LINES_LOADER( IOFilter( "PLY (.ply)",         "*.ply" ),     fromPly )
MR_ADD_LINES_LOADER( IOFilter( "Drawing Interchange Format (.dxf)", "*.dxf" ), fromDxf )

} //namespace LinesLoad

} // namespace MR
