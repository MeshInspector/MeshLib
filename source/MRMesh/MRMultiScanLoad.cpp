#include "MRMultiScanLoad.h"
#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRAffineXf3.h"
#include "MRDirectory.h"
#include "MRStringConvert.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRProgressCallback.h"
#include "MRTimer.h"

#include <fstream>
#include <map>
#include <charconv>
#include <optional>
#include <string_view>

namespace MR::PointsLoad
{

namespace
{

// file name prefixes used by the multi-scan laser export (e.g. _intemp000007.pose and _laser000007.ply)
constexpr std::string_view cScanPosePrefix = "_intemp";
constexpr std::string_view cScanPlyPrefix = "_laser";

// parses the integer index from a scan file stem with the given prefix, e.g. ("_intemp000007", "_intemp") -> 7;
// returns std::nullopt if the stem does not start with the prefix or the remainder is not a pure number
std::optional<int> parseScanIndex( const std::string& stem, std::string_view prefix )
{
    if ( !std::string_view( stem ).starts_with( prefix ) )
        return {};
    const char* first = stem.data() + prefix.size();
    const char* last = stem.data() + stem.size();
    int index = 0;
    const auto [ptr, ec] = std::from_chars( first, last, index );
    if ( ec != std::errc{} || ptr != last )
        return {};
    return index;
}

// reads a 4x4 row-major rigid transformation from a .pose file as AffineXf3f (the last row 0 0 0 1 is ignored)
Expected<AffineXf3f> readScanPose( const std::filesystem::path& file )
{
    std::ifstream in( file );
    if ( !in )
        return unexpected( "Cannot open file for reading " + utf8string( file ) );

    AffineXf3f xf;
    in  >> xf.A.x.x >> xf.A.x.y >> xf.A.x.z >> xf.b.x
        >> xf.A.y.x >> xf.A.y.y >> xf.A.y.z >> xf.b.y
        >> xf.A.z.x >> xf.A.z.y >> xf.A.z.z >> xf.b.z;
    if ( in.fail() )
        return unexpected( "Cannot parse transformation from " + utf8string( file ) );
    return xf;
}

} // anonymous namespace

Expected<PointCloud> fromMultiScanFolder( const std::filesystem::path& folder, const ProgressCallback& callback )
{
    MR_TIMER;

    std::error_code ec;
    if ( !std::filesystem::is_directory( folder, ec ) )
        return unexpected( utf8string( folder ) + " is not a folder" );

    // collect .pose and .ply files indexed by the number embedded in their names
    std::map<int, std::filesystem::path> poseFiles, plyFiles;
    for ( auto entry : Directory{ folder, ec } )
    {
        if ( !entry.is_regular_file( ec ) )
            continue;
        const auto path = entry.path();
        const auto stem = utf8string( path.stem() );
        auto ext = utf8string( path.extension() );
        for ( auto& c : ext )
            c = ( char )tolower( c );

        if ( ext == ".pose" )
        {
            if ( auto idx = parseScanIndex( stem, cScanPosePrefix ) )
                poseFiles[*idx] = path;
        }
        else if ( ext == ".ply" )
        {
            if ( auto idx = parseScanIndex( stem, cScanPlyPrefix ) )
                plyFiles[*idx] = path;
        }
    }

    // keep only the indices having both a .pose and a .ply file
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> pairs; // ( .pose, .ply )
    for ( const auto& [idx, posePath] : poseFiles )
    {
        if ( auto it = plyFiles.find( idx ); it != plyFiles.end() )
            pairs.emplace_back( posePath, it->second );
    }
    if ( pairs.empty() )
        return unexpected( "No pairs of " + std::string( cScanPosePrefix ) + "*.pose and "
            + std::string( cScanPlyPrefix ) + "*.ply files found in " + utf8string( folder ) );

    const int cReportEverySingle = 1;
    const float cProgressReadXfs = 0.1f;
    const float cProgressReadScans = 0.9f;

    std::vector<AffineXf3f> scansXf( pairs.size() );
    if ( !ParallelFor( scansXf, [&]( size_t i )
        {
            auto xf = readScanPose( pairs[i].first );
            if ( xf )
                scansXf[i] = *xf;
            //if ( !xf )
            //    return unexpected( std::move( xf.error() ) );
        }, subprogress( callback, 0.0f, cProgressReadXfs ), cReportEverySingle ) )
        return unexpectedOperationCanceled();

    std::vector<PointCloud> scans( pairs.size() );
    if ( !ParallelFor( scans, [&]( size_t i )
        {
            const auto& [posePath, plyPath] = pairs[i];

            auto cloud = fromPly( pairs[i].second );
            //if ( !cloud )
            //    return unexpected( std::move( cloud.error() ) );

            // transform the loaded points (and normals) into the common coordinate frame
            const auto xf = scansXf[i];
            BitSetParallelFor( cloud->validPoints, [&] ( VertId v )
            {
                cloud->points[v] = xf( cloud->points[v] );
                if ( v < cloud->normals.size() )
                    cloud->normals[v] = xf.A * cloud->normals[v];
            } );

            scans[i] = std::move( *cloud );
        }, subprogress( callback, cProgressReadXfs, cProgressReadScans ), cReportEverySingle ) )
        return unexpectedOperationCanceled();

    std::vector<VertId> firstScanPoint;
    firstScanPoint.reserve( pairs.size() + 1 );
    firstScanPoint.push_back( 0_v );
    for ( const auto & s : scans )
        firstScanPoint.push_back( firstScanPoint.back() + s.points.size() );
    assert( firstScanPoint.size() == pairs.size() + 1 );
    const int totalPoints( firstScanPoint.back() );

    PointCloud res;
    res.points.resizeNoInit( totalPoints );
    res.normals.resizeNoInit( totalPoints );
    res.validPoints.resize( totalPoints, true );
    if ( !ParallelFor( scans, [&]( size_t i )
        {
            const auto& scan = scans[i];
            const auto f = firstScanPoint[i];
            for ( auto v = 0_v; v < scan.points.size(); ++v )
            {
                auto t = f + (int)v;
                res.points[t] = scan.points[v];
                res.normals[t] = scan.normals[v];
            }
        }, subprogress( callback, cProgressReadScans, 1.0f ) ) )
        return unexpectedOperationCanceled();

    return res;
}

} // namespace MR::PointsLoad
