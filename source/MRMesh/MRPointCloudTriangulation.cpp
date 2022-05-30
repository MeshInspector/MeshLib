#include "MRPointCloudTriangulation.h"
#include "MRPointCloud.h"
#include "MRVector.h"
#include "MRId.h"
#include "MRPointCloudRadius.h"
#include "MRPointCloudMakeNormals.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshDelone.h"
#include "MRMeshBuilder.h"
#include "MRMeshFillHole.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRVector3.h"
#include "MRTimer.h"
#include "MRPointCloudTriangulationHelpers.h"
#include <parallel_hashmap/phmap.h>

namespace MR
{

class PointCloudTriangulator
{
public:
    PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params );

    std::optional<Mesh> triangulate( ProgressCallback progressCb );

private:
    // parallel creates local triangulated fans for each point
    bool optimizeAll_( ProgressCallback progressCb );
    // accumulate local funs to surface
    std::optional<Mesh> triangulate_( ProgressCallback progressCb );

    const PointCloud& pointCloud_;
    TriangulationParameters params_;

    Vector<TriangulationHelpers::TriangulatedFan, VertId> optimizedFans_;
};

PointCloudTriangulator::PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params ) :
    pointCloud_{pointCloud},
    params_{params}
{
}

std::optional<Mesh> PointCloudTriangulator::triangulate( ProgressCallback progressCb )
{
    MR_TIMER;
    if ( !optimizeAll_( progressCb ) )
        return {};
    return triangulate_( progressCb );
}

bool PointCloudTriangulator::optimizeAll_( ProgressCallback progressCb )
{
    MR_TIMER;
    float radius = findAvgPointsRadius( pointCloud_, params_.avgNumNeighbours );
    // const ref should prolong makeNormals lifetime
    const VertCoords& normals = pointCloud_.normals.empty() ? makeNormals( pointCloud_, params_.avgNumNeighbours ) : pointCloud_.normals;

    optimizedFans_.resize( pointCloud_.points.size() );

    auto body = [&] ( VertId v )
    {
        auto candidates = TriangulationHelpers::findNeighbors( pointCloud_, v, radius );
        auto optimizedRes = TriangulationHelpers::trianglulateFan( pointCloud_.points, v, candidates, normals, params_.critAngle );
        const auto& optimized = optimizedRes.optimized;

        float maxRadius = ( candidates.size() < 2 ) ? radius * 2.0f :
            TriangulationHelpers::updateNeighborsRadius( pointCloud_.points, v, optimized, radius );

        if ( maxRadius > radius )
        {
            // update triangulation if radius was increased
            candidates = TriangulationHelpers::findNeighbors( pointCloud_, v, radius );
            optimizedRes = TriangulationHelpers::trianglulateFan( pointCloud_.points, v, candidates, normals, params_.critAngle );
        }
        optimizedFans_[v] = optimizedRes;
    };

    ProgressCallback partialProgressCb;
    if ( progressCb )
    {
        // 0% - 35%
        partialProgressCb = [&] ( float p )
        {
            return progressCb( 0.35f * p );
        };
    }
    return BitSetParallelFor( pointCloud_.validPoints, body, partialProgressCb );
}

struct VertTriplet
{
    VertTriplet( VertId _a, VertId _b, VertId _c ) :
        a{_a}, b{_b}, c{_c}
    {
        if ( b < a && b < c )
        {
            std::swap( a, b );
            std::swap( b, c );
        }
        else if ( c < a && c < b )
        {
            std::swap( a, c );
            std::swap( b, c );
        }
    }
    VertId a, b, c;
};

bool operator==( const VertTriplet& a, const VertTriplet& b )
{
    return( a.a == b.a && a.b == b.b && a.c == b.c );
}

struct VertTripletHasher
{
    size_t operator()( const VertTriplet& triplet ) const
    {
        auto h1 = std::hash<int>{}(int( triplet.a ));
        auto h2 = std::hash<int>{}(int( triplet.b ));
        auto h3 = std::hash<int>{}(int( triplet.c ));
        return h1 ^ (h2 << 1) ^ (h3 << 3);
    }
};

std::optional<Mesh> PointCloudTriangulator::triangulate_( ProgressCallback progressCb )
{
    MR_TIMER;
    // accumulate triplets
    phmap::flat_hash_map<VertTriplet, int, VertTripletHasher> map;
    for ( auto cV : pointCloud_.validPoints )
    {
        const auto& disc = optimizedFans_[cV];
        for ( auto it = disc.optimized.begin(); it != disc.optimized.end(); ++it )
        {
            if ( disc.border.valid() && *it == disc.border )
                continue;

            auto next = std::next( it );
            if ( next == disc.optimized.end() )
                next = disc.optimized.begin();

            VertTriplet triplet{ cV,*next,*it };
            auto mIt = map.find( triplet );
            if ( mIt == map.end() )
                map[triplet] = 1;
            else
                ++mIt->second;
        }
        if ( progressCb )
        {
            if ( !progressCb( 0.35f + 0.30f * float( cV ) / float( pointCloud_.validPoints.size() ) ) ) // 35% - 65%
                return {};
        }
    }
    Mesh mesh;
    mesh.points = pointCloud_.points;

    std::vector<MeshBuilder::Triangle> tris3;
    std::vector<MeshBuilder::Triangle> tris2;
    int faceCounter = 0;
    for ( const auto& triplet : map )
    {
        if ( triplet.second == 3 )
            tris3.emplace_back( triplet.first.a, triplet.first.b, triplet.first.c, FaceId( faceCounter++ ) );
        else if ( triplet.second == 2 )
            tris2.emplace_back( triplet.first.a, triplet.first.b, triplet.first.c, FaceId( faceCounter++ ) );
    }
    // create topology
    MeshBuilder::addTriangles( mesh.topology, tris3, false );
    tris2.insert( tris2.end(), tris3.begin(), tris3.end() );
    if ( progressCb )
        if ( !progressCb( 0.67f ) ) // 67%
            return {};
    MeshBuilder::addTriangles( mesh.topology, tris2, false );
    if ( progressCb )
        if ( !progressCb( 0.70f ) ) // 70%
            return {};

    // fill small holes
    const auto bigLength = params_.critHoleLength >= 0.0f ? params_.critHoleLength : pointCloud_.getBoundingBox().diagonal() * 0.7f;
    auto boundaries = mesh.topology.findBoundary();
    for ( int i = 0; i < boundaries.size(); ++i )
    {
        const auto& boundary = boundaries[i];
        float length = 0.0f;
        for ( auto e : boundary )
            length += mesh.edgeLength( e );

        if ( length < bigLength )
            fillHole( mesh, boundary.front() );
        if ( progressCb )
            if ( !progressCb( 0.7f + 0.3f * float( i + 1 ) / float( boundaries.size() ) ) ) // 70% - 100%
                return {};
    }

    return mesh;
}

std::optional<Mesh> triangulatePointCloud( const PointCloud& pointCloud, const TriangulationParameters& params /*= {} */,
    ProgressCallback progressCb )
{
    MR_TIMER
    PointCloudTriangulator triangulator( pointCloud, params );
    return triangulator.triangulate( progressCb );
}

} //namespace MR
