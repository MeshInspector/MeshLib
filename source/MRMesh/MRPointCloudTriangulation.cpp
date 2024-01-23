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
#include "MRBox.h"
#include "MRTimer.h"
#include "MRPointCloudTriangulationHelpers.h"
#include "MRRegionBoundary.h"
#include "MRParallelFor.h"
#include <parallel_hashmap/phmap.h>

namespace MR
{

struct VertTriplet
{
    VertTriplet( VertId _a, VertId _b, VertId _c ) :
        a{ _a }, b{ _b }, c{ _c }
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
        return 
            2 * size_t( triplet.a ) +
            3 * size_t( triplet.b ) +
            5 * size_t( triplet.c );
    }
};

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
    std::vector<TriangulationHelpers::LocalTriangulations> localTriangulations_;
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
    MR_TIMER
    float radius = findAvgPointsRadius( pointCloud_, params_.avgNumNeighbours );
    float startProgress = 0.0f;

    VertNormals myNormals;
    if ( pointCloud_.normals.empty() )
    {
        auto optNormals = makeOrientedNormals( pointCloud_, radius, subprogress( progressCb, startProgress, 0.3f ) );
        if ( !optNormals )
            return false;
        if ( progressCb )
            startProgress = 0.3f;
        myNormals = std::move( *optNormals );
    }
    const VertCoords& normals = pointCloud_.normals.empty() ? myNormals : pointCloud_.normals;

    auto optLocalTriangulations = TriangulationHelpers::buildLocalTriangulations( pointCloud_, normals,
        { .radius = radius, .critAngle = params_.critAngle }, subprogress( progressCb, startProgress, 0.5f ) );
    if ( !optLocalTriangulations )
        return false;
    localTriangulations_ = std::move( *optLocalTriangulations );
    return true;
}

std::optional<Mesh> PointCloudTriangulator::triangulate_( ProgressCallback progressCb )
{
    MR_TIMER

    // accumulate triplets
    ParallelHashMap<VertTriplet, int, VertTripletHasher> map;
    if ( !ParallelFor( size_t(0), map.subcnt(), [&]( size_t myPartId )
    {
        for ( const auto& threadInfo : localTriangulations_ )
        {
            for ( int i = 0; i + 1 < threadInfo.fanRecords.size(); ++i )
            {
                const auto v = threadInfo.fanRecords[i].center;
                const auto border = threadInfo.fanRecords[i].border;
                const auto nbeg = threadInfo.fanRecords[i].firstNei;
                const auto nend = threadInfo.fanRecords[i+1].firstNei;
                for ( auto n = nbeg; n < nend; ++n )
                {
                    if ( threadInfo.neighbors[n] == border )
                        continue;
                    const auto next = threadInfo.neighbors[n + 1 < nend ? n + 1 : nbeg];
                    const VertTriplet triplet{ v, next, threadInfo.neighbors[n] };
                    const auto hashval = map.hash( triplet );
                    const auto idx = map.subidx( hashval );
                    if ( idx != myPartId )
                        continue;
                    auto [it, inserted] = map.insert( { triplet, 1 } );
                    if ( !inserted )
                        ++it->second;
                }
            }
        }
    }, subprogress( progressCb, 0.5f, 0.6f ), 1 ) )
        return {};

    Mesh mesh;
    mesh.points = pointCloud_.points;

    Triangulation t3;
    Triangulation t2;
    for ( const auto& triplet : map )
    {
        if ( triplet.second == 3 )
            t3.push_back( { triplet.first.a, triplet.first.b, triplet.first.c, } );
        else if ( triplet.second == 2 )
            t2.push_back( { triplet.first.a, triplet.first.b, triplet.first.c, } );
    }
    auto compare = [] ( const auto& l, const auto& r )->bool
    {
        if ( l[0] < r[0] )
            return true;
        if ( l[0] > r[0] )
            return false;
        if ( l[1] < r[1] )
            return true;
        if ( l[1] > r[1] )
            return false;
        return l[2] < r[2];
    };
    tbb::parallel_sort( t3.vec_.begin(), t3.vec_.end(), compare );
    tbb::parallel_sort( t2.vec_.begin(), t2.vec_.end(), compare );
    auto t3Size = t3.size();
    t3.vec_.insert( t3.vec_.end(), std::make_move_iterator( t2.vec_.begin() ), std::make_move_iterator( t2.vec_.end() ) );
    FaceBitSet region3( t3Size );
    region3.flip();
    FaceBitSet region2( t3.size() );
    region2.flip();
    region2 -= region3;

    // create topology
    MeshBuilder::addTriangles( mesh.topology, t3, { .region = &region3, .allowNonManifoldEdge = false } );
    if ( !reportProgress( progressCb, 0.67f ) )
        return {};
    region2 |= region3;
    MeshBuilder::addTriangles( mesh.topology, t3, { .region = &region2, .allowNonManifoldEdge = false } );
    if ( !reportProgress( progressCb, 0.7f ) )
        return {};

    // fill small holes
    const auto bigLength = params_.critHoleLength >= 0.0f ? params_.critHoleLength : pointCloud_.getBoundingBox().diagonal() * 0.7f;
    auto boundaries = findRightBoundary( mesh.topology );
    // setup parameters to prevent any appearance of multiple edges during hole filling
    FillHoleParams fillHoleParams;
    fillHoleParams.multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong;
    bool stoppedFlag = false;
    fillHoleParams.stopBeforeBadTriangulation = &stoppedFlag;
    for ( int i = 0; i < boundaries.size(); ++i )
    {
        const auto& boundary = boundaries[i];
        float length = 0.0f;
        for ( auto e : boundary )
            length += mesh.edgeLength( e );

        if ( length < bigLength )
            fillHole( mesh, boundary.front(), fillHoleParams );
        if ( !reportProgress( progressCb, [&]{ return 0.7f + 0.3f * float( i + 1 ) / float( boundaries.size() ); } ) ) // 70% - 100%
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

std::optional<VertBitSet> findBoundaryPoints( const PointCloud& pointCloud, float radius, float boundaryAngle,
    ProgressCallback cb )
{
    MR_TIMER;
    bool hasNormals = pointCloud.validPoints.find_last() < pointCloud.normals.size();
    std::optional<VertCoords> optNormals;
    if ( !hasNormals )
        optNormals = makeUnorientedNormals( pointCloud, radius, subprogress( cb, 0.0f, 0.5f ) );
    if ( !hasNormals && !optNormals )
        return {};
    const VertCoords& normals = hasNormals ? pointCloud.normals : *optNormals;

    VertBitSet borderPoints( pointCloud.validPoints.size() );
    tbb::enumerable_thread_specific<TriangulationHelpers::TriangulatedFanData> tls;
    auto keepGoing = BitSetParallelFor( pointCloud.validPoints, [&] ( VertId v )
    {
        auto& fanData = tls.local();
        if ( isBoundaryPoint( pointCloud, normals, v, radius, boundaryAngle, fanData ) )
            borderPoints.set( v );
    }, subprogress( cb, hasNormals ? 0.0f : 0.5f, 1.0f ) );
        
    if ( !keepGoing )
        return {};
    return borderPoints;
}

} //namespace MR
