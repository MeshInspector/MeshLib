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
#include "MRPlane3.h"
#include "MRVector3.h"
#include "MRBox.h"
#include "MRTimer.h"
#include "MRPointCloudTriangulationHelpers.h"
#include "MRRegionBoundary.h"
#include "MRParallelFor.h"
#include "MRLocalTriangulations.h"
#include "MRMeshFixer.h"
#include "MREdgePaths.h"
#include <parallel_hashmap/phmap.h>

namespace MR
{

namespace
{

class PointCloudTriangulator
{
public:
    PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params );
    PointCloudTriangulator( Mesh && targetMesh, const PointCloud& pointCloud, const TriangulationParameters& params );
    bool addPoints( PointCloud& extraPoints );
    bool triangulate( const ProgressCallback& progressCb );
    Mesh takeTargetMesh() { return std::move( targetMesh_ ); }

private:
    /// adds given triangles in targetMesh_
    bool makeMesh_( Triangulation && t3, Triangulation && t2, const ProgressCallback& progressCb );

    Mesh targetMesh_;
    const PointCloud& pointCloud_;
    TriangulationParameters params_;
    VertMap cloud2mesh_; ///< from pointCloud_ to targetMesh_
};

PointCloudTriangulator::PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params ) :
    pointCloud_{pointCloud},
    params_{params}
{
}

PointCloudTriangulator::PointCloudTriangulator( Mesh && targetMesh, const PointCloud& pointCloud, const TriangulationParameters& params ) :
    targetMesh_{std::move( targetMesh )},
    pointCloud_{pointCloud},
    params_{params}
{
}

bool PointCloudTriangulator::addPoints( PointCloud& extraPoints )
{
    MR_TIMER;
    auto nextPointId = extraPoints.points.endId();
    assert( nextPointId == extraPoints.normals.endId() );
    assert( nextPointId == extraPoints.validPoints.endId() );

    auto bdVerts = targetMesh_.topology.findBdVerts();
    if ( bdVerts.none() )
        return false;

    // add all extraPoints in targetMesh_
    targetMesh_.points.reserve( targetMesh_.points.size() + extraPoints.validPoints.count() );
    const auto totalPoints = extraPoints.points.size() + bdVerts.count();
    cloud2mesh_.resizeNoInit( totalPoints );
    for ( VertId pid : extraPoints.validPoints )
    {
        cloud2mesh_[pid] = targetMesh_.points.endId();
        targetMesh_.points.push_back( extraPoints.points[pid] );
    }

    // add boundary vertices of tagetMesh_ in extraPoints
    cloud2mesh_.resizeNoInit( totalPoints );
    extraPoints.points.reserve( totalPoints );
    extraPoints.normals.reserve( totalPoints );
    extraPoints.validPoints.resize( totalPoints, true );
    for ( auto bdV : bdVerts )
    {
        assert( nextPointId == extraPoints.normals.endId() );
        cloud2mesh_[nextPointId] = bdV;
        extraPoints.points.push_back( targetMesh_.points[bdV] );
        extraPoints.normals.push_back( targetMesh_.pseudonormal( bdV ) );
        ++nextPointId;
    }

    return true;
}

bool PointCloudTriangulator::triangulate( const ProgressCallback& progressCb )
{
    MR_TIMER;
    assert( ( params_.numNeighbours <= 0 && params_.radius > 0 )
         || ( params_.numNeighbours > 0 && params_.radius <= 0 ) );

    auto optLocalTriangulations = TriangulationHelpers::buildUnitedLocalTriangulations( pointCloud_,
        {
            .radius = params_.radius,
            .numNeis = params_.numNeighbours,
            .critAngle = params_.critAngle,
            .boundaryAngle = params_.boundaryAngle,
            .trustedNormals = pointCloud_.hasNormals() ? &pointCloud_.normals : nullptr,
            .automaticRadiusIncrease = params_.automaticRadiusIncrease,
            .searchNeighbors = params_.searchNeighbors
        }, subprogress( progressCb, 0.0f, pointCloud_.hasNormals() ? 0.4f : 0.3f ) );
    if ( !optLocalTriangulations )
        return {};
    auto & localTriangulations = *optLocalTriangulations;

    Triangulation t3, t2;
    if ( pointCloud_.hasNormals() )
        findRepeatedOrientedTriangles( localTriangulations, &t3, &t2 );
    else
        autoOrientLocalTriangulations( pointCloud_, localTriangulations, pointCloud_.validPoints, subprogress( progressCb, 0.3f, 0.5f ), &t3, &t2 );

    return makeMesh_( std::move( t3 ), std::move( t2 ), subprogress( progressCb, 0.5f, 1.0f ) );
}

bool PointCloudTriangulator::makeMesh_( Triangulation && t3, Triangulation && t2, const ProgressCallback& progressCb )
{
    MR_TIMER;

    if ( targetMesh_.points.empty() )
    {
        assert( cloud2mesh_.empty() );
        targetMesh_.points = pointCloud_.points;
    }
    else
    {
        // translate t2 and t3 from pointCloud into targetMesh_
        ParallelFor( t3, [&]( FaceId f )
        {
            for ( int i = 0; i < 3; ++i )
            {
                assert( cloud2mesh_[t3[f][i]] );
                t3[f][i] = cloud2mesh_[t3[f][i]];
            }
        } );
        ParallelFor( t2, [&]( FaceId f )
        {
            for ( int i = 0; i < 3; ++i )
            {
                assert( cloud2mesh_[t2[f][i]] );
                t2[f][i] = cloud2mesh_[t2[f][i]];
            }
        } );
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
    const MeshBuilder::BuildSettings bsettings
    {
        .region = &region3,
        .shiftFaceId = targetMesh_.topology.getValidFaces().endId(),
        .allowNonManifoldEdge = false
    };
    MeshBuilder::addTriangles( targetMesh_.topology, t3, bsettings );
    if ( !reportProgress( progressCb, 0.1f ) )
        return false;
    region3 |= region2;
    MeshBuilder::addTriangles( targetMesh_.topology, t3, bsettings );
    if ( !reportProgress( progressCb, 0.2f ) )
        return false;

    // remove bad triangles
    targetMesh_.deleteFaces( findHoleComplicatingFaces( targetMesh_ ) );

    // fill small holes
    const auto maxHolePerimeterToFill = params_.critHoleLength >= 0.0f ?
        params_.critHoleLength :
        pointCloud_.getBoundingBox().diagonal() * 0.1f;
    auto boundaries = findRightBoundary( targetMesh_.topology );
    // setup parameters to prevent any appearance of multiple edges during hole filling
    FillHoleParams fillHoleParams;
    fillHoleParams.multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong;
    bool stoppedFlag = false;
    fillHoleParams.stopBeforeBadTriangulation = &stoppedFlag;
    for ( int i = 0; i < boundaries.size(); ++i )
    {
        const auto& boundary = boundaries[i];

        if ( (float)calcPathLength( boundary, targetMesh_ ) <= maxHolePerimeterToFill )
            fillHole( targetMesh_, boundary.front(), fillHoleParams );

        if ( !reportProgress( progressCb, [&]{ return 0.3f + 0.7f * float( i + 1 ) / float( boundaries.size() ); } ) ) // 30% - 100%
            return false;
    }

    return true;
}

} // anonymous namespace

std::optional<Mesh> triangulatePointCloud( const PointCloud& pointCloud, const TriangulationParameters& params /*= {} */,
    const ProgressCallback& progressCb )
{
    MR_TIMER;
    PointCloudTriangulator triangulator( pointCloud, params );
    if ( triangulator.triangulate( progressCb ) )
        return triangulator.takeTargetMesh();
    return {};
}

bool fillHolesWithExtraPoints( Mesh & mesh, PointCloud& extraPoints,
    const TriangulationParameters& params, const ProgressCallback& progressCb )
{
    MR_TIMER;
    if ( !extraPoints.hasNormals() )
    {
        assert( false );
        return false;
    }
    const auto szPoints = extraPoints.points.size();
    extraPoints.normals.resize( szPoints );
    extraPoints.validPoints.resize( szPoints );

    PointCloudTriangulator triangulator( std::move( mesh ), extraPoints, params );
    bool res = true;
    if ( triangulator.addPoints( extraPoints ) )
        res = triangulator.triangulate( progressCb );
    // else no holes in mesh and res = true

    extraPoints.points.resize( szPoints );
    extraPoints.normals.resize( szPoints );
    extraPoints.validPoints.resize( szPoints );

    mesh = triangulator.takeTargetMesh();
    return res;
}

} //namespace MR
