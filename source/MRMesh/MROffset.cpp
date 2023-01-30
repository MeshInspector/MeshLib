#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRMeshFillHole.h"
#include "MRRegionBoundary.h"
#include "MRPch/MRSpdlog.h"
#include "MRVoxelsConversions.h"
#include "MRRingIterator.h"
#include "MRBestFit.h"
#include "MRPch/MROpenvdb.h"
#include "MREdgeIterator.h"
#include "MRVolumeIndexer.h"
#include "MRGeodesicPath.h"

namespace
{
constexpr float autoVoxelNumber = 5e6f;
}

namespace MR
{

tl::expected<Mesh, std::string> offsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER

        float voxelSize = params.voxelSize;
    // Compute voxel size if needed
    if ( voxelSize <= 0.0f )
    {
        auto bb = mp.mesh.computeBoundingBox( mp.region );
        auto vol = bb.volume();
        voxelSize = std::cbrt( vol / autoVoxelNumber );
    }
    MeshToSimpleVolumeParams msParams;
    msParams.maxDistSq = sqr( offset + 3 * params.voxelSize );
    if ( params.type == OffsetParameters::Type::Offset )
        msParams.signMode = MeshToSimpleVolumeParams::SignDetectionMode::WindingRule;
    else
        msParams.signMode = MeshToSimpleVolumeParams::SignDetectionMode::Unsigned;
    msParams.basis.b = mp.mesh.getBoundingBox().min - Vector3f::diagonal( std::abs( offset ) ) - Vector3f::diagonal( params.voxelSize * 3 );
    msParams.basis.A = Matrix3f::scale( voxelSize );
    if ( params.callBack )
        msParams.cb = [intcb = params.callBack]( float p )
    {
        return intcb( 0.5f * p );
    };
    auto maxP = mp.mesh.getBoundingBox().max + Vector3f::diagonal( std::abs( offset ) ) + Vector3f::diagonal( params.voxelSize * 3 );
    msParams.dimensions = Vector3i( mult( maxP - msParams.basis.b, Vector3f::diagonal( 1.0f / params.voxelSize ) ) );

    auto simpleVolume = meshToSimpleVolume( mp.mesh, msParams );

    if ( !simpleVolume )
        return tl::make_unexpected( "CANCELED" );

    SimpleVolumeToMeshParams smParams;
    smParams.basis = msParams.basis;
    smParams.iso = offset;
    smParams.lessInside = true;
    FaceBitSet oddFaces;
    if ( params.saveFeatures )
        smParams.oddVoxelFaces = &oddFaces;
    if ( params.callBack )
        smParams.cb = [intcb = params.callBack]( float p )
    {
        return intcb( 0.5f + 0.5f * p );
    };

    VolumeIndexer indexer( simpleVolume->dims );
    auto meshRes = simpleVolumeToMesh( std::move( *simpleVolume ), smParams );
    if ( !meshRes )
        return tl::make_unexpected( "CANCELED" );

    if ( smParams.oddVoxelFaces )
    {
        // sharpen features
        std::array<FaceId, 20> insideNeighborFaces;
        FaceId nearFace;
        std::array<VertId, 12> edgeVerts;
        std::array<Vector3f, 12> vertNormal;
        auto inverseBasis = smParams.basis.A.inverse();

        const float minFeatureThreshold = 0.8f;
        //const float maxCornerAngleThreshold = 0.7f;

        auto& tp = meshRes->topology;
        auto initFaceSize = tp.faceSize();
        FaceBitSet notVisited( initFaceSize );
        notVisited.flip();
        FaceBitSet resultFaces( initFaceSize );
        VertId firstNewVert = VertId( tp.vertSize() );
        Vector<VoxelId, VertId> voxelPerNewVert; // could be reserved in good implemetation
        while ( notVisited.any() )
        {
            nearFace = {};
            insideNeighborFaces = {};
            edgeVerts = {};
            vertNormal = {};
            insideNeighborFaces[0] = notVisited.find_first();
            assert( insideNeighborFaces[0].valid() );
            bool oddStatus = smParams.oddVoxelFaces->test( insideNeighborFaces[0] );

            // find component faces
            for ( auto f : insideNeighborFaces )
            {
                if ( !f )
                    continue;
                for ( auto e : leftRing( tp, f ) )
                {
                    auto rF = tp.right( e );
                    if ( !rF )
                        continue;
                    if ( smParams.oddVoxelFaces->test( rF ) == oddStatus )
                    {
                        for ( int i = 0; i < insideNeighborFaces.size(); ++i )
                        {
                            if ( insideNeighborFaces[i] == rF )
                                break;
                            if ( !insideNeighborFaces[i] )
                            {
                                insideNeighborFaces[i] = rF;
                                break;
                            }
                        }
                    }
                    else if ( !nearFace )
                        nearFace = rF;
                }
            }
            assert( nearFace );
            // find surface samples
            for ( auto f : insideNeighborFaces )
            {
                if ( !f )
                    continue;
                VertId v[3];
                tp.getTriVerts( f, v );
                for ( int i = 0; i < 3; ++i )
                {
                    for ( int j = 0; j < edgeVerts.size(); ++j )
                    {
                        if ( edgeVerts[j] == v[i] )
                            break;
                        if ( !edgeVerts[j] )
                        {
                            edgeVerts[j] = v[i];
                            vertNormal[j] = mp.mesh.normal( mp.mesh.topology.left( findProjection( meshRes->points[v[i]], mp.mesh ).mtp.e ) );
                            break;
                        }
                    }
                }
            }

            float minEntringDot = FLT_MAX;
            for ( int i = 0; i < edgeVerts.size(); ++i )
            {
                if ( !edgeVerts[i] )
                    continue;
                for ( int j = i; j < edgeVerts.size(); ++j )
                {
                    if ( !edgeVerts[j] )
                        continue;
                    auto dotRes = dot( vertNormal[i], vertNormal[j] );
                    if ( dotRes < minEntringDot )
                        minEntringDot = dotRes;
                }
            }

            bool feature = minEntringDot != FLT_MAX && minEntringDot < minFeatureThreshold;
            //bool angle = false;
            //if ( feature )
            //{
            //    auto crossNorm = cross( minNormalOuter, minNormalInner ).normalized();
            //
            //}
            if ( feature )
            {
                //for ( auto f : insideNeighborFaces )
                //    if ( f )
                //        resultFaces.set( f );
                for ( auto f : insideNeighborFaces )
                    if ( f )
                        tp.deleteFace( f );
                EdgeId holeEdge;
                for ( auto e : leftRing( tp, nearFace ) )
                {
                    if ( !tp.right( e ) )
                    {
                        holeEdge = e.sym();
                        break;
                    }
                }
                if ( holeEdge )
                {
                    auto oldFaceSize = FaceId( tp.faceSize() );
                    auto newV = fillHoleTrivially( *meshRes, holeEdge );
                    auto newFaceSize = tp.faceSize();
                    oddFaces.resize( newFaceSize );
                    for ( FaceId f = oldFaceSize; f < newFaceSize; ++f )
                        oddFaces.set( f, oddStatus );

                    auto radiusVectorFromOrg = meshRes->points[newV] - smParams.basis.b;
                    auto voxelPos = inverseBasis * radiusVectorFromOrg;
                    voxelPerNewVert.autoResizeSet( newV - firstNewVert, indexer.toVoxelId( Vector3i( voxelPos ) ) );

                    int planeCounter = 0;
                    PlaneAccumulator accum;
                    for ( int i = 0; i < edgeVerts.size(); ++i )
                    {
                        if ( !edgeVerts[i] )
                            continue;
                        const auto& point = meshRes->points[edgeVerts[i]];
                        accum.addPlane( Plane3f::fromDirAndPt( vertNormal[i], point ) );
                        ++planeCounter;
                    }
                    if ( planeCounter > 1 )
                    {
                        auto newPos = accum.findBestCrossPoint( meshRes->points[newV] );
                        //auto newVoxelPos = inverseBasis * ( newPos - smParams.basis.b );
                        //if ( Vector3i( newVoxelPos ) != Vector3i( voxelPos ) )
                        //{
                        //    auto posDiff = newVoxelPos - voxelPos;
                        //    int maxDir = 0;
                        //    if ( std::abs( posDiff.y ) > std::abs( posDiff.x ) && std::abs( posDiff.y ) > std::abs( posDiff.z ) )
                        //        maxDir = 1;
                        //    else if ( std::abs( posDiff.z ) > std::abs( posDiff.x ) && std::abs( posDiff.z ) > std::abs( posDiff.y ) )
                        //        maxDir = 2;
                        //    float maxDirPos = Vector3i( voxelPos )[maxDir] + 1.49f;
                        //    if ( posDiff[maxDir] < 0.0f )
                        //        maxDirPos -= 1.98f;
                        //    auto ratio = ( maxDirPos - voxelPos[maxDir] ) / ( posDiff[maxDir] );
                        //    newVoxelPos = ( 1.0f - ratio ) * voxelPos + ratio * newVoxelPos;
                        //    newPos = smParams.basis.b + smParams.basis.A * newVoxelPos;
                        //}
                        //else
                            meshRes->points[newV] = newPos;
                    }
                }
            }
            for ( auto f : insideNeighborFaces )
                if ( f && f < initFaceSize )
                    notVisited.set( f, false );
        }
        for ( auto ue : undirectedEdges( tp ) )
        {
            auto v0 = tp.dest( tp.next( ue ) );
            auto v1 = tp.dest( tp.prev( ue ) );
            if ( v0 < firstNewVert || v1 < firstNewVert || v0 == v1 )
                continue;
            //if ( isUnfoldQuadrangleConvex( meshRes->orgPnt( ue ), meshRes->points[v0], meshRes->destPnt( ue ), meshRes->points[v1] ) )
                tp.flipEdge( ue );
        }
    }

    return *meshRes;
}

tl::expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
        if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() )
        {
            spdlog::error( "Only closed meshes allowed for double offset." );
            return tl::make_unexpected( "Only closed meshes allowed for double offset." );
        }
    if ( params.type == OffsetParameters::Type::Shell )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, params.adaptivity, params.callBack );
}

tl::expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER;

    Mesh mesh;
    auto contours = polyline.topology.convertToContours<Vector3f>(
        [&points = polyline.points]( VertId v )
    {
        return points[v];
    } );

    std::vector<EdgeId> newHoles;
    newHoles.reserve( contours.size() );
    for ( auto& cont : contours )
    {
        if ( cont[0] != cont.back() )
            cont.insert( cont.end(), cont.rbegin(), cont.rend() );
        newHoles.push_back( mesh.addSeparateEdgeLoop( cont ) );
    }

    for ( auto h : newHoles )
        makeDegenerateBandAroundHole( mesh, h );

    return offsetMesh( mesh, offset, params );
}

}
#endif
