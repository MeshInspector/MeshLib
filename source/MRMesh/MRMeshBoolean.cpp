#include "MRMeshBoolean.h"
#include "MRMesh.h"
#include "MRMeshCollidePrecise.h"
#include "MRIntersectionContour.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRTorus.h"
#include "MRMatrix3.h"
#include "MRAffineXf3.h"
#include "MRLog.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation opearation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */ )
{
    MR_TIMER;
    CoordinateConverters converters;
    PreciseCollisionResult intersections;
    ContinuousContours contours;
    Mesh meshACut, meshBCut;

    bool needCutMeshA = opearation != BooleanOperation::InsideB && opearation != BooleanOperation::OutsideB;
    bool needCutMeshB = opearation != BooleanOperation::InsideA && opearation != BooleanOperation::OutsideA;

    if ( needCutMeshA )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        meshA.getAABBTree();
        meshACut = meshA;
    }
    if ( needCutMeshB )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        meshB.getAABBTree();
        meshBCut = meshB;
    }

    converters = getVectorConverters( meshA, meshB, rigidB2A );

    const Mesh& constMeshARef = needCutMeshA ? meshACut : meshA;
    const Mesh& constMeshBRef = needCutMeshB ? meshBCut : meshB;

    FaceMap new2orgSubdivideMapA;
    FaceMap new2orgSubdivideMapB;
    std::vector<int> prevLoneContoursIds;
    for ( ;;)
    {
        // find intersections
        intersections = findCollidingEdgeTrisPrecise( constMeshARef, constMeshBRef, converters.toInt, rigidB2A );
        // order intersections
        contours = orderIntersectionContours( constMeshARef.topology, constMeshBRef.topology, intersections );
        // find lone
        auto loneContoursIds = detectLoneContours( contours );

        if ( !loneContoursIds.empty() && loneContoursIds == prevLoneContoursIds )
        {
            // in some rare cases there are lone contours with zero area that cannot be resolved
            // they lead to infinite loop, so just try to remove them
            removeLoneContours( contours );
            break;
        }

        prevLoneContoursIds = loneContoursIds;

        // separate A lone from B lone
        ContinuousContours loneA;
        ContinuousContours loneB;
        for ( int i = 0; i < loneContoursIds.size(); ++i )
        {
            const auto& contour = contours[loneContoursIds[i]];
            if ( contour[0].isEdgeATriB )
                loneB.push_back( contour );
            else
                loneA.push_back( contour );
        }
        if ( loneContoursIds.empty() ||
            ( loneA.empty() && !needCutMeshB ) ||
            ( loneB.empty() && !needCutMeshA ) )
            break;
        // subdivide owners of lone
        if ( !loneA.empty() && needCutMeshA )
        {
            auto loneIntsA = getOneMeshIntersectionContours( constMeshARef, constMeshBRef, loneA, true, converters, rigidB2A );
            removeDegeneratedContours( loneIntsA );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshACut, loneIntsA, mapPointer );
            if ( new2orgSubdivideMapA.size() < new2orgLocalMap.size() )
                new2orgSubdivideMapA.resize( new2orgLocalMap.size() );
            tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId( 0 ), FaceId( int( new2orgLocalMap.size() ) ) ),
                [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId i = range.begin(); i < range.end(); ++i )
                {
                    if ( !new2orgLocalMap[i] )
                        continue;
                    FaceId refFace = new2orgLocalMap[i];
                    if ( new2orgSubdivideMapA[refFace] )
                        refFace = new2orgSubdivideMapA[refFace];
                    new2orgSubdivideMapA[i] = refFace;
                }
            } );
        }
        if ( !loneB.empty() && needCutMeshB )
        {
            auto loneIntsB = getOneMeshIntersectionContours( constMeshARef, constMeshBRef, loneB, false, converters, rigidB2A );
            removeDegeneratedContours( loneIntsB );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshBCut, loneIntsB, mapPointer );
            if ( new2orgSubdivideMapB.size() < new2orgLocalMap.size() )
                new2orgSubdivideMapB.resize( new2orgLocalMap.size() );
            tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId( 0 ), FaceId( int( new2orgLocalMap.size() ) ) ),
                [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId i = range.begin(); i < range.end(); ++i )
                {
                    if ( !new2orgLocalMap[i] )
                        continue;
                    FaceId refFace = new2orgLocalMap[i];
                    if ( new2orgSubdivideMapB[refFace] )
                        refFace = new2orgSubdivideMapB[refFace];
                    new2orgSubdivideMapB[i] = refFace;
                }
            } );
        }
    }
    std::vector<EdgePath> cutA, cutB;
    BooleanResult result;
    OneMeshContours meshBContours;
    // prepare it before as far as MeshA will be changed after cut
    Mesh meshACopyBuffer; // second copy may be necessary because sort data need mesh after separation, and cut A will break it
    std::unique_ptr<SortIntersectionsData> dataForB;
    if ( needCutMeshB )
    {
        if ( needCutMeshA )
        {
            // cutMesh A will break mesh so make copy
            meshACopyBuffer = constMeshARef;
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ meshACopyBuffer, contours, converters.toInt, rigidB2A, constMeshARef.topology.vertSize(), true } );
        }
        else
        {
            // no need to cut A, so no need to copy
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ constMeshARef, contours, converters.toInt, rigidB2A, constMeshARef.topology.vertSize(), true } );
        }
    }
    if ( needCutMeshB )
        meshBContours = getOneMeshIntersectionContours( constMeshARef, constMeshBRef, contours, false, converters, rigidB2A );

    if ( needCutMeshA )
    {
        // prepare contours per mesh
        auto meshAContours = getOneMeshIntersectionContours( constMeshARef, constMeshBRef, contours, true, converters, rigidB2A );
        SortIntersectionsData dataForA{ constMeshBRef,contours,converters.toInt,rigidB2A,constMeshARef.topology.vertSize(),false};
        FaceMap* cut2oldAPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2origin : nullptr;
        // cut meshes
        CutMeshParameters params;
        params.sortData = &dataForA;
        params.new2OldMap = cut2oldAPtr;
        auto res = cutMesh( meshACut, meshAContours, params );
        if ( cut2oldAPtr && !new2orgSubdivideMapA.empty() )
        {
            tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId( 0 ), FaceId( int( cut2oldAPtr->size() ) ) ),
                [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId i = range.begin(); i < range.end(); ++i )
                {
                    if ( !( *cut2oldAPtr )[i] )
                        continue;
                    FaceId refFace = ( *cut2oldAPtr )[i];
                    if ( new2orgSubdivideMapA.size() > refFace && new2orgSubdivideMapA[refFace] )
                        ( *cut2oldAPtr )[i] = new2orgSubdivideMapA[refFace];
                }
            } );
        }
        if ( res.fbsWithCountourIntersections.any() )
        {
            result.meshABadContourFaces = std::move( res.fbsWithCountourIntersections );
            result.errorString = "Bad contour on " + std::to_string( result.meshABadContourFaces.count() ) + " mesh A faces, " + 
                "probably mesh B has self-intersections on contours lying on these faces.";
            return result;
        }
        cutA = std::move( res.resultCut );
    }
    if ( needCutMeshB )
    {
        FaceMap* cut2oldBPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2origin : nullptr;
        // cut meshes
        CutMeshParameters params;
        params.sortData = dataForB.get();
        params.new2OldMap = cut2oldBPtr;
        auto res = cutMesh( meshBCut, meshBContours, params );
        if ( cut2oldBPtr && !new2orgSubdivideMapB.empty() )
        {
            tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId( 0 ), FaceId( int( cut2oldBPtr->size() ) ) ),
                [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId i = range.begin(); i < range.end(); ++i )
                {
                    if ( !( *cut2oldBPtr )[i] )
                        continue;
                    FaceId refFace = ( *cut2oldBPtr )[i];
                    if ( new2orgSubdivideMapB.size() > refFace && new2orgSubdivideMapB[refFace] )
                        ( *cut2oldBPtr )[i] = new2orgSubdivideMapB[refFace];
                }
            } );
        }
        if ( res.fbsWithCountourIntersections.any() )
        {
            result.meshBBadContourFaces = std::move( res.fbsWithCountourIntersections );
            result.errorString = "Bad contour on " + std::to_string( result.meshBBadContourFaces.count() ) + " mesh B faces, " +
                "probably mesh A has self-intersections on contours lying on these faces.";
            return result;
        }
        cutB = std::move( res.resultCut );
    }
    // do operation
    auto res = doBooleanOperation( constMeshARef, constMeshBRef, cutA, cutB, opearation, rigidB2A, mapper );
    if ( res.has_value() )
        result.mesh = std::move( res.value() );
    else
        result.errorString = res.error();
    return result;
}

TEST( MRMesh, MeshBoolean )
{
    Mesh meshA = makeTorus( 1.1f, 0.5f, 8, 8 );
    Mesh meshB = makeTorus( 1.0f, 0.2f, 8, 8 );
    meshB.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), Vector3f::plusY() ) ) );

    const float shiftStep = 0.2f;
    const float angleStep = PI_F;/* *1.0f / 3.0f*/;
    const std::array<Vector3f, 3> baseAxis{Vector3f::plusX(),Vector3f::plusY(),Vector3f::plusZ()};
    for ( int maskTrans = 0; maskTrans < 8; ++maskTrans )
    {
        for ( int maskRot = 0; maskRot < 8; ++maskRot )
        {
            for ( float shift = 0.01f; shift < 0.2f; shift += shiftStep )
            {
                Vector3f shiftVec;
                for ( int i = 0; i < 3; ++i )
                    if ( maskTrans & ( 1 << i ) )
                        shiftVec += shift * baseAxis[i];
                for ( float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep )
                {
                    Matrix3f rotation;
                    for ( int i = 0; i < 3; ++i )
                        if ( maskRot & ( 1 << i ) )
                            rotation = Matrix3f::rotation( baseAxis[i], angle ) * rotation;

                    AffineXf3f xf;
                    xf = AffineXf3f::translation( shiftVec ) * AffineXf3f::linear( rotation );

                    EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation::Union, &xf ).valid() );
                    EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation::Intersection, &xf ).valid() );
                }
            }
        }
    }
}

} //namespace MR
