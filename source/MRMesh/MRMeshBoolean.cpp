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
#include "MRFillContour.h"
#include "MRPrecisePredicates3.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRMeshCollide.h"

namespace
{

using namespace MR;

enum class EdgeTriComponent
{
    Edge,
    Tri,
};

Vector3f findEdgeTriIntersectionPoint( const Mesh& edgeMesh, EdgeId edge, const Mesh& triMesh, FaceId tri,
                                       const CoordinateConverters& converters,
                                       const AffineXf3f* rigidB2A = nullptr,
                                       EdgeTriComponent meshBComponent = EdgeTriComponent::Edge )
{
    Vector3f ev0, ev1;
    ev0 = edgeMesh.orgPnt( edge );
    ev1 = edgeMesh.destPnt( edge );

    Vector3f fv0, fv1, fv2;
    triMesh.getTriPoints( tri, fv0, fv1, fv2 );

    if ( rigidB2A )
    {
        const auto& xf = *rigidB2A;
        switch ( meshBComponent )
        {
        case EdgeTriComponent::Edge:
            ev0 = xf( ev0 );
            ev1 = xf( ev1 );
            break;
        case EdgeTriComponent::Tri:
            fv0 = xf( fv0 );
            fv1 = xf( fv1 );
            fv2 = xf( fv2 );
            break;
        }
    }

    return findTriangleSegmentIntersectionPrecise( fv0, fv1, fv2, ev0, ev1, converters );
}

void gatherEdgeInfo( const MeshTopology& topology, EdgeId e, FaceBitSet& faces, VertBitSet& dests )
{
    faces.set( topology.left( e ) );
    faces.set( topology.right( e ) );
    dests.set( topology.dest( e ) );
}

OneMeshContours getOtherMeshContoursByHint( const OneMeshContours& aContours, const ContinuousContours& contours, 
    const AffineXf3f* rigidB2A = nullptr )
{
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();
    OneMeshContours bMeshContours = aContours;
    for ( int j = 0; j < bMeshContours.size(); ++j )
    {
        const auto& inCont = contours[j];
        auto& outCont = bMeshContours[j].intersections;
        assert( inCont.size() == outCont.size() );
        tbb::parallel_for( tbb::blocked_range<int>( 0, int( inCont.size() ) ),
                [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                const auto& inInter = inCont[i];
                auto& outInter = outCont[i];
                outInter.primitiveId = inInter.isEdgeATriB ?
                    std::variant<FaceId, EdgeId, VertId>( inInter.tri ) : 
                    std::variant<FaceId, EdgeId, VertId>( inInter.edge );
                if ( rigidB2A )
                    outInter.coordinate = inverseXf( outCont[i].coordinate );
            }
        } );
    }
    return bMeshContours;
}

}

namespace MR
{

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */ )
{
    bool needCutMeshA = operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB;
    bool needCutMeshB = operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA;
    if ( needCutMeshA )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        meshA.getAABBTree();
    }
    if ( needCutMeshB )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        meshB.getAABBTree();
    }
    return boolean( Mesh( meshA ), Mesh( meshB ), operation, rigidB2A, mapper );
}

BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */ )
{
    MR_TIMER;
    CoordinateConverters converters;
    PreciseCollisionResult intersections;
    ContinuousContours contours;

    bool needCutMeshA = operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB;
    bool needCutMeshB = operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA;

    converters = getVectorConverters( meshA, meshB, rigidB2A );

    FaceMap new2orgSubdivideMapA;
    FaceMap new2orgSubdivideMapB;
    std::vector<int> prevLoneContoursIds;
    for ( ;;)
    {
        // find intersections
        intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
        // order intersections
        contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
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
            auto loneIntsA = getOneMeshIntersectionContours( meshA, meshB, loneA, true, converters, rigidB2A );
            removeDegeneratedContours( loneIntsA );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshA, loneIntsA, mapPointer );
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
            auto loneIntsB = getOneMeshIntersectionContours( meshA, meshB, loneB, false, converters, rigidB2A );
            removeDegeneratedContours( loneIntsB );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshB, loneIntsB, mapPointer );
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
    // clear intersections
    intersections = {};

    std::vector<EdgePath> cutA, cutB;
    BooleanResult result;
    OneMeshContours meshAContours;
    OneMeshContours meshBContours;
    // prepare it before as far as MeshA will be changed after cut
    Mesh meshACopyBuffer; // second copy may be necessary because sort data need mesh after separation, and cut A will break it
    std::unique_ptr<SortIntersectionsData> dataForB;
    if ( needCutMeshB )
    {
        if ( needCutMeshA )
        {
            // cutMesh A will break mesh so make copy
            meshACopyBuffer = meshA;
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ meshACopyBuffer, contours, converters.toInt, rigidB2A, meshA.topology.vertSize(), true } );
        }
        else
        {
            // no need to cut A, so no need to copy
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ meshA, contours, converters.toInt, rigidB2A, meshA.topology.vertSize(), true } );
        }
    }
    if ( needCutMeshA )
        meshAContours = getOneMeshIntersectionContours( meshA, meshB, contours, true, converters, rigidB2A );
    if ( needCutMeshB )
    {
        if ( needCutMeshA )
            meshBContours = getOtherMeshContoursByHint( meshAContours, contours, rigidB2A );
        else
            meshBContours = getOneMeshIntersectionContours( meshA, meshB, contours, false, converters, rigidB2A );
    }

    if ( needCutMeshA )
    {
        // prepare contours per mesh
        SortIntersectionsData dataForA{ meshB,contours,converters.toInt,rigidB2A,meshA.topology.vertSize(),false};
        FaceMap* cut2oldAPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2origin : nullptr;
        // cut meshes
        CutMeshParameters params;
        params.sortData = &dataForA;
        params.new2OldMap = cut2oldAPtr;
        auto res = cutMesh( meshA, meshAContours, params );
        meshAContours.clear();
        meshAContours.shrink_to_fit(); // free memory
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
        auto res = cutMesh( meshB, meshBContours, params );
        meshBContours.clear();
        meshBContours.shrink_to_fit(); // free memory
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
    auto res = doBooleanOperation( std::move( meshA ), std::move( meshB ), cutA, cutB, operation, rigidB2A, mapper );
    if ( res.has_value() )
        result.mesh = std::move( res.value() );
    else
        result.errorString = res.error();
    return result;
}

BooleanResultPoints getBooleanPoints( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation, const AffineXf3f* rigidB2A )
{
    MR_TIMER

    BooleanResultPoints result;
    result.meshAVerts.resize( meshA.topology.lastValidVert() + 1 );
    result.meshBVerts.resize( meshB.topology.lastValidVert() + 1 );

    const auto converters = getVectorConverters( meshA, meshB, rigidB2A );
    const auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
    result.intersectionPoints.reserve( intersections.edgesAtrisB.size() + intersections.edgesBtrisA.size() );

    FaceBitSet collFacesA, collFacesB;
    VertBitSet collOuterVertsA, collOuterVertsB;
    collFacesA.resize( meshA.topology.lastValidFace() + 1 );
    collFacesB.resize( meshB.topology.lastValidFace() + 1 );
    collOuterVertsA.resize( meshA.topology.lastValidVert() + 1 );
    collOuterVertsB.resize( meshB.topology.lastValidVert() + 1 );
    for ( const auto& et : intersections.edgesAtrisB )
    {
        gatherEdgeInfo( meshA.topology, et.edge, collFacesA, collOuterVertsA );
        collFacesB.set( et.tri );

        const auto isect = findEdgeTriIntersectionPoint( meshA, et.edge, meshB, et.tri, converters, rigidB2A, EdgeTriComponent::Tri );
        result.intersectionPoints.emplace_back( isect );
    }
    for ( const auto& et : intersections.edgesBtrisA )
    {
        gatherEdgeInfo( meshB.topology, et.edge, collFacesB, collOuterVertsB );
        collFacesA.set( et.tri );

        const auto isect = findEdgeTriIntersectionPoint( meshB, et.edge, meshA, et.tri, converters, rigidB2A, EdgeTriComponent::Edge );
        result.intersectionPoints.emplace_back( isect );
    }

    auto collBordersA = findRegionBoundary( meshA.topology, collFacesA );
    auto collBordersB = findRegionBoundary( meshB.topology, collFacesB );

    const bool needInsidePartA = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideA || operation == BooleanOperation::DifferenceBA );
    const bool needInsidePartB = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideB || operation == BooleanOperation::DifferenceAB );

    if ( operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB )
    {
        std::erase_if( collBordersA, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartA != collOuterVertsA.test( meshA.topology.dest( edgeLoop.front() ) );
        } );

        collFacesA = fillContourLeft( meshA.topology, collBordersA );
        result.meshAVerts = getInnerVerts( meshA.topology, collFacesA );

        const auto aComponents = MeshComponents::getAllComponents(meshA);

        for ( const auto& aComponent : aComponents )
        {
            const auto aComponentVerts = getInnerVerts( meshA.topology, aComponent );
            const bool intersects = aComponentVerts.intersects( result.meshAVerts );
            const bool inside = isInside( MeshPart( meshA, &aComponent ), MeshPart( meshB ), rigidB2A );
            if ( !intersects &&
                ( ( needInsidePartA &&  inside) ||
                ( !needInsidePartA && !inside ) ) )
            {
                result.meshAVerts |= aComponentVerts;
            }
        }
    }

    if ( operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA )
    {
        std::erase_if( collBordersB, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartB != collOuterVertsB.test( meshB.topology.dest( edgeLoop.front() ) );
        } );

        collFacesB = fillContourLeft(meshB.topology, collBordersB);
        result.meshBVerts = getInnerVerts( meshB.topology, collFacesB );
        
        const auto bComponents = MeshComponents::getAllComponents(meshB);
        std::unique_ptr<AffineXf3f> rigidA2B;
        if ( rigidB2A )
            rigidA2B = std::make_unique<AffineXf3f>( rigidB2A->inverse() );

        for ( const auto& bComponent : bComponents )
        {
            const auto bComponentVerts = getInnerVerts( meshB.topology, bComponent );
            const bool intersects = bComponentVerts.intersects( result.meshBVerts );
            const bool inside = isInside( MeshPart( meshB, &bComponent ), MeshPart( meshA ), rigidA2B.get() );

            if ( !intersects &&
                ( ( needInsidePartB && inside ) ||
                ( !needInsidePartB && !inside ) ) )
            {
                result.meshBVerts |= bComponentVerts;
            }
        }
    }

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
