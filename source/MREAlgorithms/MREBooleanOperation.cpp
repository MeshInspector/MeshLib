#include "MREBooleanOperation.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRFillContour.h"
#include "MRMesh/MRContoursStitch.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRAffineXf3.h"

#include "MRMesh/MRMesh.h"
#include "MREAlgorithms/MREBooleanOperation.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include "MREAlgorithms/MREIntersectionContour.h"
#include "MREAlgorithms/MREContoursCut.h"
#include "MRPch/MRTBB.h"

#pragma warning(disable: 4996) //deprecated function call
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace MRE
{
using namespace MR;

std::string findBooleanIntersections( MR::Mesh& meshA, MR::Mesh& meshB,
    std::vector<MR::EdgePath>& cutARes, std::vector<MR::EdgePath>& cutBRes )
{
    using namespace MR;
    using namespace MRE;
    MeshPart a{ meshA };
    MeshPart b{ meshB };

    CoordinateConverters converters;

    std::unique_ptr<PreciseCollisionResult> intersectionRes_;
    std::unique_ptr<ContinuousContours> contoursRes_;
    MR::FaceMap new2orgSubdivideMapA;
    MR::FaceMap new2orgSubdivideMapB;
    std::unique_ptr<OneMeshContours> meshAContours_;
    std::unique_ptr<OneMeshContours> meshBContours_;
    MR::FaceBitSet badFacesA_;
    MR::FaceBitSet badFacesB_;
    std::string errorStr;

    std::vector<int> prevLoneContours;
    for ( ;;)
    {
        converters = getVectorConverters( a, b );
        intersectionRes_ = std::make_unique<PreciseCollisionResult>(
            findCollidingEdgeTrisPrecise( a, b, converters.toInt ) );

        contoursRes_ = std::make_unique<ContinuousContours>(
            orderIntersectionContours( meshA.topology, meshB.topology, *intersectionRes_ ) );

        auto loneContoursIds = detectLoneContours( *contoursRes_ );
        if ( loneContoursIds.empty() )
            break;
        if ( loneContoursIds == prevLoneContours )
        {
            // in some rare cases there are lone contours with zero area that cannot be resolved
            // they lead to infinite loop, so just try to remove them
            removeLoneContours( *contoursRes_ );
            break;
        }

        prevLoneContours = loneContoursIds;

        ContinuousContours loneA;
        ContinuousContours loneB;
        for ( int i = 0; i < loneContoursIds.size(); ++i )
        {
            const auto& contour = ( *contoursRes_ )[loneContoursIds[i]];
            if ( contour[0].isEdgeATriB )
                loneB.push_back( contour );
            else
                loneA.push_back( contour );
        }

        auto loneIntsA = getOneMeshIntersectionContours( meshA, meshB, loneA, true, converters );
        auto loneIntsB = getOneMeshIntersectionContours( meshA, meshB, loneB, false, converters );

        if ( !loneIntsA.empty() )
        {
            FaceMap new2orgLocalMap;
            subdivideLoneContours( meshA, loneIntsA, &new2orgLocalMap );
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
        if ( !loneIntsB.empty() )
        {
            FaceMap new2orgLocalMap;
            subdivideLoneContours( meshB, loneIntsB, &new2orgLocalMap );
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


    meshAContours_ = std::make_unique<OneMeshContours>(
        getOneMeshIntersectionContours( meshA, meshB, *contoursRes_, true, converters ) );

    meshBContours_ = std::make_unique<OneMeshContours>(
        getOneMeshIntersectionContours( meshA, meshB, *contoursRes_, false, converters ) );

    SortIntersectionsData dataForA{ meshB,*contoursRes_,converters.toInt,nullptr,a.mesh.topology.vertSize(),false };
    // second copy is need because cutMesh(meshA) will break sort data for B
    // we don't need same copy for sort data A because cutMesh(meshA) is first, so mesh B can't be broken during cut
    auto meshACopy = meshA;
    SortIntersectionsData dataForB{ meshACopy,*contoursRes_,converters.toInt,nullptr,a.mesh.topology.vertSize(),true };

    auto cutA = cutMesh( meshA, *meshAContours_, { &dataForA } );
    auto cutB = cutMesh( meshB, *meshBContours_, { &dataForB } );
    if ( cutA.fbsWithCountourIntersections.any() )
    {
        for ( auto f : cutA.fbsWithCountourIntersections )
        {
            if ( f < new2orgSubdivideMapA.size() && new2orgSubdivideMapA[f].valid() )
                badFacesA_.autoResizeSet( new2orgSubdivideMapA[f] );
        }

        assert( badFacesA_.any() );
        errorStr = "Bad contour on " + std::to_string( badFacesA_.count() ) + " mesh A faces, " +
            "probably mesh B has self-intersections on contours lying on these faces.";
    }

    if ( cutB.fbsWithCountourIntersections.any() )
    {
        if ( badFacesA_.any() )
            errorStr += " ";

        for ( auto f : cutB.fbsWithCountourIntersections )
        {
            if ( f < new2orgSubdivideMapB.size() && new2orgSubdivideMapB[f].valid() )
                badFacesB_.autoResizeSet( new2orgSubdivideMapB[f] );
        }

        assert( badFacesB_.any() );

        errorStr += ( "Bad contour on " + std::to_string( badFacesB_.count() ) + " mesh B faces, " +
                    "probably mesh A has self-intersections on contours lying on these faces." );
    }
    if ( !errorStr.empty() )
        return errorStr;

    cutARes = std::move( cutA.resultCut );
    cutBRes = std::move( cutB.resultCut );

    assert( cutARes.size() == cutBRes.size() );
    return {};
}


// almost the same as MR::isInside but without collision check (it should be cheked already in this context)
bool isCompInside( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A )
{
    assert( b.mesh.topology.isClosed( b.region ) );

    auto aFace = a.mesh.topology.getFaceIds( a.region ).find_first();
    if ( !aFace )
        return true; //consider empty mesh always inside

    Vector3f aPoint = a.mesh.triCenter( aFace );
    if ( rigidB2A )
        aPoint = rigidB2A->inverse()( aPoint );

    auto signDist = b.mesh.signedDistance( aPoint, FLT_MAX );
    return signDist && signDist < 0;
}

// Finds need mesh part based on components relative positions (inside/outside)
// leftPart - left part of cut contours
FaceBitSet preparePart( const Mesh& origin, const std::vector<FaceBitSet>& components,
                        const FaceBitSet& leftPart,
                        const Mesh& otherMesh, 
                        bool needInsideComps,
                        bool originIsA, const AffineXf3f* rigidB2A )
{
    FaceBitSet res;
    FaceBitSet connectedComp;
    AffineXf3f a2b = rigidB2A ? rigidB2A->inverse() : AffineXf3f();
    bool needRightPart = needInsideComps != originIsA;
    for ( const auto& comp : components )
    {
        if ( !( comp & leftPart ).any() )
        {
            if ( isCompInside( {origin,&comp}, otherMesh, originIsA ? rigidB2A : &a2b ) == needInsideComps )
                res |= comp;
        }
        else if ( needRightPart )
        {
            connectedComp |= comp;
        }
    }
    if ( needRightPart )
        res |= ( connectedComp - leftPart );
    else
        res |= leftPart;
    return res;
}

bool prepareLeft( const Mesh& origin, const std::vector<EdgePath>& cutPaths, FaceBitSet& leftPart )
{
    const auto& fullBS = origin.topology.getValidFaces();
    leftPart = fillContourLeft( origin.topology, cutPaths );

    for ( const auto& path : cutPaths )
    {
        if ( path.empty() )
            continue;
        const auto& e0 = path[0];
        FaceId left = origin.topology.left( e0 );
        FaceId right = origin.topology.right( e0 );

        if ( fullBS.test( left ) && fullBS.test( right ) && leftPart.test( left ) && leftPart.test( right ) )
            return false;
    }

    return true;
}

// cutPaths - cut edges of origin mesh, it is modified to new indexes after preparing mesh part
// needInsidePart - part of origin that is inside otherMesh is needed
// needFlip - normals of needed part should be flipped
bool preparePart( const Mesh& origin, std::vector<EdgePath>& cutPaths, Mesh& outMesh,
                  const Mesh& otherMesh, bool needInsidePart, bool needFlip, bool originIsA,
                  const AffineXf3f* rigidB2A, BooleanResultMapper::Maps* maps )
{
    MR_TIMER;
    FaceBitSet leftPart;
    if ( !prepareLeft( origin, cutPaths, leftPart ) )
        return false;

    EdgeMap map;
    FaceMap* fMapPtr = maps ? &maps->cut2newFaces : nullptr;
    EdgeMap* eMapPtr = maps ? &maps->old2newEdges : &map;
    VertMap* vMapPtr = maps ? &maps->old2newVerts : nullptr;

    auto comps = MeshComponents::getAllComponents( origin, MeshComponents::FaceIncidence::PerVertex );
    leftPart = preparePart( origin, comps, leftPart, otherMesh, needInsidePart, originIsA, rigidB2A );


    outMesh.addPartByMask( origin, leftPart, needFlip, {}, {}, fMapPtr, vMapPtr, eMapPtr );

    for ( auto& path : cutPaths )
        for ( auto& e : path )
            e = ( *eMapPtr )[e];

    return true;
}

// transforms partB if needed and adds it to partA
// stitches parts by paths if they are not empty
// updates mapper
void connectPreparedParts( Mesh& partA, Mesh& partB, bool pathsHaveLeftHole,
                           const std::vector<EdgePath>& pathsA,
                           const std::vector<EdgePath>& pathsB,
                           const AffineXf3f* rigidB2A, BooleanResultMapper* mapper )
{
    if ( rigidB2A )
        partB.transform( *rigidB2A );

    FaceMap fMapNew;
    EdgeMap eMapNew;
    VertMap vMapNew;

    FaceMap* fMapNewPtr = mapper ? &fMapNew : nullptr;
    EdgeMap* eMapNewPtr = mapper ? &eMapNew : nullptr;
    VertMap* vMapNewPtr = mapper ? &vMapNew : nullptr;

    if ( pathsA.empty() )
        partA.addPart( partB, fMapNewPtr, vMapNewPtr, eMapNewPtr );
    else
    {
        if ( !pathsHaveLeftHole )
            partA.addPartByMask( partB, partB.topology.getValidFaces(), false, pathsA, pathsB, fMapNewPtr, vMapNewPtr, eMapNewPtr );
        else
            partB.addPartByMask( partA, partA.topology.getValidFaces(), false, pathsB, pathsA, fMapNewPtr, vMapNewPtr, eMapNewPtr );
    }

    if ( mapper )
    {
        int objectIndex = pathsHaveLeftHole ? int( BooleanResultMapper::MapObject::A ) : int( BooleanResultMapper::MapObject::B );
        FaceMap& fMap = mapper->maps[objectIndex].cut2newFaces;
        EdgeMap& eMap = mapper->maps[objectIndex].old2newEdges;
        VertMap& vMap = mapper->maps[objectIndex].old2newVerts;
        for ( int i = 0; i < fMap.size(); ++i )
            if ( fMap[FaceId( i )].valid() )
                fMap[FaceId( i )] = fMapNew[fMap[FaceId( i )]];
        for ( int i = 0; i < eMap.size(); ++i )
            if ( eMap[EdgeId( i )].valid() )
                eMap[EdgeId( i )] = eMapNew[eMap[EdgeId( i )]];
        for ( int i = 0; i < vMap.size(); ++i )
            if ( vMap[VertId( i )].valid() )
                vMap[VertId( i )] = vMapNew[vMap[VertId( i )]];
    }
}

//  Do boolean operation based only in relative positions of meshes components (inside/outside)
Mesh doTrivialBooleanOperation( const Mesh& meshACut, const Mesh& meshBCut, BooleanOperation operation, const AffineXf3f* rigidB2A, BooleanResultMapper* mapper )
{
    Mesh aPart, bPart;
    FaceBitSet aPartFbs, bPartFbs;
    std::vector<FaceBitSet> aComponents, bComponents;
    if ( operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB )
        aComponents = MeshComponents::getAllComponents( meshACut, MeshComponents::FaceIncidence::PerVertex );
    if ( operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA )
        bComponents = MeshComponents::getAllComponents( meshBCut, MeshComponents::FaceIncidence::PerVertex );

    if ( operation == BooleanOperation::OutsideA || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceAB )
        aPartFbs = preparePart( meshACut, aComponents, {}, meshBCut, false, true, rigidB2A );
    else if ( operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA )
        aPartFbs = preparePart( meshACut, aComponents, {}, meshBCut, true, true, rigidB2A );

    if ( operation == BooleanOperation::OutsideB || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceBA )
        bPartFbs = preparePart( meshBCut, bComponents, {}, meshACut, false, false, rigidB2A );
    else if ( operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB )
        bPartFbs = preparePart( meshBCut, bComponents, {}, meshACut, true, false, rigidB2A );

    if ( aPartFbs.count() != 0 )
    {
        FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2newFaces : nullptr;
        EdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newVerts : nullptr;

        aPart.addPartByMask( meshACut, aPartFbs, operation == BooleanOperation::DifferenceBA,
                             {}, {}, fMapPtr, vMapPtr, eMapPtr );
    }

    if ( bPartFbs.count() != 0 )
    {
        FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2newFaces : nullptr;
        EdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newVerts : nullptr;

        bPart.addPartByMask( meshBCut, bPartFbs, operation == BooleanOperation::DifferenceAB,
                             {}, {}, fMapPtr, vMapPtr, eMapPtr );

    }

    connectPreparedParts( aPart, bPart, false, {}, {}, rigidB2A, mapper );

    return aPart;
}

tl::expected<MR::Mesh, std::string> doBooleanOperation( 
    const Mesh& meshACutted, const Mesh& meshBCutted, 
    const std::vector<EdgePath>& cutEdgesA, const std::vector<EdgePath>& cutEdgesB,
    BooleanOperation operation, 
    const AffineXf3f* rigidB2A /*= nullptr */,
    BooleanResultMapper* mapper /*= nullptr */ )
{
    if ( cutEdgesA.size() == 0 && cutEdgesB.size() == 0 )
        return doTrivialBooleanOperation( meshACutted, meshBCutted, operation, rigidB2A, mapper );

    if ( operation == BooleanOperation::Intersection || operation == BooleanOperation::Union ||
         operation == BooleanOperation::DifferenceAB || operation == BooleanOperation::DifferenceBA )
        assert( cutEdgesA.size() == cutEdgesB.size() );

    MR_TIMER;
    Mesh aPart;
    Mesh bPart;
    bool dividableA{true};
    bool dividableB{true};
    auto pathsACpy = cutEdgesA;
    auto pathsBCpy = cutEdgesB;

    // aPart
    BooleanResultMapper::Maps* mapsAPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )] : nullptr;
    if ( operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA )
        dividableA = preparePart( meshACutted, pathsACpy, aPart, meshBCutted, true,
                                  operation == BooleanOperation::DifferenceBA, true, rigidB2A, mapsAPtr );
    else if ( operation == BooleanOperation::OutsideA || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceAB )
        dividableA = preparePart( meshACutted, pathsACpy, aPart, meshBCutted, false, false, true, rigidB2A, mapsAPtr );
    // bPart
    BooleanResultMapper::Maps* mapsBPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )] : nullptr;
    if ( operation == BooleanOperation::OutsideB || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceBA )
        dividableB = preparePart( meshBCutted, pathsBCpy, bPart, meshACutted, false, false, false, rigidB2A, mapsBPtr );
    else if ( operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB )
        dividableB = preparePart( meshBCutted, pathsBCpy, bPart, meshACutted, true,
                                  operation == BooleanOperation::DifferenceAB, false, rigidB2A, mapsBPtr );

    if ( ( ( operation == BooleanOperation::InsideA || operation == BooleanOperation::OutsideA ) && !dividableA ) ||
         ( ( operation == BooleanOperation::InsideB || operation == BooleanOperation::OutsideB ) && !dividableB ) ||
         ( ( operation == BooleanOperation::Union ||
             operation == BooleanOperation::Intersection ||
             operation == BooleanOperation::DifferenceAB ||
             operation == BooleanOperation::DifferenceBA ) && ( !dividableB || !dividableA ) ) )
    {
        std::string s;
        if ( !dividableA )
            s += "Cannot separate mesh A to inside and outside parts, probably contours on mesh A are not closed or are not consistent.";
        if ( !dividableB )
        {
            if ( !dividableA )
                s += " ";
            s += "Cannot separate mesh B to inside and outside parts, probably contours on mesh B are not closed or are not consistent.";
        }

        return tl::make_unexpected( s );
    }
    bool needStitch =
        operation != BooleanOperation::InsideA &&
        operation != BooleanOperation::OutsideA &&
        operation != BooleanOperation::InsideB &&
        operation != BooleanOperation::OutsideB;
    bool pathsHaveLeftHole = operation == BooleanOperation::Intersection;
    connectPreparedParts( aPart, bPart, pathsHaveLeftHole,
                          needStitch ? pathsACpy : std::vector<EdgePath>{}, 
                          needStitch ? pathsBCpy : std::vector<EdgePath>{}, 
                          rigidB2A, mapper );

    return pathsHaveLeftHole ? bPart : aPart;
}

FaceBitSet BooleanResultMapper::map( const FaceBitSet& oldBS, MapObject obj ) const
{
    if ( maps[int( obj )].identity )
        return oldBS;
    if ( maps[int( obj )].cut2newFaces.empty() )
        return {};
    FaceBitSet afterCutBS;
    for ( int i = 0; i < maps[int( obj )].cut2origin.size(); ++i )
        if ( oldBS.test( maps[int( obj )].cut2origin[FaceId( i )] ) )
            afterCutBS.autoResizeSet( FaceId( i ) );

    FaceBitSet res;
    for ( auto f : afterCutBS )
    {
        auto fn = maps[int( obj )].cut2newFaces[f];
        if ( fn.valid() )
            res.autoResizeSet( fn );
    }
    return res;
}

EdgeBitSet BooleanResultMapper::map( const EdgeBitSet& oldBS, MapObject obj ) const
{
    if ( maps[int( obj )].identity )
        return oldBS;
    if ( maps[int( obj )].old2newEdges.empty() )
        return {};
    EdgeBitSet res;
    for ( auto e : oldBS )
    {
        auto en = maps[int( obj )].old2newEdges[e];
        if ( en.valid() )
            res.autoResizeSet( en );
    }
    return res;
}

VertBitSet BooleanResultMapper::map( const VertBitSet& oldBS, MapObject obj ) const
{
    if ( maps[int( obj )].identity )
        return oldBS;
    if ( maps[int( obj )].old2newVerts.empty() )
        return {};
    VertBitSet res;
    for ( auto v : oldBS )
    {
        auto vn = maps[int( obj )].old2newVerts[v];
        if ( vn.valid() )
            res.autoResizeSet( vn );
    }
    return res;
}

}