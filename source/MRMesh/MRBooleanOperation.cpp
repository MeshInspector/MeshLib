#include "MRBooleanOperation.h"
#include "MRMesh.h"
#include "MRFillContour.h"
#include "MRContoursStitch.h"
#include "MRTimer.h"
#include "MRMeshComponents.h"
#include "MRMeshCollide.h"
#include "MRAffineXf3.h"

namespace MR
{

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

    auto signDist = b.mesh.signedDistance( aPoint, FLT_MAX, b.region );
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

    WholeEdgeMap map;
    FaceMap* fMapPtr = maps ? &maps->cut2newFaces : nullptr;
    WholeEdgeMap* eMapPtr = maps ? &maps->old2newEdges : &map;
    VertMap* vMapPtr = maps ? &maps->old2newVerts : nullptr;

    auto comps = MeshComponents::getAllComponents( origin );
    leftPart = preparePart( origin, comps, leftPart, otherMesh, needInsidePart, originIsA, rigidB2A );

    outMesh.addPartByMask( origin, leftPart, needFlip, {}, {}, 
        HashToVectorMappingConverter( origin.topology, fMapPtr, vMapPtr, eMapPtr ).getPartMapping() );

    for ( auto& path : cutPaths )
        for ( auto& e : path )
            e = mapEdge( *eMapPtr, e );

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
    WholeEdgeMap eMapNew;
    VertMap vMapNew;

    FaceMap* fMapNewPtr = mapper ? &fMapNew : nullptr;
    WholeEdgeMap* eMapNewPtr = mapper ? &eMapNew : nullptr;
    VertMap* vMapNewPtr = mapper ? &vMapNew : nullptr;

    if ( pathsA.empty() )
        partA.addPart( partB, fMapNewPtr, vMapNewPtr, eMapNewPtr );
    else
    {
        if ( !pathsHaveLeftHole )
            partA.addPartByMask( partB, partB.topology.getValidFaces(), false, pathsA, pathsB, 
                HashToVectorMappingConverter( partB.topology, fMapNewPtr, vMapNewPtr, eMapNewPtr ).getPartMapping() );
        else
            partB.addPartByMask( partA, partA.topology.getValidFaces(), false, pathsB, pathsA,
                HashToVectorMappingConverter( partA.topology, fMapNewPtr, vMapNewPtr, eMapNewPtr ).getPartMapping() );
    }

    if ( mapper )
    {
        int objectIndex = pathsHaveLeftHole ? int( BooleanResultMapper::MapObject::A ) : int( BooleanResultMapper::MapObject::B );
        FaceMap& fMap = mapper->maps[objectIndex].cut2newFaces;
        WholeEdgeMap& eMap = mapper->maps[objectIndex].old2newEdges;
        VertMap& vMap = mapper->maps[objectIndex].old2newVerts;
        for ( int i = 0; i < fMap.size(); ++i )
            if ( fMap[FaceId( i )].valid() )
                fMap[FaceId( i )] = fMapNew[fMap[FaceId( i )]];
        for ( int i = 0; i < eMap.size(); ++i )
            if ( eMap[UndirectedEdgeId( i )].valid() )
                eMap[UndirectedEdgeId( i )] = mapEdge( eMapNew, mapEdge( eMap, UndirectedEdgeId( i ) ) );
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
        aComponents = MeshComponents::getAllComponents( meshACut );
    if ( operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA )
        bComponents = MeshComponents::getAllComponents( meshBCut );

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
        WholeEdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newVerts : nullptr;

        aPart.addPartByMask( meshACut, aPartFbs, operation == BooleanOperation::DifferenceBA,
                             {}, {}, HashToVectorMappingConverter( meshACut.topology, fMapPtr, vMapPtr, eMapPtr ).getPartMapping() );
    }

    if ( bPartFbs.count() != 0 )
    {
        FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2newFaces : nullptr;
        WholeEdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newVerts : nullptr;

        bPart.addPartByMask( meshBCut, bPartFbs, operation == BooleanOperation::DifferenceAB,
                             {}, {}, HashToVectorMappingConverter( meshBCut.topology, fMapPtr, vMapPtr, eMapPtr ).getPartMapping() );
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
        auto en = mapEdge( maps[int( obj )].old2newEdges, e );
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

FaceBitSet BooleanResultMapper::newFaces() const
{
    FaceBitSet res( std::max( maps[0].cut2newFaces.size(), maps[1].cut2newFaces.size() ) );
    for ( const auto& map : maps )
    {
        for ( FaceId newF = 0_f; newF < map.cut2origin.size(); ++newF )
        {
            if ( newF == map.cut2origin[newF] )
                continue;
            if ( auto resF = map.cut2newFaces[newF] )
                res.autoResizeSet( resF );
        }
    }
    return res;
}

} //namespace MR
