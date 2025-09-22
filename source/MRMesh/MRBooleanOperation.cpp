#include "MRBooleanOperation.h"
#include "MRMesh.h"
#include "MRFillContour.h"
#include "MRContoursStitch.h"
#include "MRTimer.h"
#include "MRMeshComponents.h"
#include "MRMeshCollide.h"
#include "MRAffineXf3.h"
#include "MRMapEdge.h"
#include "MRPartMappingAdapters.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace
{

// Finds needed mesh part based on components relative positions (inside/outside)
// returns std::nullopt if given cuts do not divide origin mesh on good components (e.g. cuts have self-interections or components are not consistently oriented)
std::optional<FaceBitSet> findMeshPart( const Mesh& origin,
    const std::vector<EdgePath>& cutPaths, const Mesh& otherMesh, bool needInsideComps,
    bool originIsA, const AffineXf3f* rigidB2A,
    bool mergeAllNonIntersectingComponents, const BooleanInternalParameters& intParams )
{
    MR_TIMER;
    UnionFind<FaceId> unionFind;
    if ( cutPaths.empty() )
        unionFind = MeshComponents::getUnionFindStructureFaces( origin );
    else
    {
        UndirectedEdgeBitSet cutEdges( origin.topology.undirectedEdgeSize() );
        for ( const auto& path : cutPaths )
            for ( auto e : path )
                cutEdges.set( e );
        unionFind = MeshComponents::getUnionFindStructureFaces( origin, MeshComponents::PerEdge, &cutEdges );
    }

    FaceBitSet res( origin.topology.lastValidFace() + 1 );
    FaceBitSet connectedComp( origin.topology.lastValidFace() + 1 );
    AffineXf3f a2b = rigidB2A ? rigidB2A->inverse() : AffineXf3f();
    bool needRightPart = needInsideComps != originIsA;

    FaceId leftRoot;  // root of the components to the left of cutPaths
    FaceId rightRoot; // root of the components to the right of cutPaths
    if ( !cutPaths.empty() )
    {
        // unite regions separately to the left and to the right of cutPaths
        for ( const auto& path : cutPaths )
            for ( auto e : path )
            {
                if ( auto l = origin.topology.left( e ) )
                    leftRoot = leftRoot ? unionFind.unite( leftRoot, l ).first : unionFind.find( l );
                if ( auto r = origin.topology.right( e ) )
                    rightRoot = rightRoot ? unionFind.unite( rightRoot, r ).first : unionFind.find( r );
            }

        // if last unite merged left and right, we need to update roots
        if ( leftRoot )
            leftRoot = unionFind.find( leftRoot );
        if ( rightRoot )
            rightRoot = unionFind.find( rightRoot );

        if ( leftRoot && leftRoot == rightRoot )
            return std::nullopt;
    }

    // find correct part
    auto includeRoot = needRightPart ? rightRoot : leftRoot;
    auto excludeRoot = needRightPart ? leftRoot : rightRoot;
    for ( auto f : origin.topology.getValidFaces() )
    {
        if ( includeRoot && unionFind.united( includeRoot, f ) )
        {
            res.set( f );
        }
        else if ( excludeRoot && unionFind.united( excludeRoot, f ) )
        {
            //nothing
        }
        else
        {
            // a connected component without any cut
            const Mesh* otherPtr = originIsA ? intParams.originalMeshB : intParams.originalMeshA;
            if ( mergeAllNonIntersectingComponents ||
                isNonIntersectingInside( origin, f, otherPtr ? *otherPtr : otherMesh, originIsA ? rigidB2A : &a2b ) == needInsideComps )
            {
                includeRoot = includeRoot ? unionFind.unite( includeRoot, f ).first : unionFind.find( f );
                res.set( f );
            }
            else
            {
                excludeRoot = excludeRoot ? unionFind.unite( excludeRoot, f ).first : unionFind.find( f );
            }
        }
    }
    return res;
}

// Finds needed mesh part based on components relative positions (inside/outside)
FaceBitSet findMeshPart( const Mesh& origin,
    const Mesh& otherMesh, bool needInsideComps,
    bool originIsA, const AffineXf3f* rigidB2A,
    bool mergeAllNonIntersectingComponents, const BooleanInternalParameters& intParams )
{
    auto res = findMeshPart( origin, {}, otherMesh, needInsideComps, originIsA, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    assert( res.has_value() );
    return std::move( *res );
}

// cutPaths - cut edges of origin mesh, it is modified to new indexes after preparing mesh part
// needInsidePart - part of origin that is inside otherMesh is needed
// needFlip - normals of needed part should be flipped
bool preparePart( const Mesh& origin, std::vector<EdgePath>& cutPaths, Mesh& outMesh,
    const Mesh& otherMesh, bool needInsidePart, bool needFlip, bool originIsA,
    const AffineXf3f* rigidB2A, BooleanResultMapper::Maps* maps, 
    bool mergeAllNonIntersectingComponents, const BooleanInternalParameters& intParams )
{
    MR_TIMER;

    // use dense-maps inside addMeshPart instead of default hash-maps for better performance
    FaceMap fmap;
    WholeEdgeMap emap;
    VertMap vmap;

    FaceMap* fMapPtr = maps ? &maps->cut2newFaces : &fmap;
    WholeEdgeMap* eMapPtr = maps ? &maps->old2newEdges : &emap;
    VertMap* vMapPtr = maps ? &maps->old2newVerts : &vmap;

    auto maybeLeftPart = findMeshPart( origin, cutPaths, otherMesh, needInsidePart, originIsA, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    if ( !maybeLeftPart )
        return false;

    outMesh.addMeshPart( { origin, &*maybeLeftPart }, needFlip, {}, {}, Src2TgtMaps( fMapPtr, vMapPtr, eMapPtr ) );

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
    MR_TIMER;

    if ( rigidB2A )
        partB.transform( *rigidB2A );

    // use dense-maps inside addMesh(Part) instead of default hash-maps for better performance
    FaceMap fMapNew;
    WholeEdgeMap eMapNew;
    VertMap vMapNew;

    if ( pathsA.empty() )
        partA.addMesh( partB, &fMapNew, &vMapNew, &eMapNew );
    else
    {
        if ( !pathsHaveLeftHole )
            partA.addMeshPart( partB, false, pathsA, pathsB, Src2TgtMaps( &fMapNew, &vMapNew, &eMapNew ) );
        else
            partB.addMeshPart( partA, false, pathsB, pathsA, Src2TgtMaps( &fMapNew, &vMapNew, &eMapNew ) );
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
Mesh doTrivialBooleanOperation( Mesh&& meshACut, Mesh&& meshBCut, BooleanOperation operation, const AffineXf3f* rigidB2A, BooleanResultMapper* mapper, 
    bool mergeAllNonIntersectingComponents, const BooleanInternalParameters& intParams )
{
    MR_TIMER;

    tbb::task_group taskGroup;
    FaceBitSet aPartFbs;
    taskGroup.run( [&] ()
    {
        if ( operation == BooleanOperation::OutsideA || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceAB )
            aPartFbs = findMeshPart( meshACut, meshBCut, false, true, rigidB2A, mergeAllNonIntersectingComponents, intParams );
        else if ( operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA )
            aPartFbs = findMeshPart( meshACut, meshBCut, true, true, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    } );

    FaceBitSet bPartFbs;
    if ( operation == BooleanOperation::OutsideB || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceBA )
        bPartFbs = findMeshPart( meshBCut, meshACut, false, false, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    else if ( operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB )
        bPartFbs = findMeshPart( meshBCut, meshACut, true, false, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    taskGroup.wait();

    Mesh aPart;
    if ( aPartFbs.any() )
    {
        FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2newFaces : nullptr;
        WholeEdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )].old2newVerts : nullptr;

        aPart.addMeshPart( { meshACut, &aPartFbs }, operation == BooleanOperation::DifferenceBA,
                             {}, {}, Src2TgtMaps( fMapPtr, vMapPtr, eMapPtr ) );
    }

    Mesh bPart;
    if ( bPartFbs.any() )
    {
        FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2newFaces : nullptr;
        WholeEdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newEdges : nullptr;
        VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newVerts : nullptr;

        bPart.addMeshPart( { meshBCut, &bPartFbs }, operation == BooleanOperation::DifferenceAB,
                             {}, {}, Src2TgtMaps( fMapPtr, vMapPtr, eMapPtr ) );
    }

    connectPreparedParts( aPart, bPart, false, {}, {}, rigidB2A, mapper );

    return aPart;
}

} // anonymous namespace

Expected<MR::Mesh> doBooleanOperation(
    Mesh&& meshACut, Mesh&& meshBCut,
    const std::vector<EdgePath>& cutEdgesA, const std::vector<EdgePath>& cutEdgesB,
    BooleanOperation operation, 
    const AffineXf3f* rigidB2A /*= nullptr */,
    BooleanResultMapper* mapper /*= nullptr */, 
    bool mergeAllNonIntersectingComponents,
    const BooleanInternalParameters& intParams )
{
    if ( cutEdgesA.size() == 0 && cutEdgesB.size() == 0 )
        return doTrivialBooleanOperation( std::move( meshACut ), std::move( meshBCut ), operation, rigidB2A, mapper, mergeAllNonIntersectingComponents, intParams );

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
    tbb::task_group taskGroup;
    taskGroup.run( [&] ()
    {
        if ( operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA )
            dividableA = preparePart( meshACut, pathsACpy, aPart, meshBCut, true,
                                      operation == BooleanOperation::DifferenceBA, true, rigidB2A, mapsAPtr, mergeAllNonIntersectingComponents, intParams );
        else if ( operation == BooleanOperation::OutsideA || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceAB )
            dividableA = preparePart( meshACut, pathsACpy, aPart, meshBCut, false, false, true, rigidB2A, mapsAPtr, mergeAllNonIntersectingComponents, intParams );
    } );
    // bPart
    BooleanResultMapper::Maps* mapsBPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )] : nullptr;
    if ( operation == BooleanOperation::OutsideB || operation == BooleanOperation::Union || operation == BooleanOperation::DifferenceBA )
        dividableB = preparePart( std::move( meshBCut ), pathsBCpy, bPart, std::move( meshACut ), false, false, false, rigidB2A, mapsBPtr, mergeAllNonIntersectingComponents, intParams );
    else if ( operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB )
        dividableB = preparePart( std::move( meshBCut ), pathsBCpy, bPart, std::move( meshACut ), true,
                                  operation == BooleanOperation::DifferenceAB, false, rigidB2A, mapsBPtr, mergeAllNonIntersectingComponents, intParams );
    taskGroup.wait();

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

        return unexpected( s );
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

    if ( intParams.optionalOutCut )
    {
        if ( needStitch )
            *intParams.optionalOutCut = pathsHaveLeftHole ? std::move( pathsBCpy ) : std::move( pathsACpy );
        else
            *intParams.optionalOutCut = ( operation == BooleanOperation::InsideA || operation == BooleanOperation::OutsideA ) ? std::move( pathsACpy ) : std::move( pathsBCpy );
    }


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

UndirectedEdgeBitSet BooleanResultMapper::map( const UndirectedEdgeBitSet& oldBS, MapObject obj ) const
{
    if ( maps[int( obj )].identity )
        return oldBS;
    if ( maps[int( obj )].old2newEdges.empty() )
        return {};
    UndirectedEdgeBitSet res;
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
    FaceBitSet res;
    res.reserve( std::max( maps[0].cut2newFaces.size(), maps[1].cut2newFaces.size() ) );
    for ( const auto& map : maps )
    {
        for ( FaceId newF = 0_f; newF < map.cut2origin.size(); ++newF )
        {
            if ( newF == map.cut2origin[newF] || !map.cut2origin[newF].valid() )
                continue;
            if ( auto resF = map.cut2newFaces[newF] )
                res.autoResizeSet( resF );
        }
    }
    return res;
}

FaceBitSet BooleanResultMapper::filteredOldFaceBitSet( const FaceBitSet& oldBS, MapObject obj )
{
    const auto& map = maps[int( obj )];
    if ( map.identity )
        return oldBS;
    FaceBitSet outBs( oldBS.size() );
    for ( FaceId i = 0_f; i < map.cut2origin.size(); ++i )
    {
        auto orgF = map.cut2origin[i];
        if ( !orgF || !oldBS.test( orgF ) )
            continue;
        if ( map.cut2newFaces[i] )
            outBs.set( orgF );
    }
    return outBs;
}

} //namespace MR
