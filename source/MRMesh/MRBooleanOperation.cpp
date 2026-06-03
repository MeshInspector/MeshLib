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
#include "MRParallelFor.h"
#include "MRPch/MRTBB.h"
#include "MRFillContourByGraphCut.h"
#include "MREdgeMetric.h"
#include "MRRegionBoundary.h"
#include "MREdgeIterator.h"

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
        if ( intParams.graphCutSeparation )
        {
            auto left = fillContourLeftByGraphCut( origin.topology, cutPaths, edgeAbsCurvMetric( origin ) );
            cutEdges |= findRegionBoundaryUndirectedEdgesInsideMesh( origin.topology, left );
        }
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

/// deletes unnecessary part from the mesh
VacantElements preparePart( Mesh& mesh, const FaceBitSet& leftPart,
    bool needFlip, BooleanResultMapper::Maps* maps )
{
    MR_TIMER;

    auto rightPart = mesh.topology.getValidFaces() - leftPart;
    auto res = mesh.deleteFaces( rightPart );
    if ( needFlip )
        mesh.topology.flipOrientation();

    if ( maps )
    {
        auto& fmap = maps->cut2newFaces;
        if ( fmap.size() < mesh.topology.faceSize() )
            fmap.resize( mesh.topology.faceSize() );
        for ( auto f : mesh.topology.getValidFaces() )
            fmap[f] = f;

        auto& vmap = maps->old2newVerts;
        if ( vmap.size() < mesh.topology.vertSize() )
            vmap.resize( mesh.topology.vertSize() );
        for ( auto v : mesh.topology.getValidVerts() )
            vmap[v] = v;

        auto& emap = maps->old2newEdges;
        if ( emap.size() < mesh.topology.undirectedEdgeSize() )
            emap.resize( mesh.topology.undirectedEdgeSize() );
        for ( auto ue : undirectedEdges( mesh.topology ) )
            emap[ue] = ue;
    }
    return res;
}

// transforms partB if needed and adds it to partA
// stitches parts by paths if they are not empty
// updates mapper
void connectPreparedParts( Mesh& res, Mesh& partB, const FaceBitSet* bRegion, bool flipB,
                           const std::vector<EdgePath>& pathsA, const std::vector<EdgePath>& pathsB,
                           const AffineXf3f* rigidB2A, BooleanResultMapper* mapper, bool graphCut, VacantElements& vacant )
{
    MR_TIMER;

    if ( rigidB2A )
        partB.transform( *rigidB2A );

    // use dense-maps inside addMesh(Part) instead of default hash-maps for better performance
    FaceMap fmap;
    WholeEdgeMap emap;
    VertMap vmap;

    FaceMap* fMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2newFaces : &fmap;
    WholeEdgeMap* eMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newEdges : &emap;
    VertMap* vMapPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::B )].old2newVerts : &vmap;

    if ( !graphCut )
    {
        res.addMeshPart( { partB,bRegion }, flipB, pathsA, pathsB, Src2TgtMaps( fMapPtr, vMapPtr, eMapPtr ), &vacant );
    }
    else
    {
        res.addMeshPart( { partB,bRegion }, flipB, {}, {}, Src2TgtMaps( fMapPtr, vMapPtr, eMapPtr ), &vacant );
    }
}

} // anonymous namespace

Expected<MR::Mesh> doBooleanOperation(
    Mesh&& meshACut, Mesh&& meshBCut,
    std::vector<EdgePath>&& cutEdgesA, std::vector<EdgePath>&& cutEdgesB,
    BooleanOperation operation, 
    const AffineXf3f* rigidB2A /*= nullptr */,
    BooleanResultMapper* mapper /*= nullptr */, 
    bool mergeAllNonIntersectingComponents,
    const BooleanInternalParameters& intParams )
{
    MR_TIMER;

    std::optional<FaceBitSet> aPart;
    std::optional<FaceBitSet> bPart;

    bool needInsideA = operation == BooleanOperation::InsideA || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceBA;
    bool needFlipA = operation == BooleanOperation::DifferenceBA;
    bool needInsideB = operation == BooleanOperation::InsideB || operation == BooleanOperation::Intersection || operation == BooleanOperation::DifferenceAB;
    bool needFlipB = operation == BooleanOperation::DifferenceAB;
    bool onlyCutA = operation == BooleanOperation::InsideA || operation == BooleanOperation::OutsideA;
    bool onlyCutB = operation == BooleanOperation::InsideB || operation == BooleanOperation::OutsideB;
    bool needStitch = !onlyCutA && !onlyCutB;
    if ( needStitch )
        assert( cutEdgesA.size() == cutEdgesB.size() );

    // bPart
    tbb::task_group taskGroup;
    taskGroup.run( [&] ()
    {
        if ( onlyCutA )
            return;
        bPart = findMeshPart( meshBCut, cutEdgesB, meshACut, needInsideB, false, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    } );
    // aPart
    if ( !onlyCutB )
        aPart = findMeshPart( meshACut, cutEdgesA, meshBCut, needInsideA, true, rigidB2A, mergeAllNonIntersectingComponents, intParams );
    taskGroup.wait();

    if ( ( onlyCutA && !aPart ) ||
         ( onlyCutB && !bPart ) ||
         ( ( needStitch ) && ( !bPart || !aPart ) ) )
    {
        std::string s;
        if ( !aPart && !onlyCutB )
            s += "Cannot separate mesh A to inside and outside parts, probably contours on mesh A are not closed or are not consistent.";
        if ( !bPart && !onlyCutA )
        {
            if ( !aPart && !onlyCutB )
                s += " ";
            s += "Cannot separate mesh B to inside and outside parts, probably contours on mesh B are not closed or are not consistent.";
        }

        return unexpected( s );
    }

    // no need update maps since we will update it during stitch
    BooleanResultMapper::Maps* mapsBPtr = ( onlyCutB && mapper ) ? &mapper->maps[int( BooleanResultMapper::MapObject::B )] : nullptr;
    taskGroup.run( [&] ()
    {
        if ( onlyCutA )
            return;
        preparePart( meshBCut, *bPart, false, mapsBPtr );
    } );
    BooleanResultMapper::Maps* mapsAPtr = mapper ? &mapper->maps[int( BooleanResultMapper::MapObject::A )] : nullptr;
    VacantElements vacant;
    if ( !onlyCutB )
        vacant = preparePart( meshACut, *aPart, needFlipA, mapsAPtr );
    taskGroup.wait();

    if ( needStitch && operation == BooleanOperation::Intersection )
    {
        taskGroup.run( [&] ()
        {
            MR::reverse( cutEdgesA );
        } );
        MR::reverse( cutEdgesB );
        taskGroup.wait();
    }

    auto&& res = onlyCutB ? meshBCut : meshACut;

    if ( !needStitch )
    {
        if ( onlyCutB && rigidB2A )
            res.transform( *rigidB2A );
    }
    else
    {
        connectPreparedParts( res, meshBCut, &*bPart, needFlipB, cutEdgesA, cutEdgesB, rigidB2A, mapper, intParams.graphCutSeparation, vacant );
    }

    if ( intParams.optionalOutCut )
    {
        if ( needStitch )
            *intParams.optionalOutCut = std::move( cutEdgesA );
        else
            *intParams.optionalOutCut = onlyCutA ? std::move( cutEdgesA ) : std::move( cutEdgesB );
    }

    return std::move( res );
}

FaceBitSet BooleanResultMapper::map( const FaceBitSet& oldBS, MapObject obj ) const
{
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

FaceBitSet BooleanResultMapper::filteredOldFaceBitSet( const FaceBitSet& oldBS, MapObject obj ) const
{
    const auto& map = maps[int( obj )];
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

FaceMap BooleanResultMapper::getNew2OldFaceMap( MapObject obj ) const
{
    const auto& map = maps[int( obj )];
    size_t maxNewFace = 0;
    // find last "new face" for given obj part
    maxNewFace = tbb::parallel_reduce( tbb::blocked_range( size_t( 0 ), map.cut2origin.size() ), size_t( 0 ),
        [&map] ( const auto& range, auto curr )
    {
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            FaceId cf = FaceId( i );
            auto of = map.cut2origin[cf];
            if ( !of )
                continue;
            auto nf = cf < map.cut2newFaces.size() ? map.cut2newFaces[cf] : FaceId();
            if ( !nf )
                continue;
            curr = std::max( curr, size_t( nf ) );
        }
        return curr;
    }, [] ( auto a, auto b )
    {
        return std::max( a, b );
    } );

    // fill map in parallel
    FaceMap outMap( maxNewFace );
    ParallelFor( map.cut2origin, [&] ( FaceId cf )
    {
        auto of = map.cut2origin[cf];
        if ( !of )
            return;
        auto nf = cf < map.cut2newFaces.size() ? map.cut2newFaces[cf] : FaceId();
        if ( !nf )
            return;
        outMap[nf] = of;
    } );
    return outMap;
}

} //namespace MR
