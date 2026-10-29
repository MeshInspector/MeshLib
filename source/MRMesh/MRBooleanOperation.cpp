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
#include "MRUnionFindParallel.h"
#include "MRphmap.h"
#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRPrecisePredicates3.h"

namespace MR
{

namespace
{

// the exact winding numbers of the other mesh at the points of origin
struct OtherMeshWindings
{
    const Mesh& origin;
    const MeshPart& other;
    const CoordinateConverters& conv;
    bool originIsA = true;
    const AffineXf3f* rigidB2A = nullptr;
    int bVertShift = 0;

    // the number of the triangles of other that the segment from point p of origin (in its space) up above both meshes crosses from their
    // back to their front, minus those it crosses from front to back, with p vertex v of origin that the cut did not add, or no vertex of
    // either mesh if v is invalid; by the precise predicates with the coordinates and (in their order) the vertex ids of the cut,
    // so that it agrees with the paths
    int operator()( VertId v, const Vector3f& p ) const;
};

int OtherMeshWindings::operator()( VertId v, const Vector3f& p ) const
{
    const auto* xfOrigin = originIsA ? nullptr : rigidB2A;
    const auto* xfOther = originIsA ? rigidB2A : nullptr;
    std::array<PreciseVertCoords, 5> vs; // a triangle of other, the point, the point above
    // an id above the vertices of both meshes, below the one of the point above, for no vertex
    vs[3].id = v ? VertId( int( v ) + ( originIsA ? 0 : bVertShift ) ) : VertId( std::numeric_limits<int>::max() - 1 );
    vs[3].pt = conv.toInt( xfOrigin ? ( *xfOrigin )( p ) : p );
    vs[4].id = VertId( std::numeric_limits<int>::max() );
    // the int coordinates of the points of both meshes are within the half of 0.99 of the int range
    vs[4].pt = Vector3i( vs[3].pt.x, vs[3].pt.y, std::numeric_limits<int>::max() / 2 );

    const auto& tree = other.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return 0;
    // the int coordinates of the transformed points are within the transformed box in int units (not rounded, as the box can be
    // larger than the range of the int coordinates), the rounding being monotone; with a margin of a few float steps for safety
    const auto rootBox = transformed( tree[tree.rootNodeId()].box, xfOther );
    const double margin = 1 + 4 * FLT_EPSILON * conv.toInt.invRange *
        std::max( { std::abs( rootBox.min.x ), std::abs( rootBox.min.y ), std::abs( rootBox.min.z ),
                    std::abs( rootBox.max.x ), std::abs( rootBox.max.y ), std::abs( rootBox.max.z ) } );
    int res = 0;
    InplaceStack<NoInitNodeId, 32> subtasks;
    subtasks.push( tree.rootNodeId() );
    while ( !subtasks.empty() )
    {
        const auto& node = tree[subtasks.top()];
        subtasks.pop();
        const auto box = transformed( node.box, xfOther );
        const auto lo = ( Vector3d( box.min ) - conv.toInt.center ) * conv.toInt.invRange;
        const auto hi = ( Vector3d( box.max ) - conv.toInt.center ) * conv.toInt.invRange;
        if ( vs[3].pt.x < lo.x - margin || vs[3].pt.x > hi.x + margin || vs[3].pt.y < lo.y - margin || vs[3].pt.y > hi.y + margin
            || vs[3].pt.z > hi.z + margin )
            continue;
        if ( !node.leaf() )
        {
            subtasks.push( node.l );
            subtasks.push( node.r );
            continue;
        }
        const auto f = node.leafId();
        if ( other.region && !other.region->test( f ) )
            continue;
        other.mesh.topology.getTriVerts( f, vs[0].id, vs[1].id, vs[2].id );
        for ( int i = 0; i < 3; ++i )
        {
            const auto& q = other.mesh.points[vs[i].id];
            vs[i].pt = conv.toInt( xfOther ? ( *xfOther )( q ) : q );
            vs[i].id = VertId( int( vs[i].id ) + ( originIsA ? bVertShift : 0 ) );
        }
        if ( const auto x = doTriangleSegmentIntersect( vs ) )
            res += x.dIsLeftFromABC ? 1 : -1;
    }
    return res;
}

// unites the parts of origin separated by the cut (by their roots in unionFind) by the winding numbers of the other mesh around them
// (crossing a path from its left to its right leaves the other mesh on mesh A, and enters it on mesh B): the ones inside the other mesh
// in leftRoot on mesh A and in rightRoot on mesh B, the others in the other root;
// every set of parts linked by the paths is numbered from its part with the most path edges among those with a vertex not on the paths,
// whose number is the exact winding number there if given windings (on a closed other mesh); otherwise (or if no part of the set
// has such a vertex) the set is numbered from 0 at its lowest part;
// a part reached with two numbers (as by crossings in a face not reported bad, which are not split) keeps the first one;
// unites nothing and returns false if every part is only to the left or only to the right of the paths (as where the other mesh
// does not cross itself)
bool uniteByWindings( const MeshTopology& topology, const std::vector<EdgePath>& cutPaths, BaseUnionFind<FaceId>& unionFind, bool originIsA,
    const OtherMeshWindings* exactWindings, FaceId& leftRoot, FaceId& rightRoot )
{
    MR_TIMER;
    FaceBitSet leftParts( topology.faceSize() ), rightParts( topology.faceSize() );
    for ( const auto& path : cutPaths )
        for ( auto e : path )
            if ( const auto l = topology.left( e ), r = topology.right( e ); l && r )
            {
                leftParts.set( unionFind.find( l ) );
                rightParts.set( unionFind.find( r ) );
            }
    if ( !leftParts.intersects( rightParts ) )
        return false;

    if ( exactWindings && !exactWindings->other.mesh.topology.isClosed( exactWindings->other.region ) )
        exactWindings = nullptr; // an open mesh has no winding numbers

    // the vertices that the cut added are on the paths
    VertBitSet pathVerts;
    if ( exactWindings )
    {
        pathVerts.resize( topology.vertSize() );
        for ( const auto& path : cutPaths )
            for ( auto e : path )
            {
                pathVerts.set( topology.org( e ) );
                pathVerts.set( topology.dest( e ) );
            }
    }

    HashMap<FaceId, Vector<std::pair<FaceId, int>>> links; // the parts across every path edge, with the change of the number
    HashMap<FaceId, VertId> probes; // a vertex not on the paths, of a face next to them, in the parts having one
    const int step = originIsA ? -1 : 1;
    for ( const auto& path : cutPaths )
        for ( auto e : path )
            if ( const auto l = topology.left( e ), r = topology.right( e ); l && r )
            {
                const auto lr = unionFind.find( l ), rr = unionFind.find( r );
                links[lr].emplace_back( rr, step );
                links[rr].emplace_back( lr, -step );
                if ( !exactWindings )
                    continue;
                if ( const auto v = topology.dest( topology.next( e ) ); !pathVerts.test( v ) )
                    probes.insert( { lr, v } );
                if ( const auto v = topology.dest( topology.next( e.sym() ) ); !pathVerts.test( v ) )
                    probes.insert( { rr, v } );
            }

    Vector<std::tuple<bool, size_t, FaceId>> starts; // has a probe, the number of path edges, the part
    starts.reserve( links.size() );
    for ( const auto& [x, xLinks] : links )
        starts.emplace_back( probes.contains( x ), xLinks.size(), x );
    if ( exactWindings )
        std::sort( starts.begin(), starts.end(), std::greater() );

    // a part on both sides of the paths makes a number above 1 in its set
    struct LinkedSet
    {
        size_t end = 0; // of its parts in linked
        int shift = 0; // to the winding numbers
        VertId probe; // of its start, which has number 0
    };
    Vector<LinkedSet> sets;
    Vector<std::pair<FaceId, int>> linked; // the parts of all sets with their numbers, set after set
    FaceBitSet numbered( topology.faceSize() );
    for ( const auto& [probed, unused, start] : starts )
    {
        if ( numbered.test_set( start ) )
            continue;
        linked.emplace_back( start, 0 );
        int minW = 0;
        for ( size_t i = sets.empty() ? 0 : sets.back().end; i < linked.size(); ++i )
        {
            const auto [x, wx] = linked[i];
            minW = std::min( minW, wx );
            for ( const auto& [y, d] : links.at( x ) )
                if ( !numbered.test_set( y ) )
                    linked.emplace_back( y, wx + d );
        }
        sets.push_back( { .end = linked.size(), .shift = -minW, .probe = probed ? probes.at( start ) : VertId{} } );
    }
    if ( exactWindings )
        ParallelFor( sets, [&] ( size_t i )
        {
            if ( sets[i].probe )
                sets[i].shift = ( *exactWindings )( sets[i].probe, exactWindings->origin.points[sets[i].probe] );
        } );

    size_t i = 0;
    for ( const auto& s : sets )
        for ( ; i < s.end; ++i )
        {
            const auto [x, w] = linked[i];
            auto& root = ( w + s.shift > 0 ) == originIsA ? leftRoot : rightRoot;
            root = root ? unionFind.uniteUnbalanced( root, x ).first : unionFind.find( x );
        }
    return true;
}

// Finds needed mesh part based on components relative positions (inside/outside)
// returns std::nullopt if given cuts do not divide origin mesh on good components (e.g. cuts have self-interections or components are not consistently oriented)
std::optional<FaceBitSet> findMeshPart( const Mesh& origin,
    const std::vector<EdgePath>& cutPaths, const Mesh& otherMesh, bool needInsideComps,
    bool originIsA, const AffineXf3f* rigidB2A,
    bool mergeAllNonIntersectingComponents, const BooleanInternalParameters& intParams,
    int bVertShift )
{
    MR_TIMER;
    BaseUnionFind<FaceId> unionFind;
    if ( cutPaths.empty() )
        unionFind = MeshComponents::getUnionFindStructureFaces( origin );
    else
    {
        UndirectedEdgeBitSet cutEdges( origin.topology.undirectedEdgeSize() );
        for ( const auto& path : cutPaths )
            for ( auto e : path )
                cutEdges.set( e );
        // under force, the graph cut completes open paths (as on a mesh with holes); closed paths alone bound the parts,
        // and the winding numbers below decide the parts on both sides of them
        if ( intParams.graphCutSeparation && std::any_of( cutPaths.begin(), cutPaths.end(),
            [&] ( const EdgePath& p ) { return !p.empty() && origin.topology.org( p.front() ) != origin.topology.dest( p.back() ); } ) )
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
    const Mesh* otherPtr = originIsA ? intParams.originalMeshB : intParams.originalMeshA;
    const MeshPart other( otherPtr ? *otherPtr : otherMesh );

    std::optional<OtherMeshWindings> exactWindings;
    if ( intParams.converters )
        exactWindings.emplace( OtherMeshWindings{ origin, other, *intParams.converters, originIsA, rigidB2A, bVertShift } );

    FaceId leftRoot;  // root of the components to the left of cutPaths
    FaceId rightRoot; // root of the components to the right of cutPaths
    if ( !cutPaths.empty() )
    {
        // unite regions separately to the left and to the right of cutPaths
        // (by the winding numbers where the forced cut splits crossing contours, as where the other mesh crosses itself)
        if ( !intParams.graphCutSeparation ||
            !uniteByWindings( origin.topology, cutPaths, unionFind, originIsA, exactWindings ? &*exactWindings : nullptr, leftRoot, rightRoot ) )
            for ( const auto& path : cutPaths )
                for ( auto e : path )
                {
                    if ( auto l = origin.topology.left( e ) )
                        leftRoot = leftRoot ? unionFind.uniteUnbalanced( leftRoot, l ).first : unionFind.find( l );
                    if ( auto r = origin.topology.right( e ) )
                        rightRoot = rightRoot ? unionFind.uniteUnbalanced( rightRoot, r ).first : unionFind.find( r );
                }

        // if last unite merged left and right, we need to update roots
        if ( leftRoot )
            leftRoot = unionFind.find( leftRoot );
        if ( rightRoot )
            rightRoot = unionFind.find( rightRoot );

        if ( leftRoot && leftRoot == rightRoot )
            return std::nullopt;
    }

    updateRootsParallel( unionFind );

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
            if ( mergeAllNonIntersectingComponents ||
                ( intParams.converters ?
                    isNonIntersectingInsidePrecise( origin, f, other, *intParams.converters,
                        originIsA ? 0 : bVertShift, originIsA ? bVertShift : 0,
                        originIsA ? nullptr : rigidB2A, originIsA ? rigidB2A : nullptr ) :
                    isNonIntersectingInside( origin, f, other, originIsA ? rigidB2A : &a2b ) ) == needInsideComps )
            {
                includeRoot = includeRoot ? unionFind.uniteUnbalanced( includeRoot, f ).first : unionFind.find( f );
                res.set( f );
            }
            else
            {
                excludeRoot = excludeRoot ? unionFind.uniteUnbalanced( excludeRoot, f ).first : unionFind.find( f );
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

    // the vertex ids of mesh B are shifted by this value in the precise predicates to make them
    // distinct from the ids of mesh A, equally in both calls of findMeshPart below
    const int bVertShift = int( std::max( meshACut.topology.vertSize(),
        intParams.originalMeshA ? intParams.originalMeshA->topology.vertSize() : size_t( 0 ) ) );

    // bPart
    tbb::task_group taskGroup;
    taskGroup.run( [&] ()
    {
        if ( onlyCutA )
            return;
        bPart = findMeshPart( meshBCut, cutEdgesB, meshACut, needInsideB, false, rigidB2A, mergeAllNonIntersectingComponents, intParams, bVertShift );
    } );
    // aPart
    if ( !onlyCutB )
        aPart = findMeshPart( meshACut, cutEdgesA, meshBCut, needInsideA, true, rigidB2A, mergeAllNonIntersectingComponents, intParams, bVertShift );
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
    FaceMap outMap( maxNewFace + 1 );
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
