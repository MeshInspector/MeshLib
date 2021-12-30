#include "MRMeshCollidePrecise.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRPrecisePredicates3.h"
#include "MRFaceFace.h"
#include "MRTimer.h"
#include <tbb/parallel_for.h>
#include <array>

namespace
{
// INT_MAX in double for mapping in int range
constexpr double cRangeIntMax = 0.99 * std::numeric_limits<int>::max(); // 0.99 to be sure the no overflow will ever happen due to rounding errors
}

namespace MR
{

struct NodeNode
{
    AABBTree::NodeId aNode;
    AABBTree::NodeId bNode;
    NodeNode( AABBTree::NodeId a, AABBTree::NodeId b ) : aNode( a ), bNode( b ) { }
};

PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const MeshPart & b, 
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    MR_TIMER;

    PreciseCollisionResult res;
    const AABBTree & aTree = a.mesh.getAABBTree();
    const AABBTree & bTree = b.mesh.getAABBTree();
    if ( aTree.nodes().empty() || bTree.nodes().empty() )
        return res;

    std::vector<NodeNode> subtasks{ { AABBTree::NodeId{ 0 }, AABBTree::NodeId{ 0 } } };
    std::vector<FaceFace> collidingBoxes;

    while( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        const auto & aNode = aTree[s.aNode];
        const auto & bNode = bTree[s.bNode];

        const auto overlap = aNode.box.intersection( transformed( bNode.box, rigidB2A ) );
        if ( !overlap.valid() )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            const auto aFace = aNode.leafId();
            if ( a.region && !a.region->test( aFace ) )
                continue;
            const auto bFace = bNode.leafId();
            if ( b.region && !b.region->test( bFace ) )
                continue;
            collidingBoxes.emplace_back( aFace, bFace );
            continue;
        }
        
        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            subtasks.emplace_back( aNode.l, s.bNode );
            subtasks.emplace_back( aNode.r, s.bNode );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            subtasks.emplace_back( s.aNode, bNode.l );
            subtasks.emplace_back( s.aNode, bNode.r );
        }
    }

    // there can be at most two triangle-edge intersections in a triangle pair
    res.edgesAtrisB.resize( collidingBoxes.size() * 2 );
    res.edgesBtrisA.resize( collidingBoxes.size() * 2 );

    // we do not check an edge if its right triangle has smaller index and also in the mesh part
    auto checkEdge = [&]( EdgeId e, const MeshPart & mp )
    {
        const auto r = mp.mesh.topology.right( e );
        if ( !r )
            return true;
        if ( mp.region && !mp.region->test( r ) )
            return true;

        const auto l = mp.mesh.topology.left( e );
        assert ( l );
        return l < r;
    };

    const int aVertsSize = (int)a.mesh.topology.vertSize();
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)collidingBoxes.size() ),
        [&]( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            const auto aTri = collidingBoxes[i].aFace;
            const auto bTri = collidingBoxes[i].bFace;

            PreciseVertCoords avc[3], bvc[3];
            a.mesh.topology.getTriVerts( aTri, avc[0].id, avc[1].id, avc[2].id );
            b.mesh.topology.getTriVerts( bTri, bvc[0].id, bvc[1].id, bvc[2].id );

            for ( int j = 0; j < 3; ++j )
            {
                avc[j].pt = conv( a.mesh.points[avc[j].id] );
                const auto bf = b.mesh.points[bvc[j].id];
                bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
                bvc[j].id += aVertsSize;
            }

            // check edges from A
            int numA = 0;
            EdgeId aEdge = a.mesh.topology.edgeWithLeft( aTri );
            auto aEdgeCheck = [&]( int v0, int v1 )
            {
                if ( !checkEdge( aEdge, a ) )
                    return EdgeId{};
                auto isect = doTriangleSegmentIntersect( { bvc[0], bvc[1], bvc[2], avc[v0], avc[v1] } );
                if ( !isect )
                    return EdgeId{};
                return isect.dIsLeftFromABC ? aEdge : aEdge.sym();
            };
            if ( auto e = aEdgeCheck( 0, 1 ) )
            {
                res.edgesAtrisB[ 2*i + numA++ ] = EdgeTri( e, bTri );
            }
            aEdge = a.mesh.topology.prev( aEdge.sym() );
            if ( auto e = aEdgeCheck( 1, 2 ) )
            {
                res.edgesAtrisB[ 2*i + numA++ ] = EdgeTri( e, bTri );
            }
            aEdge = a.mesh.topology.prev( aEdge.sym() );
            if ( numA < 2 )
            {
                if ( auto e = aEdgeCheck( 2, 0 ) )
                    res.edgesAtrisB[ 2*i + numA++ ] = EdgeTri( e, bTri );
            }

            // check edges from B
            int numB = 0;
            EdgeId bEdge = b.mesh.topology.edgeWithLeft( bTri );
            auto bEdgeCheck = [&]( int v0, int v1 )
            {
                if ( !checkEdge( bEdge, b ) )
                    return EdgeId{};
                auto isect = doTriangleSegmentIntersect( { avc[0], avc[1], avc[2], bvc[v0], bvc[v1] } );
                if ( !isect )
                    return EdgeId{};
                return isect.dIsLeftFromABC ? bEdge : bEdge.sym();
            };
            if ( auto e = bEdgeCheck( 0, 1 ) )
            {
                res.edgesBtrisA[ 2*i + numB++ ] = EdgeTri( e, aTri );
            }
            bEdge = b.mesh.topology.prev( bEdge.sym() );
            if ( auto e = bEdgeCheck( 1, 2 ) )
            {
                res.edgesBtrisA[ 2*i + numB++ ] = EdgeTri( e, aTri );
            }
            bEdge = b.mesh.topology.prev( bEdge.sym() );
            if ( numB < 2 )
            {
                if ( auto e = bEdgeCheck( 2, 0 ) )
                    res.edgesBtrisA[ 2*i + numB++ ] = EdgeTri( e, aTri );
            }
        }
    } );

    res.edgesAtrisB.erase( std::remove_if( res.edgesAtrisB.begin(), res.edgesAtrisB.end(), []( const EdgeTri & et ) { return !et.edge.valid(); } ), res.edgesAtrisB.end() );
    res.edgesBtrisA.erase( std::remove_if( res.edgesBtrisA.begin(), res.edgesBtrisA.end(), []( const EdgeTri & et ) { return !et.edge.valid(); } ), res.edgesBtrisA.end() );

    return res;
}

std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<EdgeId> & edgesA,
    const Mesh & b, const std::vector<FaceId> & facesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    using EdgePreciseCoords = std::array<PreciseVertCoords, 2>;
    std::vector<EdgePreciseCoords> edgeACoords( edgesA.size() );
    for ( int i = 0; i < edgesA.size(); ++i )
    {
        EdgeId eA = edgesA[i];
        auto & avc = edgeACoords[i];
        avc[0].id = a.topology.org( eA );
        avc[0].pt = conv( a.points[avc[0].id] );
        avc[1].id = a.topology.dest( eA );
        avc[1].pt = conv( a.points[avc[1].id] );
    }

    const int aVertsSize = (int)a.topology.vertSize();
    using TriPreciseCoords = std::array<PreciseVertCoords, 3>;
    std::vector<TriPreciseCoords> faceBCoords( facesB.size() );
    for ( int i = 0; i < facesB.size(); ++i )
    {
        FaceId fB = facesB[i];
        auto & bvc = faceBCoords[i];
        b.topology.getTriVerts( fB, bvc[0].id, bvc[1].id, bvc[2].id );

        for ( int j = 0; j < 3; ++j )
        {
            const auto bf = b.points[bvc[j].id];
            bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
            bvc[j].id += aVertsSize;
        }
    }

    std::vector<EdgeTri> res;
    for ( int i = 0; i < edgesA.size(); ++i )
    {
        EdgeId eA = edgesA[i];
        const auto & avc = edgeACoords[i];
        for ( int j = 0; j < facesB.size(); ++j )
        {
            FaceId fB = facesB[j];
            const auto & bvc = faceBCoords[j];
            auto isect = doTriangleSegmentIntersect( { bvc[0], bvc[1], bvc[2], avc[0], avc[1] } );
            if ( !isect )
                continue;
            res.emplace_back( isect.dIsLeftFromABC ? eA : eA.sym(), fB );
        }
    }

    return res;
}

std::vector<EdgeTri> findCollidingEdgeTrisPrecise(
    const Mesh & a, const std::vector<FaceId> & facesA,
    const Mesh & b, const std::vector<EdgeId> & edgesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A )
{
    using TriPreciseCoords = std::array<PreciseVertCoords, 3>;
    std::vector<TriPreciseCoords> faceACoords( facesA.size() );
    for ( int i = 0; i < facesA.size(); ++i )
    {
        FaceId fA = facesA[i];
        auto & avc = faceACoords[i];
        a.topology.getTriVerts( fA, avc[0].id, avc[1].id, avc[2].id );
        for ( int j = 0; j < 3; ++j )
        {
            const auto af = a.points[avc[j].id];
            avc[j].pt = conv( af );
        }
    }

    const int aVertsSize = (int)a.topology.vertSize();
    using EdgePreciseCoords = std::array<PreciseVertCoords, 2>;
    std::vector<EdgePreciseCoords> edgeBCoords( edgesB.size() );
    for ( int i = 0; i < edgesB.size(); ++i )
    {
        EdgeId eB = edgesB[i];
        auto & bvc = edgeBCoords[i];
        bvc[0].id = b.topology.org( eB );
        bvc[1].id = b.topology.dest( eB );
        for ( int j = 0; j < 2; ++j )
        {
            const auto bf = b.points[bvc[j].id];
            bvc[j].pt = conv( rigidB2A ? (*rigidB2A)( bf ) : bf );
            bvc[j].id += aVertsSize;
        }
    }

    std::vector<EdgeTri> res;
    for ( int i = 0; i < facesA.size(); ++i )
    {
        FaceId fA = facesA[i];
        const auto & avc = faceACoords[i];
        for ( int j = 0; j < edgesB.size(); ++j )
        {
            EdgeId eB = edgesB[i];
            const auto & bvc = edgeBCoords[i];
            auto isect = doTriangleSegmentIntersect( { avc[0], avc[1], avc[2], bvc[0], bvc[1] } );
            if ( !isect )
                continue;
            res.emplace_back( isect.dIsLeftFromABC ? eB : eB.sym(), fA );
        }
    }

    return res;
}

ConvertToIntVector getToIntConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double invRange = cRangeIntMax / maxDim;

    return [invRange, center] ( const Vector3f& v )
    {
        // perform intermediate operations in double for better precision
        return Vector3i( ( Vector3d{ v } - center ) * invRange );
    };
}

ConvertToFloatVector getToFloatConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );
        
    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double range = maxDim / cRangeIntMax;

    return [range, center] ( const Vector3i& v )
    {
        return Vector3f( Vector3d{ v }*range + center );
    };
}

CoordinateConverters getVectorConverters( const MeshPart& a, const MeshPart& b, const AffineXf3f* rigidB2A )
{
    Box3d bb;
    bb.include( Box3d( a.mesh.computeBoundingBox() ) );
    Box3f bMeshBox = transformed( b.mesh.computeBoundingBox(), rigidB2A );
    bb.include( Box3d( bMeshBox ) );
    CoordinateConverters res;
    res.toInt = getToIntConverter( bb );
    res.toFloat = getToFloatConverter( bb );
    return res;
}

} //namespace MR
