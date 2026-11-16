#include "MRMeshIntersect.h"
#include "MRAABBTreeTraversal.hpp"
#include "MRAABBTree.h"
#include "MRInplaceStack.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRayBoxIntersection.h"
#include "MRTriangleIntersection.h"
#include "MRPrecisePredicates3.h"
#include "MRVector3.h"
#include "MRLine3.h"
#include "MRPlane3.h"
#include "MRMeshBuilder.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include <type_traits>

namespace MR
{

MeshTriPoint MeshIntersectionResult::mtp( const MeshTopology & topology ) const
{
    if ( !proj.valid() )
        return {};
    return { topology.edgeWithLeft( proj.face ), tp };
}

// Filtered: region or validFaces is given, otherwise the leaves skip their checks
template<typename T, bool Filtered>
MeshIntersectionResult meshRayIntersectT_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    const auto& nodes = tree.nodes();
    MeshIntersectionResult res;
    if( nodes.empty() )
        return res;

    // the float version takes the stored box by reference: a copy of it before rayBoxIntersect costs 20% of the time
    auto nodeBox = [&]( NodeId n ) -> decltype(auto)
    {
        if constexpr ( std::is_same_v<T, float> )
            return ( nodes[n].box );
        else
            return Box3<T>( nodes[n].box );
    };

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( nodeBox( NodeId( 0 ) ), rayOrigin, s, e, prec ) )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        T start;
    };
    // every level adds at most MaxChildren - 1 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;
    FaceId face;
    float distance = 0;
    TriPointf triP;
    // the node to visit next is not pushed in the stack
    SubTask st{ NodeId( 0 ), rayStart };
    for (;;)
    {
        if ( st.start < rayEnd )
        {
            const auto & node = nodes[st.n];
            if ( node.leaf() )
            {
                bool stop = false;
                for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
                {
                    const auto & lt = tree.leafTris()[i];
                    if constexpr ( Filtered )
                    {
                        if ( ( region && !region->test( lt.face ) ) || ( validFaces && !validFaces( lt.face ) ) )
                            continue;
                    }
                    const Vector3<T> vA = Vector3<T>( points[lt.verts[0]] ) - line.p;
                    const Vector3<T> vB = Vector3<T>( points[lt.verts[1]] ) - line.p;
                    const Vector3<T> vC = Vector3<T>( points[lt.verts[2]] ) - line.p;
                    const auto triIsect = rayTriangleIntersect( vA, vB, vC, prec );
                    if ( !triIsect )
                        continue;
                    const T t( triIsect->t );
                    if ( !( t < rayEnd && t > rayStart ) )
                        continue;
                    distance = triIsect->t;
                    face = lt.face;
                    triP = triIsect->bary;
                    if ( t == 0 )
                    {
                        stop = true; // intersection exactly at ray origin
                        break;
                    }
                    if ( t < 0 )
                    {
                        assert( rayStart < 0 );
                        rayStart = t;
                        rayEnd = std::min( rayEnd, -t );
                    }
                    else
                    {
                        assert( rayEnd > 0 );
                        rayEnd = t;
                        rayStart = std::max( rayStart, -t );
                    }
                    if ( !closestIntersect )
                    {
                        stop = true;
                        break;
                    }
                }
                if ( stop )
                    break;
            }
            else
            {
                // the children crossed by the ray ordered by the entry, the one entered last first
                SubTask ch[AABBTree::MaxChildren];
                int numCh = 0;
                for ( int i = 0; i < node.count; ++i )
                {
                    const NodeId n( node.first + i );
                    T cs = rayStart, ce = rayEnd;
                    if ( !rayBoxIntersect( nodeBox( n ), rayOrigin, cs, ce, prec ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].start < cs; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { n, cs };
                }
                if ( numCh > 0 )
                {
                    // descend in the child entered first, the others wait in the stack
                    for ( int i = 0; i + 1 < numCh; ++i )
                        subtasks.push( ch[i] );
                    st = ch[numCh - 1];
                    continue;
                }
            }
        }
        if ( subtasks.empty() )
            break;
        st = subtasks.top();
        subtasks.pop();
    }

    if( face )
    {
        res.proj.face = face;
        res.distanceAlongLine = distance;
        res.proj.point = Vector3f( line.p + res.distanceAlongLine * line.d );
        res.tp = triP;
    }
    return res;
}

template<typename T>
MeshIntersectionResult meshRayIntersect_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    if ( region || validFaces )
        return meshRayIntersectT_<T, true>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
    return meshRayIntersectT_<T, false>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
}

template<typename T>
MeshIntersectionResult rayMeshIntersect_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>* prec, bool closestIntersect, const FacePredicate & validFaces )
{
    if( prec )
    {
        return meshRayIntersect_<T>( points, tree, region, line, rayStart, rayEnd, *prec, closestIntersect, validFaces );
    }
    else
    {
        const IntersectionPrecomputes<T> precNew( line.d );
        return meshRayIntersect_<T>( points, tree, region, line, rayStart, rayEnd, precNew, closestIntersect, validFaces );
    }
}

MeshIntersectionResult rayMeshIntersect( const MeshPart& meshPart, const Line3f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec, bool closestIntersect, const FacePredicate & validFaces )
{
    return rayMeshIntersect_( meshPart.mesh.points, meshPart.mesh.getAABBTree(), meshPart.region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
}

MeshIntersectionResult rayMeshIntersect( const MeshPart& meshPart, const Line3d& line,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, bool closestIntersect, const FacePredicate & validFaces )
{
    return rayMeshIntersect_( meshPart.mesh.points, meshPart.mesh.getAABBTree(), meshPart.region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
}

MeshIntersectionResult rayMeshIntersect( const VertCoords & points, const AABBTree & tree, const Line3f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec, bool closestIntersect, const FacePredicate & validFaces, const FaceBitSet * region )
{
    return rayMeshIntersect_( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
}

MeshIntersectionResult rayMeshIntersect( const VertCoords & points, const AABBTree & tree, const Line3d& line,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, bool closestIntersect, const FacePredicate & validFaces, const FaceBitSet * region )
{
    return rayMeshIntersect_( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
}

void multiRayMeshIntersect(
    const MeshPart& meshPart,
    const std::vector<Vector3f>& origins,
    const std::vector<Vector3f>& dirs,
    const MultiRayMeshIntersectResult& result,
    float rayStart, float rayEnd,
    bool closestIntersect,
    const FacePredicate & validFaces
)
{
    multiRayMeshIntersect( meshPart.mesh.points, meshPart.mesh.getAABBTree(), origins, dirs, result, rayStart, rayEnd, closestIntersect, validFaces, meshPart.region );
}

void multiRayMeshIntersect(
    const VertCoords & points, const AABBTree & tree,
    const std::vector<Vector3f>& origins,
    const std::vector<Vector3f>& dirs,
    const MultiRayMeshIntersectResult& result,
    float rayStart, float rayEnd,
    bool closestIntersect,
    const FacePredicate & validFaces,
    const FaceBitSet * region
)
{
    MR_TIMER;

    const auto sz = origins.size();
    assert( dirs.size() == sz );
    if ( result.intersectingRays )
    {
        result.intersectingRays->clear();
        result.intersectingRays->resize( sz, false );
    }
    constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();
    if ( result.rayDistances )
    {
        result.rayDistances->clear();
        result.rayDistances->resize( sz, cQuietNan );
    }
    if ( result.isectFaces )
    {
        result.isectFaces->clear();
        result.isectFaces->resize( sz );
    }
    if ( result.isectBary )
    {
        result.isectBary->clear();
        result.isectBary->resize( sz, TriPointf( cQuietNan, cQuietNan ) );
    }
    if ( result.isectPts )
    {
        result.isectPts->clear();
        result.isectPts->resize( sz, Vector3f( cQuietNan, cQuietNan, cQuietNan ) );
    }

    auto processRay = [&]( size_t i )
    {
        auto res = rayMeshIntersect_<float>( points, tree, region, Line3f( origins[i], dirs[i] ), rayStart, rayEnd, nullptr, closestIntersect, validFaces );
        if ( !res )
            return;
        if ( result.intersectingRays )
            result.intersectingRays->set( i );
        if ( result.rayDistances )
            ( *result.rayDistances )[i] = res.distanceAlongLine;
        if ( result.isectFaces )
            (*result.isectFaces)[i] = res.proj.face;
        if ( result.isectBary )
            (*result.isectBary)[i] = res.tp;
        if ( result.isectPts )
            (*result.isectPts)[i] = res.proj.point;
    };

    if ( result.intersectingRays )
        BitSetParallelForAll( *result.intersectingRays, processRay );
    else
        ParallelFor( size_t( 0 ), sz, processRay );
}

template<typename T>
MultiMeshIntersectionResult rayMultiMeshAnyIntersect_( const std::vector<Line3Mesh<T>> & lineMeshes,
    T rayStart /*= 0.0f*/, T rayEnd /*= FLT_MAX */ )
{
    MultiMeshIntersectionResult res;
    for ( const auto & lm : lineMeshes )
    {
        assert( lm.mesh );
        auto prec = lm.prec;
        IntersectionPrecomputes<T> myPrec;
        if ( !prec )
        {
            myPrec = { lm.line.d };
            prec = &myPrec;
        }
        if ( auto r = meshRayIntersect_( lm.mesh->points, lm.mesh->getAABBTree(), lm.region, lm.line, rayStart, rayEnd, *prec, false, {} ) )
        {
            res = MultiMeshIntersectionResult{ r };
            break;
        }
    }
    return res;
}

MultiMeshIntersectionResult rayMultiMeshAnyIntersect( const std::vector<Line3fMesh> & lineMeshes,
    float rayStart, float rayEnd )
{
    return rayMultiMeshAnyIntersect_( lineMeshes, rayStart, rayEnd );
}

MultiMeshIntersectionResult rayMultiMeshAnyIntersect( const std::vector<Line3dMesh> & lineMeshes,
    double rayStart, double rayEnd )
{
    return rayMultiMeshAnyIntersect_( lineMeshes, rayStart, rayEnd );
}

template<typename T, typename F>
void rayMeshIntersectAll_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line, F && callback,
    T rayStart /*= 0.0f*/, T rayEnd /*= FLT_MAX */, const IntersectionPrecomputes<T>& prec )
{
    // the side of a crossing is computed only if the callback takes it
    const auto report = [&callback] ( const MeshIntersectionResult & found, [[maybe_unused]] auto && fromFront ) -> bool
    {
        if constexpr ( std::is_invocable_v<F, const MeshIntersectionResult &, bool> )
            return callback( found, fromFront() );
        else
            return callback( found );
    };

    const auto& nodes = tree.nodes();
    if( nodes.empty() )
        return;

    // the double version decides every intersection precisely in integer coordinates of the segment's ends and triangle's vertices;
    // rounding to integers can bring a vertex and a point of the segment closer by one integer step,
    // so all boxes are expanded by two steps (the second one is for the errors of computations in double), otherwise
    // 1) a segment's end clipped by the root box can get exactly on a face lying in the plane of the box,
    //    and simulation-of-simplicity can put this end inside the mesh, losing the crossing of that face,
    // 2) a box can be culled although its triangle is touched by the segment in integer coordinates
    ConvertToIntVector convToInt;
    ConvertToFloatVector convToFloat;
    [[maybe_unused]] T boxExpansion = 0;
    if constexpr ( std::is_same_v<T, double> )
    {
        const Box3d rootBox( nodes[NodeId( 0 )].box );
        convToInt = getToIntConverter( rootBox );
        convToFloat = getToFloatConverter( rootBox );
        boxExpansion = 2 * convToFloat.range;
    }
    // the float version takes the stored box by reference, avoiding a copy
    auto nodeBox = [&] ( NodeId n ) -> decltype(auto)
    {
        if constexpr ( std::is_same_v<T, double> )
            return Box3d( nodes[n].box ).expanded( Vector3d::diagonal( boxExpansion ) );
        else
            return ( nodes[n].box );
    };

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( nodeBox( NodeId( 0 ) ), rayOrigin, s, e, prec ) )
    {
        return;
    }

    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( NodeId( 0 ) );
    Vector3d dP, eP;
    [[maybe_unused]] double invDirLenSq = 0;
    std::array<PreciseVertCoords, 5> pvc;

    if constexpr ( std::is_same_v<T, double> )
    {
        // the ends are rounded to integers directly from double: rounding them to float first could move them by many integer steps
        dP = line( s );
        eP = line( e );
        pvc[3].pt = convToInt( dP );
        pvc[3].id = VertId( points.size() ); // after all vertices
        pvc[4].pt = convToInt( eP );
        pvc[4].id = pvc[3].id + 1;
        invDirLenSq = 1 / line.d.lengthSq();
    }

    while( !subtasks.empty() )
    {
        const auto& node = nodes[subtasks.top()];
        subtasks.pop();
        if( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
            {
                const auto & lt = tree.leafTris()[i];
                const auto face = lt.face;
                if ( region && !region->test( face ) )
                    continue;
                for ( int j = 0; j < 3; ++j )
                    pvc[j].id = lt.verts[j];
                if constexpr ( std::is_same_v<T, double> )
                {
                    // double version (more precise); a triangle outside its expanded box cannot be crossed in integer coordinates
                    T bs = rayStart, be = rayEnd;
                    if ( !rayBoxIntersect( Box3d( tree.leafBoxes()[i] ).expanded( Vector3d::diagonal( boxExpansion ) ), rayOrigin, bs, be, prec ) )
                        continue;
                    for ( int j = 0; j < 3; ++j )
                        pvc[j].pt = convToInt( points[pvc[j].id] );
                    if ( const auto isect = doTriangleSegmentIntersect( pvc ) )
                    {
                        MeshIntersectionResult found;
                        found.proj.face = face;
                        const auto ip = findTriangleSegmentIntersectionPreciseInt( pvc[0].pt, pvc[1].pt, pvc[2].pt, pvc[3].pt, pvc[4].pt );
                        // no intersection point only if the segment lies in the plane of the triangle without crossing its sides
                        found.proj.point = ip ? convToFloat( *ip ) : Vector3f( ( dP + eP ) * 0.5 );
                        // the segment is already limited by [rayStart, rayEnd], so the hit is reported without checking its distance,
                        // which is computed from the rounded point and can be slightly outside of the range
                        found.distanceAlongLine = float( dot( Vector3d( found.proj.point ) - line.p, line.d ) * invDirLenSq );
                        found.tp = TriPointf( found.proj.point, points[pvc[0].id], points[pvc[1].id], points[pvc[2].id] );
                        // segment start pvc[3] is behind the triangle if dIsLeftFromABC
                        if ( !report( found, [&] { return !isect.dIsLeftFromABC; } ) )
                            return;
                    }
                }
                else
                {
                    // float version (faster); the box of the triangle rejects the near-miss hits of the float triangle test
                    T bs = rayStart, be = rayEnd;
                    if ( !rayBoxIntersect( tree.leafBoxes()[i], rayOrigin, bs, be, prec ) )
                        continue;
                    const Vector3<T> vA = Vector3<T>( points[pvc[0].id] ) - line.p;
                    const Vector3<T> vB = Vector3<T>( points[pvc[1].id] ) - line.p;
                    const Vector3<T> vC = Vector3<T>( points[pvc[2].id] ) - line.p;
                    const auto triIsect = rayTriangleIntersect( vA, vB, vC, prec );
                    if ( triIsect && triIsect->t < rayEnd && triIsect->t > rayStart )
                    {
                        MeshIntersectionResult found;
                        found.proj.face = face;
                        found.proj.point = Vector3f( line( triIsect->t ) );
                        found.tp = triIsect->bary;
                        found.distanceAlongLine = float( triIsect->t );
                        if ( !report( found, [&] { return dot( cross( vB - vA, vC - vA ), line.d ) < 0; } ) )
                            return;
                    }
                }
            }
        }
        else
        {
            for ( int i = 0; i < node.count; ++i )
            {
                const NodeId n( node.first + i );
                s = rayStart, e = rayEnd;
                if( rayBoxIntersect( nodeBox( n ), rayOrigin, s, e, prec ) )
                    subtasks.push( n );
            }
        }
    }
}

template<typename T, typename F>
void rayMeshIntersectAllPrec_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line, const F & callback,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>* prec )
{
    assert( callback );
    if ( !callback )
        return;
    if( prec )
    {
        return rayMeshIntersectAll_<T>( points, tree, region, line, callback, rayStart, rayEnd, *prec );
    }
    else
    {
        const IntersectionPrecomputes<T> precNew( line.d );
        return rayMeshIntersectAll_<T>( points, tree, region, line, callback, rayStart, rayEnd, precNew );
    }
}

void rayMeshIntersectAll( const MeshPart& meshPart, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec )
{
    rayMeshIntersectAllPrec_( meshPart.mesh.points, meshPart.mesh.getAABBTree(), meshPart.region, line, callback, rayStart, rayEnd, prec );
}

void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec )
{
    rayMeshIntersectAllPrec_( meshPart.mesh.points, meshPart.mesh.getAABBTree(), meshPart.region, line, callback, rayStart, rayEnd, prec );
}

void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionWithSideCallback callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec )
{
    rayMeshIntersectAllPrec_( meshPart.mesh.points, meshPart.mesh.getAABBTree(), meshPart.region, line, callback, rayStart, rayEnd, prec );
}

void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec, const FaceBitSet * region )
{
    rayMeshIntersectAllPrec_( points, tree, region, line, callback, rayStart, rayEnd, prec );
}

void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, const FaceBitSet * region )
{
    rayMeshIntersectAllPrec_( points, tree, region, line, callback, rayStart, rayEnd, prec );
}

void rayMeshIntersectAll( const VertCoords & points, const AABBTree & tree, const Line3d& line, MeshIntersectionWithSideCallback callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, const FaceBitSet * region )
{
    rayMeshIntersectAllPrec_( points, tree, region, line, callback, rayStart, rayEnd, prec );
}

namespace
{

// calls onTri( leafTri ) for the triangles of the tree (only from region if given), which boxes pass boxOk
template <typename BoxOk, typename OnTri>
void boxedTreeIntersect( const AABBTree & tree, const FaceBitSet * region, BoxOk && boxOk, OnTri && onTri )
{
    forWideTris( tree, boxOk, boxOk, [&]( int i )
    {
        const auto & lt = tree.leafTris()[i];
        if ( region && !region->test( lt.face ) )
            return Processing::Continue;
        onTri( lt );
        return Processing::Continue;
    } );
}

// adds to the outputs the triangles of the mesh part, which boxes pass boxOk
template <typename BoxOk>
void boxedMeshIntersect( const MeshPart& meshPart, BoxOk && boxOk,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    assert( fs || ues || vs || fsVec );
    const auto& m = meshPart.mesh;

    assert( !fs  || fs->size()  >= m.topology.faceSize() );
    assert( !ues || ues->size() >= m.topology.undirectedEdgeSize() );
    assert( !vs  || vs->size()  >= m.topology.vertSize() );

    boxedTreeIntersect( m.getAABBTree(), meshPart.region, boxOk, [&]( const AABBTree::LeafTri & lt )
    {
        const auto face = lt.face;
        if ( fs )
            fs->set( face );
        if ( fsVec )
            fsVec->push_back( face );
        if ( vs )
            for ( auto v : lt.verts )
                vs->set( v );
        if ( ues )
        {
            EdgeId e0, e1, e2;
            m.topology.getTriEdges( face, e0, e1, e2 );
            ues->set( e0 );
            ues->set( e1 );
            ues->set( e2 );
        }
    } );
}

// adds to the outputs the triangles of the tree (only from region if given), which boxes pass boxOk
template <typename BoxOk>
void boxedTreeIntersect( const AABBTree & tree, const FaceBitSet * region, BoxOk && boxOk,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    assert( fs || vs || fsVec );
    boxedTreeIntersect( tree, region, boxOk, [&]( const AABBTree::LeafTri & lt )
    {
        if ( fs )
            fs->autoResizeSet( lt.face );
        if ( fsVec )
            fsVec->push_back( lt.face );
        if ( vs )
            for ( auto v : lt.verts )
                vs->autoResizeSet( v );
    } );
}

auto planeBoxOk( const Plane3f & plane )
{
    auto minCorner = Box3f::getMinBoxCorner( plane.n );
    auto maxCorner = minCorner;
    for ( auto & v : maxCorner )
        v = !v;
    return [&plane, minCorner, maxCorner]( const Box3f & box )
    {
        return dot( plane.n, box.corner( minCorner ) ) <= plane.d && dot( plane.n, box.corner( maxCorner ) ) >= plane.d;
    };
}

auto xyPlaneBoxOk( float zLevel )
{
    return [zLevel]( const Box3f & box )
    {
        return box.min.z <= zLevel && box.max.z >= zLevel;
    };
}

} // anonymous namespace

void planeMeshIntersect( const MeshPart& meshPart, const Plane3f & plane,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    MR_TIMER;
    boxedMeshIntersect( meshPart, planeBoxOk( plane ), fs, ues, vs, fsVec );
}

void planeMeshIntersect( const AABBTree & tree, const Plane3f & plane,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec, const FaceBitSet * region )
{
    MR_TIMER;
    boxedTreeIntersect( tree, region, planeBoxOk( plane ), fs, vs, fsVec );
}

void xyPlaneMeshIntersect( const MeshPart& meshPart, float zLevel,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    MR_TIMER;
    boxedMeshIntersect( meshPart, xyPlaneBoxOk( zLevel ), fs, ues, vs, fsVec );
}

void xyPlaneMeshIntersect( const AABBTree & tree, float zLevel,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec, const FaceBitSet * region )
{
    MR_TIMER;
    boxedTreeIntersect( tree, region, xyPlaneBoxOk( zLevel ), fs, vs, fsVec );
}

} //namespace MR
