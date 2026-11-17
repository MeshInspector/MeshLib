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
#include "MRSimd4.hpp"
#include <optional>
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
    MeshIntersectionResult res;
    if( tree.empty() )
        return res;
    const auto & leafTris = tree.leafTris();

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box3<T>( tree.getBoundingBox() ), rayOrigin, s, e, prec ) )
        return res;

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        T start;
    };
    // every level adds at most MaxChildren - 1 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;
    FaceId face;
    float distance = 0;
    TriPointf triP;
    // the node to visit next is not pushed in the stack
    SubTask st{ tree.rootRef(), rayStart };
    for (;;)
    {
        if ( st.start < rayEnd )
        {
            if ( st.ref < 0 )
            {
                bool stop = false;
                for ( int i = ~st.ref * AABBTree::MaxLeafTris, end = i + AABBTree::MaxLeafTris; i < end && leafTris[i].face; ++i )
                {
                    const auto & lt = leafTris[i];
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
                const auto & node = tree.simdNodes()[st.ref];
                SubTask ch[AABBTree::MaxChildren];
                int numCh = 0;
                for ( int i = 0; i < AABBTree::MaxChildren && node.child[i]; ++i )
                {
                    T cs = rayStart, ce = rayEnd;
                    if ( !rayBoxIntersect( Box3<T>( node.get( i ) ), rayOrigin, cs, ce, prec ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].start < cs; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { node.child[i], cs };
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

namespace
{

/// 1 / line.d as stored in IntersectionPrecomputes
Vector3f invDirOf( const IntersectionPrecomputes<float>& prec )
{
#if defined(__x86_64__) || defined(_M_X64)
    alignas( 16 ) float v[4];
    _mm_store_ps( v, prec.invDir );
    return { v[0], v[1], v[2] };
#elif defined(__aarch64__) || defined(_M_ARM64)
    float v[4];
    vst1q_f32( v, prec.invDir );
    return { v[0], v[1], v[2] };
#elif defined(__wasm_simd128__)
    float v[4];
    wasm_v128_store( v, prec.invDir );
    return { v[0], v[1], v[2] };
#else
    return prec.invDir;
#endif
}

/// the intersections of a ray with four boxes in SoA order by one SIMD test, computed exactly as rayBoxIntersect:
/// the plane of the box entered first along each axis is selected once for the ray,
/// the same choice as min/max of the distances to both planes in rayBoxIntersect, including zero direction (invDir = FLT_MAX)
struct SimdRayBoxes
{
    int nearIdx[3], farIdx[3];
    Simd4::F ox, oy, oz, idx, idy, idz;

    SimdRayBoxes( const Line3f & line, const IntersectionPrecomputes<float> & prec )
    {
        using namespace Simd4;
        const Vector3f invDir = invDirOf( prec );
        for ( int a = 0; a < 3; ++a )
        {
            nearIdx[a] = invDir[a] >= 0 ? a : a + 3;
            farIdx[a] = invDir[a] >= 0 ? a + 3 : a;
        }
        ox = set1( line.p.x ); oy = set1( line.p.y ); oz = set1( line.p.z );
        idx = set1( invDir.x ); idy = set1( invDir.y ); idz = set1( invDir.z );
    }

    /// the parameters of the entries into the boxes, after rayStart, and the bit mask of the boxes crossed within [rayStart, rayEnd]
    /// (the unused slots have empty boxes, which are not crossed)
    int test( const AABBTree::SimdBoxes & b, float rayStart, float rayEnd, Simd4::F & tNear ) const
    {
        using namespace Simd4;
        const F tx0 = mul( sub( load( b.box[nearIdx[0]] ), ox ), idx );
        const F ty0 = mul( sub( load( b.box[nearIdx[1]] ), oy ), idy );
        const F tz0 = mul( sub( load( b.box[nearIdx[2]] ), oz ), idz );
        const F tx1 = mul( sub( load( b.box[farIdx[0]] ), ox ), idx );
        const F ty1 = mul( sub( load( b.box[farIdx[1]] ), oy ), idy );
        const F tz1 = mul( sub( load( b.box[farIdx[2]] ), oz ), idz );
        tNear = max( max( max( tx0, tz0 ), ty0 ), set1( rayStart ) );
        const F tFar = min( min( min( tx1, tz1 ), ty1 ), set1( rayEnd ) );
        return bits( le( tNear, tFar ) );
    }
    int test( const AABBTree::SimdBoxes & b, float rayStart, float rayEnd ) const
    {
        Simd4::F tNear;
        return test( b, rayStart, rayEnd, tNear );
    }
};

/// the intersections of a ray with the (up to) four triangles of a leaf of AABBTree by one SIMD test, computed exactly as rayTriangleIntersect
struct SimdRayTris
{
    int kx, ky, kz;
    Simd4::F okx, oky, okz, Sx, Sy, Sz;

    SimdRayTris( const Line3f & line, const IntersectionPrecomputes<float> & prec )
        : kx( prec.idxX ), ky( prec.idxY ), kz( prec.maxDimIdxZ ),
        okx( Simd4::set1( line.p[kx] ) ), oky( Simd4::set1( line.p[ky] ) ), okz( Simd4::set1( line.p[kz] ) ),
        Sx( Simd4::set1( prec.Sx ) ), Sy( Simd4::set1( prec.Sy ) ), Sz( Simd4::set1( prec.Sz ) )
    {
    }

    /// returns the bit mask of the crossed triangles, and fills for them the parameter t along the ray, V, W and 1/det of rayTriangleIntersect
    int test( const AABBTree::SimdLeaf & leaf, float ( &ts )[4], float ( &vs )[4], float ( &ws )[4], float ( &ids )[4] ) const
    {
        using namespace Simd4;
        const F epsilon = set1( std::numeric_limits<float>::epsilon() );
        const F zero = set1( 0.0f );
        const F aZ = sub( load( leaf.coord[0][kz] ), okz );
        const F bZ = sub( load( leaf.coord[1][kz] ), okz );
        const F cZ = sub( load( leaf.coord[2][kz] ), okz );
        const F Ax = sub( sub( load( leaf.coord[0][kx] ), okx ), mul( Sx, aZ ) );
        const F Ay = sub( sub( load( leaf.coord[0][ky] ), oky ), mul( Sy, aZ ) );
        const F Bx = sub( sub( load( leaf.coord[1][kx] ), okx ), mul( Sx, bZ ) );
        const F By = sub( sub( load( leaf.coord[1][ky] ), oky ), mul( Sy, bZ ) );
        const F Cx = sub( sub( load( leaf.coord[2][kx] ), okx ), mul( Sx, cZ ) );
        const F Cy = sub( sub( load( leaf.coord[2][ky] ), oky ), mul( Sy, cZ ) );
        const F eps = mul( mul( epsilon, max( max( abs( Ax ), abs( Bx ) ), abs( Cx ) ) ), max( max( abs( Ay ), abs( By ) ), abs( Cy ) ) );
        const F U = sub( mul( Cx, By ), mul( Cy, Bx ) );
        const F V = sub( mul( Ax, Cy ), mul( Ay, Cx ) );
        const F W = sub( mul( Bx, Ay ), mul( By, Ax ) );
        const F negEps = sub( zero, eps );
        const M someNeg = orm( orm( lt( U, negEps ), lt( V, negEps ) ), lt( W, negEps ) );
        const M somePos = orm( orm( gt( U, eps ), gt( V, eps ) ), gt( W, eps ) );
        const F det = add( add( U, V ), W );
        // the unused slots have NaN coordinates, so neither someNeg nor somePos
        const int hits = bits( andm( xorm( someNeg, somePos ), ne( det, zero ) ) );
        if ( hits )
        {
            const F invDet = div( set1( 1.0f ), det );
            store( ts, mul( add( add( mul( U, mul( Sz, aZ ) ), mul( V, mul( Sz, bZ ) ) ), mul( W, mul( Sz, cZ ) ) ), invDet ) );
            store( vs, V );
            store( ws, W );
            store( ids, invDet );
        }
        return hits;
    }
};

} // anonymous namespace

/// the float version of meshRayIntersectT_ over the SIMD layout of the tree: tests all children of a node and all triangles of a leaf at once;
/// visits the nodes and the triangles in the same order and computes the same values as meshRayIntersectT_, so it finds the same intersection;
/// the coordinates of the triangles are taken from the tree (the same as the points it was made or updated for)
template<bool Filtered>
MeshIntersectionResult meshRayIntersectSimd_( const AABBTree & tree, const FaceBitSet * region, const Line3f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    using namespace Simd4;
    MeshIntersectionResult res;
    if ( tree.empty() )
        return res;
    const auto& simdNodes = tree.simdNodes();
    const auto& simdLeaves = tree.simdLeaves();
    const auto& leafTris = tree.leafTris();

    RayOrigin<float> rayOrigin{ line.p };
    {
        float s = rayStart, e = rayEnd;
        if ( !rayBoxIntersect( tree.getBoundingBox(), rayOrigin, s, e, prec ) )
            return res;
    }

    const SimdRayBoxes rayBoxes( line, prec );

    // ray-triangles: the same sheared coordinates as in rayTriangleIntersect
    const SimdRayTris rayTris( line, prec );

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        float start;
    };
    // every level adds at most MaxChildren - 1 tasks, and the tree has at most MaxDepth levels
    InplaceStack<SubTask, 64> subtasks;
    FaceId face;
    float distance = 0;
    TriPointf triP;
    // the node to visit next is not pushed in the stack
    SubTask st{ tree.rootRef(), rayStart };
    for (;;)
    {
        if ( st.start < rayEnd )
        {
            if ( st.ref < 0 )
            {
                alignas( 16 ) float ts[4], vs[4], ws[4], ids[4];
                int hits = rayTris.test( simdLeaves[~st.ref], ts, vs, ws, ids );
                bool stop = false;
                if ( hits )
                {
                    // the triangles in their order, as rayStart and rayEnd change after each intersection
                    for ( ; hits; hits &= hits - 1 )
                    {
                        const int i = lowestBit( hits );
                        if constexpr ( Filtered )
                        {
                            const auto f = leafTris[~st.ref * AABBTree::MaxLeafTris + i].face;
                            if ( ( region && !region->test( f ) ) || ( validFaces && !validFaces( f ) ) )
                                continue;
                        }
                        const float ti = ts[i];
                        if ( !( ti < rayEnd && ti > rayStart ) )
                            continue;
                        distance = ti;
                        face = leafTris[~st.ref * AABBTree::MaxLeafTris + i].face;
                        triP = TriPointf( vs[i] * ids[i], ws[i] * ids[i] );
                        if ( ti == 0 )
                        {
                            stop = true; // intersection exactly at ray origin
                            break;
                        }
                        if ( ti < 0 )
                        {
                            assert( rayStart < 0 );
                            rayStart = ti;
                            rayEnd = std::min( rayEnd, -ti );
                        }
                        else
                        {
                            assert( rayEnd > 0 );
                            rayEnd = ti;
                            rayStart = std::max( rayStart, -ti );
                        }
                        if ( !closestIntersect )
                        {
                            stop = true;
                            break;
                        }
                    }
                }
                if ( stop )
                    break;
            }
            else
            {
                const auto & node = simdNodes[st.ref];
                F tNear;
                int crossed = rayBoxes.test( node, rayStart, rayEnd, tNear );
                if ( crossed )
                {
                    alignas( 16 ) float tn[4];
                    store( tn, tNear );
                    // the children crossed by the ray ordered by the entry, the one entered last first
                    SubTask ch[AABBTree::MaxChildren];
                    int numCh = 0;
                    for ( ; crossed; crossed &= crossed - 1 )
                    {
                        const int i = lowestBit( crossed );
                        int k = numCh++;
                        for ( ; k > 0 && ch[k - 1].start < tn[i]; --k )
                            ch[k] = ch[k - 1];
                        ch[k] = { node.child[i], tn[i] };
                    }
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
        res.proj.point = line.p + res.distanceAlongLine * line.d;
        res.tp = triP;
    }
    return res;
}

template<typename T>
MeshIntersectionResult meshRayIntersect_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    if constexpr ( std::is_same_v<T, float> )
    {
        if ( region || validFaces )
            return meshRayIntersectSimd_<true>( tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
        return meshRayIntersectSimd_<false>( tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
    }
    else
    {
        if ( region || validFaces )
            return meshRayIntersectT_<T, true>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
        return meshRayIntersectT_<T, false>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
    }
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

    if( tree.empty() )
        return;
    const auto & leafTris = tree.leafTris();

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
        const Box3d rootBox( tree.getBoundingBox() );
        convToInt = getToIntConverter( rootBox );
        convToFloat = getToFloatConverter( rootBox );
        boxExpansion = 2 * convToFloat.range;
    }
    auto nodeBox = [&] ( const Box3f & box )
    {
        if constexpr ( std::is_same_v<T, double> )
            return Box3d( box ).expanded( Vector3d::diagonal( boxExpansion ) );
        else
            return box;
    };

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( nodeBox( tree.getBoundingBox() ), rayOrigin, s, e, prec ) )
    {
        return;
    }

    // the float version tests the four boxes of the children of a node or of the triangles of a leaf at once,
    // computing the same as rayBoxIntersect (see meshRayIntersectSimd_); returns the bit mask of the crossed boxes
    // the float version tests the four boxes of the children of a node or of the triangles of a leaf at once
    [[maybe_unused]] std::optional<SimdRayBoxes> simdRayBoxes;
    if constexpr ( std::is_same_v<T, float> )
        simdRayBoxes.emplace( line, prec );
    [[maybe_unused]] auto simdCrossed = [&]( const AABBTree::SimdBoxes & b ) { return simdRayBoxes->test( b, float( rayStart ), float( rayEnd ) ); };

    [[maybe_unused]] std::optional<SimdRayTris> simdRayTris;
    if constexpr ( std::is_same_v<T, float> )
        simdRayTris.emplace( line, prec );

    // every level adds at most MaxChildren - 1 nodes, and the tree has at most MaxDepth levels
    InplaceStack<int, 64> subtasks;
    subtasks.push( tree.rootRef() );
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
        const int r = subtasks.top();
        subtasks.pop();
        if( r < 0 )
        {
            // the float version tests all triangles of the leaf at once, and with their boxes rejects the near-miss hits;
            // the boxes are computed only for the leaves with crossed triangles, which are few
            [[maybe_unused]] int crossedTris = 0;
            [[maybe_unused]] alignas( 16 ) float ts[4], vs[4], ws[4], ids[4];
            if constexpr ( std::is_same_v<T, float> )
            {
                crossedTris = simdRayTris->test( tree.simdLeaves()[~r], ts, vs, ws, ids );
                if ( crossedTris )
                    crossedTris &= simdCrossed( leafTriBoxes( tree, ~r ) );
                if ( !crossedTris )
                    continue;
            }
            [[maybe_unused]] AABBTree::SimdBoxes leafBoxes( noInit );
            if constexpr ( std::is_same_v<T, double> )
                leafBoxes = leafTriBoxes( tree, ~r );
            for ( int k = 0; k < AABBTree::MaxLeafTris; ++k )
            {
                const auto & lt = leafTris[~r * AABBTree::MaxLeafTris + k];
                if ( !lt.face )
                    break;
                const auto face = lt.face;
                if ( region && !region->test( face ) )
                    continue;
                for ( int j = 0; j < 3; ++j )
                    pvc[j].id = lt.verts[j];
                if constexpr ( std::is_same_v<T, double> )
                {
                    // double version (more precise); a triangle outside its expanded box cannot be crossed in integer coordinates
                    T bs = rayStart, be = rayEnd;
                    if ( !rayBoxIntersect( Box3d( leafBoxes.get( k ) ).expanded( Vector3d::diagonal( boxExpansion ) ), rayOrigin, bs, be, prec ) )
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
                    // float version (faster): the triangle and its box are crossed, the coordinates are taken from the tree
                    if ( !( crossedTris & ( 1 << k ) ) )
                        continue;
                    const float t = ts[k];
                    if ( t < rayEnd && t > rayStart )
                    {
                        MeshIntersectionResult found;
                        found.proj.face = face;
                        found.proj.point = Vector3f( line( t ) );
                        found.tp = TriPointf( vs[k] * ids[k], ws[k] * ids[k] );
                        found.distanceAlongLine = t;
                        const int slot = ~r * AABBTree::MaxLeafTris + k;
                        if ( !report( found, [&]
                        {
                            const Vector3f vA = tree.leafPoint( slot, 0 ) - line.p;
                            const Vector3f vB = tree.leafPoint( slot, 1 ) - line.p;
                            const Vector3f vC = tree.leafPoint( slot, 2 ) - line.p;
                            return dot( cross( vB - vA, vC - vA ), line.d ) < 0;
                        } ) )
                            return;
                    }
                }
            }
        }
        else
        {
            const auto & node = tree.simdNodes()[r];
            if constexpr ( std::is_same_v<T, float> )
            {
                for ( int m = simdCrossed( node ); m; m &= m - 1 )
                    subtasks.push( node.child[Simd4::lowestBit( m )] );
            }
            else
            {
                for ( int i = 0; i < AABBTree::MaxChildren && node.child[i]; ++i )
                {
                    s = rayStart, e = rayEnd;
                    if( rayBoxIntersect( nodeBox( node.get( i ) ), rayOrigin, s, e, prec ) )
                        subtasks.push( node.child[i] );
                }
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

// adds to the outputs the triangles of the mesh part, which boxes pass boxOk
template <typename BoxOk>
void boxedMeshIntersect( const MeshPart& meshPart, BoxOk && boxOk,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    assert( fs || ues || vs || fsVec );
    const auto& m = meshPart.mesh;
    const auto& tree = m.getAABBTree();

    assert( !fs  || fs->size()  >= m.topology.faceSize() );
    assert( !ues || ues->size() >= m.topology.undirectedEdgeSize() );
    assert( !vs  || vs->size()  >= m.topology.vertSize() );

    forWideTris( tree, boxOk, [&]( int i )
    {
        const auto & lt = tree.leafTris()[i];
        const auto face = lt.face;
        if ( meshPart.region && !meshPart.region->test( face ) )
            return Processing::Continue;
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
        return Processing::Continue;
    } );
}

// adds to the outputs the triangles of the tree (only from region if given), which boxes pass boxOk
template <typename BoxOk>
void boxedTreeIntersect( const AABBTree & tree, const FaceBitSet * region, BoxOk && boxOk,
    FaceBitSet * fs, VertBitSet * vs, std::vector<FaceId> * fsVec )
{
    assert( fs || vs || fsVec );
    forWideTris( tree, boxOk, [&]( int i )
    {
        const auto & lt = tree.leafTris()[i];
        if ( region && !region->test( lt.face ) )
            return Processing::Continue;
        if ( fs )
            fs->autoResizeSet( lt.face );
        if ( fsVec )
            fsVec->push_back( lt.face );
        if ( vs )
            for ( auto v : lt.verts )
                vs->autoResizeSet( v );
        return Processing::Continue;
    } );
}

auto planeBoxOk( const Plane3f & plane )
{
    auto minCorner = Box3f::getMinBoxCorner( plane.n );
    auto maxCorner = minCorner;
    for ( auto & v : maxCorner )
        v = !v;
    // the rows of SIMD boxes with the coordinates of the corners
    int minRow[3], maxRow[3];
    for ( int c = 0; c < 3; ++c )
    {
        minRow[c] = minCorner[c] ? c + 3 : c;
        maxRow[c] = maxCorner[c] ? c + 3 : c;
    }
    using namespace Simd4;
    return WideBoxTest
    {
        [&plane, minCorner, maxCorner]( const Box3f & box )
        {
            return dot( plane.n, box.corner( minCorner ) ) <= plane.d && dot( plane.n, box.corner( maxCorner ) ) >= plane.d;
        },
        // the same as dot above for four boxes at once; the unused slots have infinite or NaN dot products
        [nx = set1( plane.n.x ), ny = set1( plane.n.y ), nz = set1( plane.n.z ), d = set1( plane.d ),
            minRow0 = minRow[0], minRow1 = minRow[1], minRow2 = minRow[2], maxRow0 = maxRow[0], maxRow1 = maxRow[1], maxRow2 = maxRow[2]]
            ( const AABBTree::SimdBoxes & sb )
        {
            const F lo = add( add( mul( nx, load( sb.box[minRow0] ) ), mul( ny, load( sb.box[minRow1] ) ) ), mul( nz, load( sb.box[minRow2] ) ) );
            const F hi = add( add( mul( nx, load( sb.box[maxRow0] ) ), mul( ny, load( sb.box[maxRow1] ) ) ), mul( nz, load( sb.box[maxRow2] ) ) );
            return bits( andm( le( lo, d ), le( d, hi ) ) );
        }
    };
}

auto xyPlaneBoxOk( float zLevel )
{
    using namespace Simd4;
    return WideBoxTest
    {
        [zLevel]( const Box3f & box )
        {
            return box.min.z <= zLevel && box.max.z >= zLevel;
        },
        [z = set1( zLevel )]( const AABBTree::SimdBoxes & sb )
        {
            return bits( andm( le( load( sb.box[2] ), z ), le( z, load( sb.box[5] ) ) ) );
        }
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
