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
                for ( int i = ~st.ref * AABBTree::MaxLeafTris, end = i + AABBTree::MaxLeafTris; i < end; ++i )
                {
                    const auto & lt = leafTris[i];
                    if ( !lt.face )
                        break;
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

} // anonymous namespace

// whether the float rays test the boxes of the triangles of a leaf before the triangles: on Arm (Apple M1) the tests of the triangles alone are faster
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr bool cRayTestsLeafBoxes = false;
#else
constexpr bool cRayTestsLeafBoxes = true;
#endif

/// the float version of meshRayIntersectT_ over the SIMD layout of the tree: tests the boxes of all children of a node or all triangles of a leaf at once;
/// visits the nodes and the triangles in the same order and computes the same values as meshRayIntersectT_, so it finds the same intersection
template<bool Filtered>
MeshIntersectionResult meshRayIntersectSimd_( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Line3f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    using namespace Simd4;
    MeshIntersectionResult res;
    if ( tree.empty() )
        return res;
    const auto& simdNodes = tree.simdNodes();
    const auto& leafTris = tree.leafTris();

    RayOrigin<float> rayOrigin{ line.p };
    {
        float s = rayStart, e = rayEnd;
        if ( !rayBoxIntersect( tree.getBoundingBox(), rayOrigin, s, e, prec ) )
            return res;
    }

    const SimdRayBoxes rayBoxes( line, prec );

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
                bool stop = false;
                // the triangles with crossed boxes (or all on Arm) in their order, as rayStart and rayEnd change after each intersection;
                // a triangle is inside its box, so the others cannot be crossed
                {
                    const int first = ~st.ref * AABBTree::MaxLeafTris;
                    const int tested = cRayTestsLeafBoxes ? rayBoxes.test( tree.leafBoxes( ~st.ref ), rayStart, rayEnd ) : 15;
                    for ( int crossed = tested; crossed; crossed &= crossed - 1 )
                    {
                        const auto & lt = leafTris[first + lowestBit( crossed )];
                        if ( !cRayTestsLeafBoxes && !lt.face )
                            break;
                        if constexpr ( Filtered )
                        {
                            if ( ( region && !region->test( lt.face ) ) || ( validFaces && !validFaces( lt.face ) ) )
                                continue;
                        }
                        const auto triIsect = rayTriangleIntersect( points[lt.verts[0]] - line.p, points[lt.verts[1]] - line.p, points[lt.verts[2]] - line.p, prec );
                        if ( !triIsect )
                            continue;
                        const float ti = triIsect->t;
                        if ( !( ti < rayEnd && ti > rayStart ) )
                            continue;
                        distance = ti;
                        face = lt.face;
                        triP = triIsect->bary;
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
            return meshRayIntersectSimd_<true>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
        return meshRayIntersectSimd_<false>( points, tree, region, line, rayStart, rayEnd, prec, closestIntersect, validFaces );
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
    // computing the same as rayBoxIntersect; simdCrossed returns the bit mask of the crossed boxes
    [[maybe_unused]] std::optional<SimdRayBoxes> simdRayBoxes;
    if constexpr ( std::is_same_v<T, float> )
        simdRayBoxes.emplace( line, prec );
    [[maybe_unused]] auto simdCrossed = [&]( const AABBTree::SimdBoxes & b ) { return simdRayBoxes->test( b, float( rayStart ), float( rayEnd ) ); };

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
        // the ends of the segment get the ids after all vertices: simulation-of-simplicity only compares the ids
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
            const auto & leafBoxes = tree.leafBoxes( ~r );
            // the float version tests the boxes of all triangles of the leaf at once (except on Arm)
            [[maybe_unused]] int crossedBoxes = 15;
            if constexpr ( std::is_same_v<T, float> && cRayTestsLeafBoxes )
            {
                crossedBoxes = simdCrossed( leafBoxes );
                if ( !crossedBoxes )
                    continue;
            }
            for ( int k = 0; k < AABBTree::MaxLeafTris; ++k )
            {
                const int slot = ~r * AABBTree::MaxLeafTris + k;
                const auto & lt = leafTris[slot];
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
                    // float version (faster)
                    if ( !( crossedBoxes & ( 1 << k ) ) )
                        continue;
                    const Vector3f vA = points[pvc[0].id] - line.p;
                    const Vector3f vB = points[pvc[1].id] - line.p;
                    const Vector3f vC = points[pvc[2].id] - line.p;
                    if ( const auto triIsect = rayTriangleIntersect( vA, vB, vC, prec ) )
                    {
                        if ( triIsect->t < rayEnd && triIsect->t > rayStart )
                        {
                            MeshIntersectionResult found;
                            found.proj.face = face;
                            found.proj.point = Vector3f( line( triIsect->t ) );
                            found.tp = triIsect->bary;
                            found.distanceAlongLine = triIsect->t;
                            if ( !report( found, [&] { return dot( cross( vB - vA, vC - vA ), line.d ) < 0; } ) )
                                return;
                        }
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
