// Prototype: MeshLib's AABBTree re-laid out for 4-wide SIMD ray traversal.
// 1) every internal node keeps the boxes of its (up to 4) children in SoA order and tests them in one pass;
// 2) every leaf keeps the vertex coordinates of its (up to 4) triangles in SoA order and tests them in one pass
//    with the same watertight test as MR::rayTriangleIntersect.
// The tree shape is exactly MeshLib's (built by AABBTree), only the memory layout and the inner loops differ.
#pragma once

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#else
#error "SoaRayTree prototype supports x86-64 and ARM64 only"
#endif

namespace SoaProto
{

#if defined(__x86_64__) || defined(_M_X64)
using V4 = __m128;
using M4 = __m128;
inline V4 ld( const float* p ) { return _mm_load_ps( p ); }
inline V4 set1( float x ) { return _mm_set1_ps( x ); }
inline V4 add( V4 a, V4 b ) { return _mm_add_ps( a, b ); }
inline V4 sub( V4 a, V4 b ) { return _mm_sub_ps( a, b ); }
inline V4 mul( V4 a, V4 b ) { return _mm_mul_ps( a, b ); }
inline V4 vdiv( V4 a, V4 b ) { return _mm_div_ps( a, b ); }
inline V4 vmin( V4 a, V4 b ) { return _mm_min_ps( a, b ); }
inline V4 vmax( V4 a, V4 b ) { return _mm_max_ps( a, b ); }
inline V4 vabs( V4 a ) { return _mm_andnot_ps( _mm_set1_ps( -0.0f ), a ); }
inline M4 lt( V4 a, V4 b ) { return _mm_cmplt_ps( a, b ); }
inline M4 le( V4 a, V4 b ) { return _mm_cmple_ps( a, b ); }
inline M4 gt( V4 a, V4 b ) { return _mm_cmpgt_ps( a, b ); }
inline M4 ne( V4 a, V4 b ) { return _mm_cmpneq_ps( a, b ); }
inline M4 mand( M4 a, M4 b ) { return _mm_and_ps( a, b ); }
inline M4 mor( M4 a, M4 b ) { return _mm_or_ps( a, b ); }
inline M4 mxor( M4 a, M4 b ) { return _mm_xor_ps( a, b ); }
inline int bits( M4 m ) { return _mm_movemask_ps( m ); }
inline void st( float* p, V4 a ) { _mm_store_ps( p, a ); }
#else
using V4 = float32x4_t;
using M4 = uint32x4_t;
inline V4 ld( const float* p ) { return vld1q_f32( p ); }
inline V4 set1( float x ) { return vdupq_n_f32( x ); }
inline V4 add( V4 a, V4 b ) { return vaddq_f32( a, b ); }
inline V4 sub( V4 a, V4 b ) { return vsubq_f32( a, b ); }
inline V4 mul( V4 a, V4 b ) { return vmulq_f32( a, b ); }
inline V4 vdiv( V4 a, V4 b ) { return vdivq_f32( a, b ); }
inline V4 vmin( V4 a, V4 b ) { return vminq_f32( a, b ); }
inline V4 vmax( V4 a, V4 b ) { return vmaxq_f32( a, b ); }
inline V4 vabs( V4 a ) { return vabsq_f32( a ); }
inline M4 lt( V4 a, V4 b ) { return vcltq_f32( a, b ); }
inline M4 le( V4 a, V4 b ) { return vcleq_f32( a, b ); }
inline M4 gt( V4 a, V4 b ) { return vcgtq_f32( a, b ); }
inline M4 ne( V4 a, V4 b ) { return vmvnq_u32( vceqq_f32( a, b ) ); }
inline M4 mand( M4 a, M4 b ) { return vandq_u32( a, b ); }
inline M4 mor( M4 a, M4 b ) { return vorrq_u32( a, b ); }
inline M4 mxor( M4 a, M4 b ) { return veorq_u32( a, b ); }
inline int bits( M4 m )
{
    static const uint32_t w[4] = { 1, 2, 4, 8 };
    return int( vaddvq_u32( vandq_u32( m, vld1q_u32( w ) ) ) );
}
inline void st( float* p, V4 a ) { vst1q_f32( p, a ); }
#endif

inline int lowestBit4( int m ) { return ( m & 1 ) ? 0 : ( m & 2 ) ? 1 : ( m & 4 ) ? 2 : 3; }

// child reference: >= 0 is an internal node, < 0 is ~leaf index
struct alignas( 16 ) Node
{
    float b[6][4]; // [0..2] min x,y,z of the 4 children, [3..5] max x,y,z
    int child[4];
};
static_assert( sizeof( Node ) == 112 );

struct alignas( 16 ) Leaf
{
    float v[3][3][4]; // [vertex][coordinate][triangle]
    int face[4];      // -1 in unused lanes (their coordinates are NaN, so they never hit)
};
static_assert( sizeof( Leaf ) == 160 );

struct Tree
{
    std::vector<Node> nodes;
    std::vector<Leaf> leaves;
    int root = 0;

    size_t heapBytes() const { return nodes.capacity() * sizeof( Node ) + leaves.capacity() * sizeof( Leaf ); }

    explicit Tree( const MR::Mesh& mesh )
    {
        const auto& t = mesh.getAABBTree();
        const auto& src = t.nodes();
        const int n = int( src.size() );
        if ( n == 0 )
            return;
        std::vector<int> ref( n );
        int ni = 0, nl = 0;
        for ( int i = 0; i < n; ++i )
            ref[i] = src[MR::NodeId( i )].leaf() ? ~( nl++ ) : ni++;
        nodes.resize( ni );
        leaves.resize( nl );
        root = ref[0];
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        tbb::parallel_for( tbb::blocked_range<int>( 0, n, 4096 ), [&]( const tbb::blocked_range<int>& r )
        {
            for ( int i = r.begin(); i < r.end(); ++i )
            {
                const auto& s = src[MR::NodeId( i )];
                if ( s.leaf() )
                {
                    Leaf& l = leaves[~ref[i]];
                    for ( int k = 0; k < 4; ++k )
                    {
                        const bool used = k < s.numLeafTris();
                        const auto* lt = used ? &t.leafTris()[s.first + k] : nullptr;
                        l.face[k] = used ? int( lt->face ) : -1;
                        for ( int v = 0; v < 3; ++v )
                        {
                            const MR::Vector3f p = used ? mesh.points[lt->verts[v]] : MR::Vector3f( nan, nan, nan );
                            for ( int c = 0; c < 3; ++c )
                                l.v[v][c][k] = p[c];
                        }
                    }
                }
                else
                {
                    Node& d = nodes[ref[i]];
                    for ( int k = 0; k < 4; ++k )
                    {
                        const bool used = k < s.count;
                        const auto* c = used ? &src[MR::NodeId( s.first + k )] : nullptr;
                        d.child[k] = used ? ref[s.first + k] : 0;
                        for ( int a = 0; a < 3; ++a )
                        {
                            d.b[a][k] = used ? c->box.min[a] : inf;
                            d.b[a + 3][k] = used ? c->box.max[a] : -inf;
                        }
                    }
                }
            }
        } );
    }
};

struct Hit
{
    float t = std::numeric_limits<float>::infinity();
    int face = -1;
    float u = 0, v = 0; // barycentrics of the hit point as in MR::TriPointf
};

// closest (or any) intersection of the ray org + t * dir with t in (rayStart, rayEnd), rayStart >= 0
template <bool Closest>
inline Hit intersect( const Tree& tree, const MR::Vector3f& org, const MR::Vector3f& dir, float rayStart = 0, float rayEnd = FLT_MAX )
{
    Hit res;
    if ( tree.nodes.empty() && tree.leaves.empty() )
        return res;

    // ray precomputes, the same as in MR::IntersectionPrecomputes
    int kx, ky, kz;
    MR::findMaxVectorDim( kx, ky, kz, dir );
    const float Sxs = dir[kx] / dir[kz], Sys = dir[ky] / dir[kz], Szs = 1.0f / dir[kz];
    float rd[3];
    int nearIdx[3], farIdx[3];
    for ( int a = 0; a < 3; ++a )
    {
        rd[a] = dir[a] == 0 ? FLT_MAX : 1 / dir[a];
        nearIdx[a] = dir[a] >= 0 ? a : a + 3;
        farIdx[a] = dir[a] >= 0 ? a + 3 : a;
    }
    const V4 ox = set1( org.x ), oy = set1( org.y ), oz = set1( org.z );
    const V4 rdx = set1( rd[0] ), rdy = set1( rd[1] ), rdz = set1( rd[2] );
    const V4 tMin = set1( rayStart );
    const V4 okx = set1( org[kx] ), oky = set1( org[ky] ), okz = set1( org[kz] );
    const V4 Sx = set1( Sxs ), Sy = set1( Sys ), Sz = set1( Szs );
    const V4 epsV = set1( std::numeric_limits<float>::epsilon() );
    const V4 zero = set1( 0 );

    struct Entry
    {
        int ref;
        float tnear;
    };
    Entry stack[64];
    int sp = 0;
    int cur = tree.root;
    alignas( 16 ) float tn[4];
    for ( ;; )
    {
        if ( cur >= 0 )
        {
            const Node& nd = tree.nodes[cur];
            const V4 tMax = set1( rayEnd );
            const V4 tx0 = mul( sub( ld( nd.b[nearIdx[0]] ), ox ), rdx );
            const V4 ty0 = mul( sub( ld( nd.b[nearIdx[1]] ), oy ), rdy );
            const V4 tz0 = mul( sub( ld( nd.b[nearIdx[2]] ), oz ), rdz );
            const V4 tx1 = mul( sub( ld( nd.b[farIdx[0]] ), ox ), rdx );
            const V4 ty1 = mul( sub( ld( nd.b[farIdx[1]] ), oy ), rdy );
            const V4 tz1 = mul( sub( ld( nd.b[farIdx[2]] ), oz ), rdz );
            const V4 tNear = vmax( vmax( tx0, ty0 ), vmax( tz0, tMin ) );
            const V4 tFar = vmin( vmin( tx1, ty1 ), vmin( tz1, tMax ) );
            int m = bits( le( tNear, tFar ) );
            if ( m )
            {
                if ( ( m & ( m - 1 ) ) == 0 )
                {
                    cur = nd.child[lowestBit4( m )];
                    continue;
                }
                st( tn, tNear );
                // the hit children sorted by entry distance, nearest first
                Entry e[4];
                int c = 0;
                for ( ; m; m &= m - 1 )
                {
                    const int k = lowestBit4( m );
                    Entry x{ nd.child[k], tn[k] };
                    int j = c++;
                    for ( ; j > 0 && e[j - 1].tnear > x.tnear; --j )
                        e[j] = e[j - 1];
                    e[j] = x;
                }
                for ( int j = c - 1; j > 0; --j )
                    stack[sp++] = e[j];
                cur = e[0].ref;
                continue;
            }
        }
        else
        {
            const Leaf& lf = tree.leaves[~cur];
            // vertices relative to the ray origin, sheared as in MR::rayTriangleIntersect
            const V4 aZ = sub( ld( lf.v[0][kz] ), okz ), bZ = sub( ld( lf.v[1][kz] ), okz ), cZ = sub( ld( lf.v[2][kz] ), okz );
            const V4 Ax = sub( sub( ld( lf.v[0][kx] ), okx ), mul( Sx, aZ ) );
            const V4 Ay = sub( sub( ld( lf.v[0][ky] ), oky ), mul( Sy, aZ ) );
            const V4 Bx = sub( sub( ld( lf.v[1][kx] ), okx ), mul( Sx, bZ ) );
            const V4 By = sub( sub( ld( lf.v[1][ky] ), oky ), mul( Sy, bZ ) );
            const V4 Cx = sub( sub( ld( lf.v[2][kx] ), okx ), mul( Sx, cZ ) );
            const V4 Cy = sub( sub( ld( lf.v[2][ky] ), oky ), mul( Sy, cZ ) );
            const V4 eps = mul( mul( epsV, vmax( vmax( vabs( Ax ), vabs( Bx ) ), vabs( Cx ) ) ),
                vmax( vmax( vabs( Ay ), vabs( By ) ), vabs( Cy ) ) );
            const V4 U = sub( mul( Cx, By ), mul( Cy, Bx ) );
            const V4 V = sub( mul( Ax, Cy ), mul( Ay, Cx ) );
            const V4 W = sub( mul( Bx, Ay ), mul( By, Ax ) );
            const V4 nEps = sub( zero, eps );
            const M4 someNeg = mor( mor( lt( U, nEps ), lt( V, nEps ) ), lt( W, nEps ) );
            const M4 somePos = mor( mor( gt( U, eps ), gt( V, eps ) ), gt( W, eps ) );
            const V4 det = add( add( U, V ), W );
            M4 ok = mand( mxor( someNeg, somePos ), ne( det, zero ) );
            if ( bits( ok ) )
            {
                const V4 T = add( add( mul( U, mul( Sz, aZ ) ), mul( V, mul( Sz, bZ ) ) ), mul( W, mul( Sz, cZ ) ) );
                const V4 invDet = vdiv( set1( 1 ), det );
                const V4 t = mul( T, invDet );
                ok = mand( ok, mand( gt( t, tMin ), lt( t, set1( rayEnd ) ) ) );
                int m = bits( ok );
                if ( m )
                {
                    alignas( 16 ) float ts[4], vs[4], ws[4], ids[4];
                    st( ts, t );
                    int best = lowestBit4( m );
                    for ( int mm = m & ( m - 1 ); mm; mm &= mm - 1 )
                    {
                        const int k = lowestBit4( mm );
                        if ( ts[k] < ts[best] )
                            best = k;
                    }
                    st( vs, V );
                    st( ws, W );
                    st( ids, invDet );
                    res.t = rayEnd = ts[best];
                    res.face = lf.face[best];
                    res.u = vs[best] * ids[best];
                    res.v = ws[best] * ids[best];
                    if ( !Closest || rayEnd == 0 )
                        return res;
                }
            }
        }
        // pop the nearest pending subtree that can still contain a closer hit
        for ( ;; )
        {
            if ( sp == 0 )
                return res;
            const Entry& e = stack[--sp];
            if ( e.tnear < rayEnd )
            {
                cur = e.ref;
                break;
            }
        }
    }
}

} // namespace SoaProto
