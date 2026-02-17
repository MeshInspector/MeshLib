#include "MRRayBoxIntersection.h"

#include <xsimd/xsimd.hpp>

namespace MR
{

namespace
{

struct RayBoxIntersectSimd
{
    template <typename Arch>
    bool operator()( Arch, const Box3<float>& box, const RayOrigin<float>& rayOrigin, float& t0, float& t1, const IntersectionPrecomputes<float>& prec ) const
    {
        assert( xsimd::is_aligned<Arch>( prec.invDir.data() ) );
        using batch = xsimd::batch<float, Arch>;
        static_assert( batch::size == 4 );
        const batch
            boxMin = { box.min.x, box.min.y, box.min.z, t0 },
            boxMax = { box.max.x, box.max.y, box.max.z, t1 },
            origin = { rayOrigin.p.x, rayOrigin.p.y, rayOrigin.p.z, 0.f },
            invDir = xsimd::load_aligned<Arch>( prec.invDir.data() );

        const auto l = ( boxMin - origin ) * invDir;
        const auto r = ( boxMax - origin ) * invDir;
        const auto a = xsimd::min( l, r );
        const auto b = xsimd::max( l, r );
        t0 = xsimd::reduce_max( a );
        t1 = xsimd::reduce_min( b );

        return t0 <= t1;
    }
};

} // namespace

template <>
bool rayBoxIntersect<float>( const Box3<float>& box, const RayOrigin<float>& rayOrigin, float& t0, float& t1, const IntersectionPrecomputes<float>& prec )
{
#define CALL_FOR( arch ) \
    using archs = xsimd::arch_list<xsimd::arch>; \
    return xsimd::dispatch<archs>( RayBoxIntersectSimd{} )( box, rayOrigin, t0, t1, prec );
#if XSIMD_WITH_SSE2
    CALL_FOR( sse2 )
#elif XSIMD_WITH_NEON64
    CALL_FOR( neon64 )
#elif XSIMD_WITH_WASM
    CALL_FOR( wasm )
#else
    return rayBoxIntersectGeneric( box, rayOrigin, t0, t1, prec );
#endif
}

} // namespace MR
