#include "MRIterativeSampling.h"
#include "MRPointCloud.h"
#include "MRHeap.h"
#include "MRBuffer.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRTimer.h"
#include <cfloat>

#include "MRGTest.h"
#include "MRTorus.h"
#include "MRMesh.h"
#include "MRMeshToPointCloud.h"

namespace MR
{

namespace
{

// an element of heap
struct PointInfo
{
    /// squared distance to the closest neighbor
    float closeDistSq;
    /// sum of squared distances to nearby removed (not-sampled) points
    float sumDelDistSq;

    // points with minimal sum of distances are first removed from samples
    float sumDistSq() const { return closeDistSq + sumDelDistSq; }
    bool operator <( const PointInfo & b ) const { return sumDistSq() > b.sumDistSq(); }

    explicit PointInfo( NoInit ) noexcept {}

    // invalid info, last in heap
    PointInfo() : closeDistSq( FLT_MAX ), sumDelDistSq( FLT_MAX ) {}
};

} //anonymous namespace

std::optional<VertBitSet> pointIterativeSampling( const PointCloud& cloud, int numSamples, const ProgressCallback & cb )
{
    MR_TIMER
    VertBitSet res = cloud.validPoints;
    auto toRemove = (int)res.count() - numSamples;
    if ( toRemove <= 0 )
        return res;

    const auto sz = cloud.validPoints.size();
    Buffer<VertId, VertId> closestNei( sz );
    Buffer<PointInfo, VertId> info( sz );
    cloud.getAABBTree();
    BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        const auto prj = findProjectionOnPoints( cloud.points[v], cloud, FLT_MAX, nullptr, 0, [v]( VertId x ) { return v == x; } );
        closestNei[v] = prj.vId;
        info[v].closeDistSq = prj.distSq;
        info[v].sumDelDistSq = 0;
    } );

    if ( !reportProgress( cb, 0.1f ) )
        return {};

    Vector<VertId, VertId> first( sz ); ///< first[v] contains a pointId having closest point v
    Buffer<VertId, VertId> next( sz );  ///< next[v] contains pointId having the same closest point as v's closest point
    for ( auto v : cloud.validPoints )
    {
        const auto cv = closestNei[v];
        next[v] = first[cv];
        first[cv] = v;
    }

    if ( !reportProgress( cb, 0.2f ) )
        return {};

    using HeapT = Heap<PointInfo, VertId>;
    std::vector<HeapT::Element> elms;
    elms.reserve( numSamples + toRemove );
    for ( auto vid : cloud.validPoints )
        elms.push_back( { vid, info[vid] } );
    info.clear();
    HeapT heap( std::move( elms ) );

    if ( !reportProgress( cb, 0.3f ) )
        return {};

    const auto k = 1.0f / toRemove;
    while ( toRemove > 0 )
    {
        auto [v, vinfo] = heap.top();
        assert( vinfo.closeDistSq < FLT_MAX );
        --toRemove;
        assert( res.test( v ) );
        res.reset( v );
        heap.setSmallerValue( v, PointInfo() );

        auto cv = closestNei[v];
        auto cvinfo = heap.value( cv );
        cvinfo.sumDelDistSq += vinfo.sumDistSq();
        heap.setSmallerValue( cv, cvinfo );

        auto nr = first[v];
        for ( auto r = nr; nr = (r ? next[r] : r), r; r = nr )
        {
            if ( !res.test( r ) )
                continue;
            const auto prj = findProjectionOnPoints( cloud.points[r], cloud, FLT_MAX, nullptr, 0,
                [&res, r]( VertId x ) { return x == r || !res.test( x ); } );
            assert( closestNei[r] == v );
            const auto cr = closestNei[r] = prj.vId;
            auto rinfo = heap.value( r );
            assert( rinfo.closeDistSq < FLT_MAX );
            assert( rinfo.closeDistSq <= prj.distSq );
            if ( rinfo.closeDistSq < prj.distSq )
            {
                rinfo.closeDistSq = prj.distSq;
                heap.setSmallerValue( r, rinfo );
            }
            next[r] = first[cr];
            first[cr] = r;
        }
        if ( !reportProgress( cb, [&] { return 0.3f + 0.7f * ( 1 - k * toRemove ); }, toRemove, 0x10000 ) )
            return {};
    }

    if ( !reportProgress( cb, 1.0f ) )
        return {};
    return res;
}

TEST( MRMesh, IterativeSampling )
{
    auto cloud = meshToPointCloud( makeTorus() );
    auto numSamples = (int)cloud.validPoints.count() / 2;
    auto optSamples = pointIterativeSampling( cloud, numSamples );
    EXPECT_EQ( numSamples, optSamples->count() );
}

} //namespace MR
