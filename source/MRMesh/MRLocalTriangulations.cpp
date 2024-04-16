#include "MRLocalTriangulations.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRProgressCallback.h"
#include "MRVector3.h"
#include "MRUnorientedTriangle.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRHeap.h"
#include "MRBitSetParallelFor.h"
#include <parallel_hashmap/phmap.h>
#include <algorithm>
#include <cassert>
#include <cfloat>

namespace MR
{

std::optional<AllLocalTriangulations> uniteLocalTriangulations( const std::vector<SomeLocalTriangulations> & in, const ProgressCallback & progress )
{
    MR_TIMER

    if ( in.empty() )
        return {};

    const VertId maxVertId = std::max_element( in.begin(), in.end(),
        [&]( const SomeLocalTriangulations & a, const SomeLocalTriangulations & b ) { return a.maxCenterId < b.maxCenterId; } )->maxCenterId;

    if ( !reportProgress( progress, 0.0f ) )
        return {};

    AllLocalTriangulations res;
    res.fanRecords.resize( maxVertId + 2 );
    Buffer<const VertId*, VertId> firstNeiSrc( maxVertId + 2 );

    // temporary write in firstNei the _number_ of neighbours and fill firstNeiSrc
    for ( const auto& lt : in )
    {
        for ( int i = 0; i + 1 < lt.fanRecords.size(); ++i )
        {
            const auto v = lt.fanRecords[i].center;
            res.fanRecords[v].border = lt.fanRecords[i].border;
            res.fanRecords[v].firstNei = lt.fanRecords[i+1].firstNei - lt.fanRecords[i].firstNei;
            firstNeiSrc[v] = lt.neighbors.data() + lt.fanRecords[i].firstNei;
        }
    }
    if ( !reportProgress( progress, 0.25f ) )
        return {};

    // rewrite firstNei with the position in res.neighbors
    std::uint32_t numNei = 0;
    for ( auto & fan : res.fanRecords )
    {
        auto num = fan.firstNei;
        fan.firstNei = numNei;
        numNei += num;
    }
    assert( res.fanRecords.back().firstNei == numNei );
    if ( !reportProgress( progress, 0.5f ) )
        return {};

    // copy neighbours of each fan
    res.neighbors.resize( numNei );
    if ( !ParallelFor( 0_v, res.fanRecords.backId(), [&]( VertId v )
    {
        const auto * src = firstNeiSrc[v];
        auto i = res.fanRecords[v].firstNei;
        auto iEnd = res.fanRecords[v + 1].firstNei;
        for ( ; i < iEnd; ++i )
            res.neighbors[i] = *src++;
    }, subprogress( progress, 0.5f, 1.0f ) ) )
        return {};

    return res;
}

Vector3f computeNormal( const AllLocalTriangulations & triangs, const VertCoords & points, VertId v )
{
    assert( v && v + 1 < triangs.fanRecords.size() );
    const auto border = triangs.fanRecords[v].border;
    const auto nbeg = triangs.fanRecords[v].firstNei;
    const auto nend = triangs.fanRecords[v+1].firstNei;
    const auto pv = points[v];
    Vector3f sum;
    for ( auto n = nbeg; n < nend; ++n )
    {
        const auto curr = triangs.neighbors[n];
        if ( curr == border )
            continue;
        const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg]; // in cw order
        auto d0 = points[next] - pv;
        auto d1 = points[curr] - pv;
        auto angle = MR::angle( d0, d1 );
        sum += angle * cross( d0, d1 ).normalized();
    }
    return sum.normalized();
}

void orientLocalTriangulations( AllLocalTriangulations & triangs, const VertCoords & coords, const VertBitSet & region, const VertNormals & targetDir )
{
    return orientLocalTriangulations( triangs, coords, region, [&targetDir]( VertId v ) { return targetDir[v]; } );
}

void orientLocalTriangulations( AllLocalTriangulations & triangs, const VertCoords & coords, const VertBitSet & region, const std::function<Vector3f(VertId)> & targetDir )
{
    MR_TIMER
    if ( triangs.fanRecords.size() <= 1 )
        return;
    BitSetParallelFor( region, [&]( VertId c )
    {
        const auto nbeg = triangs.fanRecords[c].firstNei;
        const auto nend = triangs.fanRecords[c + 1].firstNei;
        if ( nbeg >= nend )
            return;

        const VertId bd = triangs.fanRecords[c].border;
        const Vector3f cp = coords[c];
        int sum = 0;
        VertId otherBd;
        for ( auto n = nbeg; n < nend; ++n )
        {
            const auto curr = triangs.neighbors[n];
            const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg];
            if ( curr == bd )
            {
                otherBd = next;
                continue;
            }
            const auto d = dot( targetDir( c ), cross( coords[next] - cp, coords[curr] - cp ) );
            if ( d > 0 )
                ++sum;
            else if ( d < 0 )
                --sum;
        }
        if ( sum >= 0 )
            return; // already oriented properly
        // reverse the orientation
        std::reverse( triangs.neighbors.data() + nbeg, triangs.neighbors.data() + nend );
        triangs.fanRecords[c].border = otherBd;
    } );
}

struct Repetitions
{
    unsigned char sameOriented : 4 = 0;
    unsigned char oppositeOriented : 4 = 0;
};

static_assert( sizeof( Repetitions ) == 1 );

static ParallelHashMap<UnorientedTriangle, Repetitions> makeTriangleHashMap( const AllLocalTriangulations & triangs )
{
    MR_TIMER

    ParallelHashMap<UnorientedTriangle, Repetitions> map;
    ParallelFor( size_t(0), map.subcnt(), [&]( size_t myPartId )
    {
        for ( VertId v = 0_v; v + 1 < triangs.fanRecords.size(); ++v )
        {
            const auto border = triangs.fanRecords[v].border;
            const auto nbeg = triangs.fanRecords[v].firstNei;
            const auto nend = triangs.fanRecords[v+1].firstNei;
            for ( auto n = nbeg; n < nend; ++n )
            {
                if ( triangs.neighbors[n] == border )
                    continue;
                const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg];
                bool flipped = false;
                const UnorientedTriangle triplet( { v, next, triangs.neighbors[n] }, &flipped );
                const auto hashval = map.hash( triplet );
                const auto idx = map.subidx( hashval );
                if ( idx != myPartId )
                    continue;
                Repetitions & r = map[triplet];
                if ( flipped )
                    ++r.oppositeOriented;
                else
                    ++r.sameOriented;
            }
        }
    } );
    return map;
}

TrianglesRepetitions computeTrianglesRepetitions( const AllLocalTriangulations & triangs )
{
    MR_TIMER

    const auto map = makeTriangleHashMap( triangs );

    TrianglesRepetitions res{};
    for ( auto & [key, val] : map )
    {
        int c = val.sameOriented + val.oppositeOriented;
        assert( c >= 1 && c <= 3 );
        ++res[c];
        if ( val.sameOriented >= 1 && val.oppositeOriented >= 1 )
            ++res[0];
    }
    return res;
}

std::vector<UnorientedTriangle> findRepeatedUnorientedTriangles( const AllLocalTriangulations & triangs, int repetitions )
{
    MR_TIMER
    assert( repetitions >= 1 && repetitions <= 3 );

    const auto map = makeTriangleHashMap( triangs );

    std::vector<UnorientedTriangle> res;
    for ( auto & [key, val] : map )
    {
        int c = val.sameOriented + val.oppositeOriented;
        assert( c >= 1 && c <= 3 );
        if ( c == repetitions )
            res.push_back( key );
    }
    return res;
}

Triangulation findRepeatedOrientedTriangles( const AllLocalTriangulations & triangs, int repetitions )
{
    MR_TIMER
    assert( repetitions >= 1 && repetitions <= 3 );

    const auto map = makeTriangleHashMap( triangs );

    Triangulation res;
    for ( auto & [triplet, r] : map )
    {
        assert( r.sameOriented >= 0 && r.sameOriented <= 3 );
        assert( r.oppositeOriented >= 0 && r.oppositeOriented <= 3 );
        assert( r.sameOriented + r.oppositeOriented >= 1 );
        if ( r.sameOriented == repetitions )
            res.push_back( triplet );
        if ( r.oppositeOriented == repetitions )
            res.push_back( triplet.getFlipped() );
    }
    return res;
}

void findRepeatedOrientedTriangles( const AllLocalTriangulations & triangs, Triangulation * outRep3, Triangulation * outRep2 )
{
    MR_TIMER
    assert( outRep3 || outRep2 );

    const auto map = makeTriangleHashMap( triangs );

    for ( auto & [triplet, r] : map )
    {
        assert( r.sameOriented >= 0 && r.sameOriented <= 3 );
        assert( r.oppositeOriented >= 0 && r.oppositeOriented <= 3 );
        assert( r.sameOriented + r.oppositeOriented >= 1 );
        if ( outRep3 )
        {
            if ( r.sameOriented == 3 )
                outRep3->push_back( triplet );
            else if ( r.oppositeOriented == 3 )
                outRep3->push_back( triplet.getFlipped() );
        }
        if ( outRep2 )
        {
            if ( r.sameOriented == 2 )
                outRep2->push_back( triplet );
            else if ( r.oppositeOriented == 2 )
                outRep2->push_back( triplet.getFlipped() );
        }
    }
}

bool autoOrientLocalTriangulations( const PointCloud & pointCloud, AllLocalTriangulations & triangs,
    const VertBitSet & region, ProgressCallback progress,
    Triangulation * outRep3, Triangulation * outRep2 )
{
    MR_TIMER

    const auto bbox = pointCloud.computeBoundingBox();
    if ( !reportProgress( progress, 0.025f ) )
        return false;

    const auto center = bbox.center();
    const auto maxDistSqToCenter = bbox.size().lengthSq() / 4
        * 1.1f; // make it slightly larger to overcome rounding errors

    constexpr auto InvalidWeight = -FLT_MAX;
    using HeapT = Heap<float, VertId>;
    std::vector<HeapT::Element> elements;
    const auto sz = pointCloud.points.size();
    elements.reserve( sz );
    for ( VertId v = 0_v; v < sz; ++v )
        elements.push_back( { v, InvalidWeight } );

    if ( !reportProgress( progress, 0.025f ) )
        return false;

    // initially orient local triangulations of region points to have normal looking away from the center;
    // this orientation will remain only for most distant points in each connected component not having fixed (out-of-region) points
    orientLocalTriangulations( triangs, pointCloud.points, region, [&]( VertId v )
    {
        return pointCloud.points[v] - center;
    } );

    if ( !reportProgress( progress, 0.05f ) )
        return false;

    // fill elements with negative weights: larger weight (smaller by magnitude) for points further from the center
    if ( !BitSetParallelFor( pointCloud.validPoints, [&]( VertId v )
    {
        if ( !region.test( v ) )
        {
            elements[(int)v].val = InvalidWeight;
            return;
        }
        const auto dcenter = pointCloud.points[v] - center;
        const auto w = dcenter.lengthSq() - maxDistSqToCenter;
        assert( w <= 0 );
        elements[(int)v].val = w;
    }, subprogress( progress, 0.05f, 0.075f ) ) )
        return false;

    HeapT heap( std::move( elements ) );

    if ( !reportProgress( progress, 0.1f ) )
        return false;

    progress = subprogress( progress, 0.1f, 1.0f );

    // HashMap is about 10% faster than ParallelHashMap here
    HashMap<UnorientedTriangle, Repetitions> map;

    auto computeVertWeight = [&triangs, &map]( VertId v )
    {
        int sameOriented = 0;
        int oppositeOriented = 0;
        const auto border = triangs.fanRecords[v].border;
        const auto nbeg = triangs.fanRecords[v].firstNei;
        const auto nend = triangs.fanRecords[v+1].firstNei;
        VertId otherBd;
        for ( auto n = nbeg; n < nend; ++n )
        {
            const auto curr = triangs.neighbors[n];
            const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg];
            if ( curr == border )
            {
                otherBd = next;
                continue;
            }
            bool flipped = false;
            const UnorientedTriangle triplet( { v, next, curr }, &flipped );
            auto it = map.find( triplet );
            if ( it == map.end() )
                continue;
            if ( it->second.sameOriented == 0 && it->second.oppositeOriented > 0 )
                flipped ? ++sameOriented : ++oppositeOriented;
            if ( it->second.sameOriented > 0 && it->second.oppositeOriented == 0 )
                flipped ? ++oppositeOriented : ++sameOriented;
        }
        if ( oppositeOriented > sameOriented )
        {
            // reverse the orientation
            std::reverse( triangs.neighbors.data() + nbeg, triangs.neighbors.data() + nend );
            triangs.fanRecords[v].border = otherBd;
        }
        return std::abs( sameOriented - oppositeOriented );
    };

    // notVisited are points that can change orientation of their local triangulation
    VertBitSet notVisited = region;
    
    auto enqueueNeighbors = [&]( VertId base )
    {
        const auto nbeg = triangs.fanRecords[base].firstNei;
        const auto nend = triangs.fanRecords[base+1].firstNei;
        const auto border = triangs.fanRecords[base].border;
        for ( auto n = nbeg; n < nend; ++n )
        {
            const auto curr = triangs.neighbors[n];
            const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg];
            if ( curr == border )
                continue;
            bool flipped = false;
            const UnorientedTriangle triplet( { base, next, curr }, &flipped );

            /// all three vertices of the triangle have been visited, it will never be searched for again
            const bool del = !notVisited.test( curr ) && !notVisited.test( next );
            if ( !outRep2 && !outRep3 && del )
            {
                map.erase( triplet );
                continue;
            }
            HashMap<UnorientedTriangle, Repetitions>::iterator it;
            if ( del )
            {
                it = map.find( triplet );
                if ( it == map.end() )
                    continue; // no record in the map exists, so the number of repetitions is 1
            }
            else
                it = map.insert( { triplet, Repetitions{} } ).first;
            Repetitions & r = it->second;
            if ( flipped )
                ++r.oppositeOriented;
            else
                ++r.sameOriented;
            if ( del )
            {
                if ( outRep2 )
                {
                    if ( r.sameOriented == 2 )
                        outRep2->push_back( triplet );
                    else if ( r.oppositeOriented == 2 )
                        outRep2->push_back( triplet.getFlipped() );
                }
                if ( outRep3 )
                {
                    if ( r.sameOriented == 3 )
                        outRep3->push_back( triplet );
                    else if ( r.oppositeOriented == 3 )
                        outRep3->push_back( triplet.getFlipped() );
                }
                /// all three vertices of the triangle have been visited, it will never be searched for again
                map.erase( it );
            }
        }
        for ( auto n = nbeg; n < nend; ++n )
        {
            const auto v = triangs.neighbors[n];
            if ( notVisited.test( v ) )
                heap.setValue( v, float( computeVertWeight( v ) ) );
        }
    };

    // both in- and out-of- region
    const auto totalCount = pointCloud.validPoints.count();
    size_t visitedCount = 0;

    // process out-of-region points with fixed orientation of local triangulations
    for ( auto v : pointCloud.validPoints - region )
    {
        assert( heap.value( v ) == InvalidWeight );
        assert( !notVisited.test( v ) );
        ++visitedCount;
        enqueueNeighbors( v );
        if ( !reportProgress( progress, [&] { return (float)visitedCount / totalCount; }, visitedCount, 0x10000 ) )
            return false;
    }

    // orient local triangulations of region points
    for (;;)
    {
        auto [v, weight] = heap.top();
        if ( weight == InvalidWeight )
            break;
        heap.setSmallerValue( v, InvalidWeight );
        assert( notVisited.test( v ) );
        notVisited.reset( v );
        ++visitedCount;
        enqueueNeighbors( v );
        if ( !reportProgress( progress, [&] { return (float)visitedCount / totalCount; }, visitedCount, 0x10000 ) )
            return false;
    }
    assert( visitedCount == totalCount );

    if ( outRep2 )
    {
        for ( const auto & [triplet, r] : map )
        {
            assert( r.sameOriented < 3 );
            assert( r.oppositeOriented < 3 );
            assert( r.sameOriented + r.oppositeOriented >= 1 );
            if ( r.sameOriented == 2 )
                outRep2->push_back( triplet );
            else if ( r.oppositeOriented == 2 )
                outRep2->push_back( triplet.getFlipped() );
        }
    }

    return true;
}

} //namespace MR
