#include "MRLocalTriangulations.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRProgressCallback.h"
#include "MRVector3.h"
#include "MRUnorientedTriangle.h"
#include <parallel_hashmap/phmap.h>
#include <algorithm>
#include <cassert>

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

void orientLocalTriangulations( AllLocalTriangulations & triangs, const VertCoords & coords, const VertNormals & normals )
{
    MR_TIMER
    if ( triangs.fanRecords.size() <= 1 )
        return;
    ParallelFor( 0_v, triangs.fanRecords.backId(), [&]( VertId c )
    {
        const auto nbeg = triangs.fanRecords[c].firstNei;
        const auto nend = triangs.fanRecords[c + 1].firstNei;
        if ( nbeg >= nend )
            return;

        const VertId bd = triangs.fanRecords[c].border;
        const Vector3f cp = coords[c];
        Vector3f sum;
        VertId otherBd;
        for ( auto n = nbeg; n < nend; ++n )
        {
            const auto curr = triangs.neighbors[n];
            const auto next = triangs.neighbors[n + 1 < nend ? n + 1 : nbeg];
            if ( curr == bd )
            {
                otherBd = bd;
                continue;
            }
            sum += cross( coords[next] - cp, coords[curr] - cp );
        }
        if ( dot( sum, normals[c] ) >= 0 )
            return; // already oriented properly
        // reverse the orientation
        std::reverse( triangs.neighbors.data() + nbeg, triangs.neighbors.data() + nend );
        triangs.fanRecords[c].border = otherBd;
    } );
}

static ParallelHashMap<UnorientedTriangle, int, UnorientedTriangleHasher> makeTriangleHashMap( const AllLocalTriangulations & triangs )
{
    MR_TIMER

    ParallelHashMap<UnorientedTriangle, int, UnorientedTriangleHasher> map;
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
                const UnorientedTriangle triplet({ v, next, triangs.neighbors[n] });
                const auto hashval = map.hash( triplet );
                const auto idx = map.subidx( hashval );
                if ( idx != myPartId )
                    continue;
                auto [it, inserted] = map.insert( { triplet, 0 } );
                if ( !inserted )
                    ++it->second;
            }
        }
    } );
    return map;
}

VotesForTriangles computeTrianglesRepetitions( const AllLocalTriangulations & triangs )
{
    MR_TIMER

    const auto map = makeTriangleHashMap( triangs );

    VotesForTriangles res{};
    for ( auto & [key, val] : map )
    {
        assert( val >= 0 && val <= 2 );
        ++res[val];
    }
    return res;
}

std::vector<UnorientedTriangle> findRepeatedTriangles( const AllLocalTriangulations & triangs, int repetitions )
{
    MR_TIMER
    assert( repetitions >= 0 && repetitions <= 2 );

    const auto map = makeTriangleHashMap( triangs );

    std::vector<UnorientedTriangle> res;
    for ( auto & [key, val] : map )
    {
        assert( val >= 0 && val <= 2 );
        if ( val == repetitions )
            res.push_back( key );
    }
    return res;
}

} //namespace MR
