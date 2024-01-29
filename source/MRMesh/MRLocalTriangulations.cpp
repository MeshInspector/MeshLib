#include "MRLocalTriangulations.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRProgressCallback.h"
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

} //namespace MR
