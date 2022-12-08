#include "MROrder.h"
#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <execution>

namespace MR
{

VertexOrdering getVertexOrdering( const Buffer<FaceId> & invFaceMap, const MeshTopology & topology )
{
    MR_TIMER
    assert( topology.lastValidFace() < invFaceMap.size() );
    VertexOrdering res( topology.vertSize() );

    Timer t( "fill" );

    tbb::parallel_for( tbb::blocked_range<VertId>( 0_v, VertId{ topology.vertSize() } ),
    [&]( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            auto f = ~std::uint32_t(0);
            for ( EdgeId e : orgRing( topology, v ) )
                f = std::min( f, std::uint32_t( invFaceMap[ topology.left( e ) ] ) );
            res[v] = OrderedVertex{ v, f };
        }
    } );

    t.restart( "sort" );
    std::sort( std::execution::par, res.data(), res.data() + res.size() );
    res.resize( topology.numValidVerts() );

    return res;
}

} //namespace MR
