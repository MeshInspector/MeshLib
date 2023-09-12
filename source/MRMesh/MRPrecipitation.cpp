#include "MRWatershedGraph.h"
#include "MRHeap.h"
#include "MRTimer.h"

namespace MR
{

void simulatePrecipitation( WatershedGraph & wg )
{
    MR_TIMER
    constexpr auto infTime = FLT_MAX;
    Heap<float, Graph::VertId, std::greater<float>> heap( wg.numBasins(), infTime );
    for ( auto basin = Graph::VertId( 0 ); basin < wg.numBasins(); ++basin )
    {
        const auto & info = wg.basinInfo( basin );
        heap.setValue( basin, info.timeTillOverflow() );
    }
}

} //namespace MR
