#include "MRPrecipitationSimulator.h"
#include "MRWatershedGraph.h"
#include "MRTimer.h"

namespace MR
{

PrecipitationSimulator::PrecipitationSimulator( WatershedGraph & wg )
    : wg_( wg )
    , heap_( wg_.numBasins(), infTime )
{
    MR_TIMER

    for ( auto basin = Graph::VertId( 0 ); basin < wg.numBasins(); ++basin )
    {
        const auto& info = wg.basinInfo( basin );
        heap_.setValue( basin, info.timeTillOverflow() );
    }
}

auto PrecipitationSimulator::simulateOne() -> SimulationStep
{
    SimulationStep res;
    const auto [basin, time] = heap_.top();
    if ( time == infTime )
        return res;
    res.time = time;

    auto& info = wg_.basinInfo( basin );
    assert( !info.overflowTo );
    info.lastUpdateTime = time;
    info.remVolume = 0;

    for ( auto bd : wg_.graph().neighbours( basin ) )
    {
        const auto & bdInfo = wg_.bdInfo( bd );
        if ( wg_.getHeightAt( bdInfo.lowestVert ) != info.lowestBdLevel )
            continue;
        const auto neiBasin = wg_.graph().ends( bd ).otherEnd( basin );
        res.basin = basin;
        res.neiBasin = neiBasin;
        auto targetBasin = wg_.flowsTo( neiBasin );
        if ( targetBasin == basin )
        {
            res.event = Event::Merge;
            auto& neiInfo = wg_.basinInfo( neiBasin );
            neiInfo.update( time );
            wg_.merge( basin, neiBasin );
            heap_.setSmallerValue( neiBasin, infTime );
            heap_.setSmallerValue( basin, time + info.timeTillOverflow() );
            return res;
        }
        if ( targetBasin != wg_.outsideId() )
        {
            auto& targetInfo = wg_.basinInfo( targetBasin );
            targetInfo.update( time );
            targetInfo.area += info.area;
            heap_.setLargerValue( targetBasin, time + targetInfo.timeTillOverflow() );
        }
        heap_.setSmallerValue( basin, infTime );
        info.area = 0;
        info.overflowTo = neiBasin;
        res.event = Event::BasinFull;
        return res;
    }
    assert( false );
    heap_.setSmallerValue( basin, infTime );
    return res;
}

} //namespace MR
