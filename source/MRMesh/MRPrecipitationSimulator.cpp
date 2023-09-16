#include "MRPrecipitationSimulator.h"
#include "MRWatershedGraph.h"
#include "MRTimer.h"

namespace MR
{

static constexpr float infAmount = FLT_MAX;

PrecipitationSimulator::PrecipitationSimulator( WatershedGraph & wg )
    : wg_( wg )
    , heap_( wg_.numBasins(), infAmount )
{
    MR_TIMER

    for ( auto basin = Graph::VertId( 0 ); basin < wg.numBasins(); ++basin )
    {
        const auto& info = wg.basinInfo( basin );
        heap_.setValue( basin, info.amountTillOverflow() );
    }
}

auto PrecipitationSimulator::simulateOne() -> SimulationStep
{
    SimulationStep res;
    const auto [basin, amount] = heap_.top();
    if ( amount == infAmount )
        return res;
    res.amount = amount;

    auto& info = wg_.basinInfo( basin );
    assert( !info.overflowVia );
    info.lastUpdateAmount = amount;
    info.accVolume = info.maxVolume;

    for ( auto bd : wg_.graph().neighbours( basin ) )
    {
        const auto & bdInfo = wg_.bdInfo( bd );
        if ( wg_.getHeightAt( bdInfo.lowestVert ) != info.lowestBdLevel )
            continue;
        const auto neiBasin = wg_.graph().ends( bd ).otherEnd( basin );
        res.basin = basin;
        res.neiBasin = neiBasin;
        auto targetBasin = wg_.flowsFinallyTo( neiBasin );
        if ( targetBasin == basin )
        {
            res.event = Event::Merge;
            auto& neiInfo = wg_.basinInfo( neiBasin );
            neiInfo.updateAccVolume( amount );
            wg_.merge( basin, neiBasin );
            heap_.setSmallerValue( neiBasin, infAmount );
            heap_.setSmallerValue( basin, amount + info.amountTillOverflow() );
            return res;
        }
        if ( targetBasin != wg_.outsideId() )
        {
            auto& targetInfo = wg_.basinInfo( targetBasin );
            targetInfo.updateAccVolume( amount );
            targetInfo.area += info.area;
            heap_.setLargerValue( targetBasin, amount + targetInfo.amountTillOverflow() );
        }
        heap_.setSmallerValue( basin, infAmount );
        info.area = 0;
        info.overflowVia = bd;
        res.event = Event::BasinFull;
        return res;
    }
    assert( false );
    heap_.setSmallerValue( basin, infAmount );
    return res;
}

} //namespace MR
