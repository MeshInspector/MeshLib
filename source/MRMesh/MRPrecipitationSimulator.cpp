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
            // all flow from nei was already in basin:
            assert( info.area >= neiInfo.area );
            neiInfo.area = 0;
            wg_.merge( basin, neiBasin );
            heap_.setSmallerValue( neiBasin, infAmount );
            heap_.setSmallerValue( basin, amount + info.amountTillOverflow() );
            return res;
        }
        auto v = neiBasin;
        for (;;)
        {
            auto& vInfo = wg_.basinInfo( v );
            if ( !vInfo.overflowVia )
                vInfo.updateAccVolume( amount );
            vInfo.area += info.area;
            auto v2 = wg_.flowsTo( v );
            if ( v2 == v )
            {
                if ( targetBasin != wg_.outsideId() )
                    heap_.setLargerValue( v, amount + vInfo.amountTillOverflow() );
                break;
            }
            v = v2;
        }
        assert( v == targetBasin );
        heap_.setSmallerValue( basin, infAmount );
        info.overflowVia = bd;
        res.event = Event::BasinFull;
        return res;
    }
    assert( false );
    heap_.setSmallerValue( basin, infAmount );
    return res;
}

} //namespace MR
