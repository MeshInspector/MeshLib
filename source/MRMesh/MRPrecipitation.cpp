#include "MRWatershedGraph.h"
#include "MRHeap.h"
#include "MRTimer.h"

namespace MR
{

class PrecipitationSimulator
{
public:
    PrecipitationSimulator( WatershedGraph & wg );

    enum class Event
    {
        Finish,     ///< all basins are full and water goes outside
        BasinFull,  ///< one basin just became full
        Merge       ///< two basins just merged
    };

    struct SimulationStep
    {
        Event event = Event::Finish;
        float time = FLT_MAX;
        Graph::VertId basin;     ///< BasinFull: this basin just became full
                                 ///< Merge: this basin just absorbed the other basin
        Graph::VertId neiBasin;  ///< BasinFull: the flow from full basin will first go here (may be not the last destination)
                                 ///< Merge: this basin was just absorbed
    };

    /// processes the next event in precipitation
    SimulationStep simulateOne();

private:
    WatershedGraph & wg_;
    static constexpr float infTime = FLT_MAX;
    Heap<float, Graph::VertId, std::greater<float>> heap_;
};

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
        auto targetBasin = wg_.flowsTo( neiBasin );
        if ( targetBasin == basin )
        {
            res.event = Event::Merge;
            //auto& neiInfo = wg_.basinInfo( basin );
            //...
            return res;
        }
        if ( targetBasin != wg_.outsideId() )
        {
            auto& targetInfo = wg_.basinInfo( targetBasin );
            targetInfo.update( time );
            targetInfo.area += info.area;
        }
        info.area = 0;
        info.overflowTo = neiBasin;
        res.event = Event::BasinFull;
        res.basin = basin;
        res.neiBasin = neiBasin;
        return res;
    }
    assert( false );
    return res;
}


} //namespace MR
