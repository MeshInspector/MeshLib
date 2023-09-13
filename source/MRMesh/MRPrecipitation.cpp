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
        Graph::VertId neiBasin;  ///< BasinFull: the flow from full basin will go here
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
    info.lastUpdateTime = time;
    info.remVolume = 0;

    res.event = Event::BasinFull;
    res.basin = basin;
    return res;
}


} //namespace MR
