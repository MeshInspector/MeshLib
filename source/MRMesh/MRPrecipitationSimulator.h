#pragma once

#include "MRId.h"
#include "MRHeap.h"
#include <cfloat>

namespace MR
{

/// the class models water increase in the terrain under the rain with constant precipitation
class PrecipitationSimulator
{
public:
    /// initializes modeling from the initial subdivision of the terrain
    MRMESH_API PrecipitationSimulator( WatershedGraph& wg );

    enum class Event
    {
        Finish,     ///< all basins are full and water goes outside
        BasinFull,  ///< one basin just became full
        Merge       ///< two basins just merged
    };

    struct SimulationStep
    {
        Event event = Event::Finish;
        float amount = FLT_MAX;///< amount of precipitation (in same units as mesh coordinates and water level)
        GraphVertId basin;     ///< BasinFull: this basin just became full
                               ///< Merge: this basin just absorbed the other basin
        GraphVertId neiBasin;  ///< BasinFull: the flow from full basin will first go here (may be not the last destination)
                               ///< Merge: this basin was just absorbed
    };

    /// processes the next event happened with the terrain basins
    MRMESH_API SimulationStep simulateOne();

private:
    WatershedGraph& wg_;
    Heap<float, GraphVertId, std::greater<float>> heap_;
};

} //namespace MR
