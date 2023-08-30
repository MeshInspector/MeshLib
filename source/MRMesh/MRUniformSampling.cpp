#include "MRUniformSampling.h"
#include "MRPointCloud.h"
#include "MRCloseVertices.h"
#include "MRBitSetParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"

namespace MR
{

std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback & cb )
{
    MR_TIMER
    const auto c = findSmallestCloseVertices( pointCloud, distance, cb );
    if ( !c )
        return {};
    const auto & vmap = *c;
    VertBitSet res( vmap.size() );
    BitSetParallelForAll( pointCloud.validPoints, [&]( VertId v )
    {
        if ( vmap[v] == v )
            res.set( v );
    } );
    return res;
}

}
