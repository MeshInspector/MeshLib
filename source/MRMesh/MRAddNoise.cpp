#include "MRAddNoise.h"

#include "MRParallelFor.h"
#include "MRMesh/MRBitSet.h"

#include <random>

namespace MR
{

Expected<void> addNoise( VertCoords& points, const VertBitSet& validVerts, NoiseSettings settings )
{
    if ( validVerts.count() > 1000 )
    {
        const size_t numBlock = 128;
        const size_t step = validVerts.size() / numBlock;
        auto res = ParallelFor( size_t( 0 ), numBlock,
        [&] ( size_t block )
        {
            std::mt19937 gen_{ settings.seed + ( unsigned int )block };
            std::normal_distribution d{ 0.0f, settings.sigma };
            auto end = step * ( block + 1 );
            if ( end > validVerts.size() )
                end = validVerts.size();
            for ( auto i = step * block; i < end; i++ )
            {
                if ( !validVerts.test( VertId( i ) ) )
                    continue;
                points[VertId( i )] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
            }
        }, settings.callback );

        if ( !res )
        {
            return unexpectedOperationCanceled();
        }
    }
    else
    {
        std::mt19937 gen_{ settings.seed };
        std::normal_distribution d{ 0.0f, settings.sigma };
        for ( const auto& v : validVerts )
        {
            points[v] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
        }
    }

    return {};
}

}
