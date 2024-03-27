#include "MRAddNoise.h"

#include "MRPch/MRTBB.h"
#include "MRMesh/MRBitSet.h"

#include <random>

namespace MR
{

void addNoise( VertCoords& points, const VertBitSet& validVerts, float sigma, unsigned int seed )
{
    if ( validVerts.count() > 100000 )
    {
        const size_t numBlock = 128;
        const size_t step = validVerts.size() / numBlock;
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, numBlock ),
        [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t block = range.begin(); block < range.end(); block++ )
            {
                std::mt19937 gen_{ seed + ( unsigned int )block };
                std::normal_distribution d{ 0.0f, sigma };
                auto end = step * ( block + 1 );
                if ( end > validVerts.size() )
                    end = validVerts.size();
                for ( auto i = step * block; i < end; i++ )
                {
                    if ( !validVerts.test( VertId( i ) ) )
                        continue;
                    points[VertId( i )] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
                }
            }
        } );
    }
    else
    {
        std::mt19937 gen_{ seed };
        std::normal_distribution d{ 0.0f, sigma };
        for ( const auto& v : validVerts )
        {
            points[v] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
        }
    }
}

}
