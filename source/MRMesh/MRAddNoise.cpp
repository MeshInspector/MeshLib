#include "MRAddNoise.h"

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
            std::mt19937 gen_{ seed + ( unsigned int )range.begin() };
            std::normal_distribution d{ 0.0f, sigma };
            for ( size_t pos = range.begin(); pos < range.end(); pos++ )
            {
                for ( auto i = step * pos; i < step * (pos + 1); i++ )
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
