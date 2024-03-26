#include "MRAddNoise.h"

namespace MR
{

void addNoise( VertCoords& points, const VertBitSet& validVerts, float sigma, unsigned int startValue )
{
    if ( validVerts.size() > 100000 )
    {
        const size_t numBlock = 256;
        const size_t step = validVerts.size() / numBlock;
        std::array<size_t, numBlock + 1> chunk;
        for ( size_t i = 0; i < numBlock; i++ )
        {
            chunk[i] = step * i;
        }
        chunk.back() = validVerts.size() - 1;

        tbb::parallel_for( tbb::blocked_range<size_t>( 0, numBlock ),
        [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t pos = range.begin(); pos < range.end(); pos++ )
            {
                std::mt19937 gen_{ value + ( unsigned int )pos };
                std::normal_distribution d{ 0.0f, sigma };
                for ( auto i = chunk[pos]; i < chunk[pos + 1]; i++ )
                {
                    const auto& v = validVerts.find_next( Id<VertTag>( i ) );
                    points[v] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
                }
            }
        } );
    }
    else
    {
        std::mt19937 gen_{ value };
        std::normal_distribution d{ 0.0f, sigma };
        for ( const auto& v : validVerts )
        {
            points[v] += Vector3f( d( gen_ ), d( gen_ ), d( gen_ ) );
        }
    }
}

}
