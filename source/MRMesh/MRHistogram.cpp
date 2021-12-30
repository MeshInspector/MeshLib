#include "MRHistogram.h"
#include <algorithm>

namespace MR
{

Histogram::Histogram( float min, float max, size_t size ) :
    min_{min},
    max_{max}
{
    if ( size == 0 )
        return;
    bins_.resize( size );
    binSize_ = ( max_ - min_ ) / size;
}

void Histogram::addSample( float sample )
{
    sample = std::clamp( sample, min_, max_ );
    ++bins_[getBinId( sample )];
}

void Histogram::addHistogram( const Histogram& hist )
{
    if ( hist.bins_.size() != bins_.size() )
        return;

    for ( size_t i = 0; i < bins_.size(); ++i )
        bins_[i] += hist.bins_[i];
}

const std::vector<size_t>& Histogram::getBins() const
{
    return bins_;
}

float Histogram::getMin() const
{
    return min_;
}

float Histogram::getMax() const
{
    return max_;
}

size_t Histogram::getBinId( float sample ) const
{
    size_t id = ( binSize_ == 0.0f ? 0 : size_t( ( sample - min_ ) / binSize_ ) );
    return std::clamp( id, size_t( 0 ), bins_.size() - 1 );
}

std::pair<float, float> Histogram::getBinMinMax( size_t binId ) const
{
    float min = min_ + binId * binSize_;
    return {min,min + binSize_};
}

}
