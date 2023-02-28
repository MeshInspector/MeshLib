#pragma once

#include "MRHeapBytes.h"

namespace MR
{
/// Simple class for calculating histogram
/// \ingroup MathGroup
class Histogram
{
public:
    Histogram() = default;

    /// Initialize histogram with minimum and maximum values, and number of bins
    MRMESH_API Histogram( float min, float max, size_t size );

    /// Adds sample to corresponding bin
    MRMESH_API void addSample( float sample, size_t count = 1 );
    /// Adds bins of input hist to this
    MRMESH_API void addHistogram( const Histogram& hist );

    /// Gets bins
    MRMESH_API const std::vector<size_t>& getBins() const;

    /// Gets minimum value of histogram
    MRMESH_API float getMin() const;
    /// Gets maximum value of histogram
    MRMESH_API float getMax() const;

    /// Gets id of bin that inherits sample
    MRMESH_API size_t getBinId( float sample ) const;
    /// Gets minimum and maximum of diapason inherited by bin
    MRMESH_API std::pair<float, float> getBinMinMax( size_t binId ) const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return MR::heapBytes( bins_ ); }

private:
    std::vector<size_t> bins_;
    float min_{0.0f};
    float max_{0.0f};
    float binSize_{0.0f};
};
}