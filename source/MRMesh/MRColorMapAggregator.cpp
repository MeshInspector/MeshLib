#include "MRColorMapAggregator.h"
#include "MRBitSetParallelFor.h"

namespace MR
{

template<typename Tag>
void ColorMapAggregator<Tag>::setDefaultColor( const Color& color )
{
    defaultColor_ = color;
    needUpdate_ = true;
}

template<typename Tag>
void ColorMapAggregator<Tag>::pushBack( const PartialColorMap& partitialColorMap )
{
    assert( checkInputData_( partitialColorMap ) );
    if ( partitialColorMap.elements.none() )
        dataSet_.push_back( {} );
    else
    {
        dataSet_.push_back( partitialColorMap );
        needUpdate_ = true;
    }
}

template<typename Tag>
void ColorMapAggregator<Tag>::insert( int i, const PartialColorMap& partitialColorMap )
{
    assert( i <= dataSet_.size() );
    assert( checkInputData_( partitialColorMap ) );
    if ( partitialColorMap.elements.none() )
        dataSet_.insert( dataSet_.begin() + i, {} );
    else
    {
        dataSet_.insert( dataSet_.begin() + i, partitialColorMap );
        needUpdate_ = true;
    }
}

template<typename Tag>
void ColorMapAggregator<Tag>::replace( int i, const PartialColorMap& partitialColorMap )
{
    assert( i >= 0 && i < dataSet_.size() );
    assert( checkInputData_( partitialColorMap ) );
    if ( partitialColorMap.elements.none() && dataSet_[i].elements.none() )
        return;
    if ( partitialColorMap.elements.none() )
        dataSet_[i] = {};
    else
        dataSet_[i] = partitialColorMap;
    needUpdate_ = true;
}

template<typename Tag>
void ColorMapAggregator<Tag>::reset()
{
    dataSet_.clear();
    needUpdate_ = true;
}

template<typename Tag>
void ColorMapAggregator<Tag>::erase( int i, int n /*= 1*/ )
{
    assert( i >= 0 && i + n <= dataSet_.size() );
    bool allEmpty = true;
    for ( int it = i; it < i + n; ++it )
        if ( dataSet_[it].elements.any() )
        {
            allEmpty = false;
            break;
        }
    dataSet_.erase( dataSet_.begin() + i, dataSet_.begin() + i + n );
    needUpdate_ = !allEmpty;
}

template<typename Tag>
void ColorMapAggregator<Tag>::setMode( AggregateMode mode )
{
    if ( mode == mode_ )
        return;
    mode_ = mode;
    needUpdate_ = true;
}

template<typename Tag>
typename ColorMapAggregator<Tag>::ColorMap ColorMapAggregator<Tag>::aggregate( const ElementBitSet& elementBitSet )
{
    if ( elementBitSet.none() )
        return {};
    int last = elementBitSet.find_last();
    if ( needUpdate_ )
        updateAggregated_( last + 1 );
    else if ( last >= aggregatedColorMap_.size() )
        aggregatedColorMap_.resize( last + 1, defaultColor_ );

    ColorMap res( elementBitSet.size() );
    for ( const auto& e : elementBitSet )
        res[e] = aggregatedColorMap_[e];

    return res;
}

template<typename Tag>
bool ColorMapAggregator<Tag>::checkInputData_( const PartialColorMap& partitialColorMap )
{
    return partitialColorMap.elements.none() ||
        ( partitialColorMap.colorMap.size() > partitialColorMap.elements.find_last() );
}

template<typename Tag>
void ColorMapAggregator<Tag>::updateAggregated_( int newSize )
{
    aggregatedColorMap_.clear();
    int maxSize = newSize;
    for ( int i = 0; i < dataSet_.size(); ++i )
    {
        if ( dataSet_[i].elements.none() )
            continue;
        maxSize = std::max( maxSize, int( dataSet_[i].elements.find_last() ) + 1 );
    }
    aggregatedColorMap_.resize( maxSize, defaultColor_ );

    if ( mode_ == AggregateMode::Overlay )
    {
        ElementBitSet remaining( maxSize, true );

        for ( int i = int( dataSet_.size() ) - 1; i >= 0; --i )
        {
            if ( dataSet_[i].elements.none() )
                continue;
            const auto& colors = dataSet_[i].colorMap;
            ElementBitSet availableElements = remaining & dataSet_[i].elements;
            for ( const auto& e : availableElements )
            {
                aggregatedColorMap_[e] = colors[e];
            }
            remaining -= dataSet_[i].elements;
        }
    }
    else
    {
        for ( int i = 0; i < int( dataSet_.size() ); ++i )
        {
            if ( dataSet_[i].elements.none() )
                continue;
            const auto& colorMap = dataSet_[i].colorMap;
            BitSetParallelFor( dataSet_[i].elements, [&]( typename ElementBitSet::IndexType e )
            {
                aggregatedColorMap_[e] = blend( colorMap[e], aggregatedColorMap_[e] );
            } );
        }
    }

    needUpdate_ = false;
}

template class ColorMapAggregator<VertTag>;
template class ColorMapAggregator<UndirectedEdgeTag>;
template class ColorMapAggregator<FaceTag>;

}
