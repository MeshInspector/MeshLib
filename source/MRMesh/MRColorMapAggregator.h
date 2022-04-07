#pragma once
#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRId.h"
#include "MRBitSet.h"


namespace MR
{

template<typename Tag>
class ColorMapAggregator
{
public:
    using ColorMap = Vector<Color, Id<Tag>>;
    using ElementsBitSet = TaggedBitSet<Tag>;

    ColorMapAggregator() = default;

    void setDefaultColor( const Color& color );

    void setColorMap( int i, const ColorMap& colorMap, const ElementsBitSet& elementsBitSet );

    void resetColorMap( int i );
    
    enum class AggregateMode
    {
        Overlay,
        Mixing
    };

    void setMode( AggregateMode mode );

    ColorMap aggregate();
private:
    Color defaultColor_;
    static const int MaxColorMap{ 10 };

    struct Data
    {
        bool active = false;
        ColorMap colorMap;
        ElementsBitSet elements;
    };
    std::vector<Data> dataSet_{ MaxColorMap };

    ColorMap aggregatedColorMap_;
    int colorMapSize_{ 0 };
    bool needUpdate_{ false };
    AggregateMode mode_{ AggregateMode::Overlay };

    void updateAggregated();
};

template<typename TypeId>
void ColorMapAggregator<TypeId>::setDefaultColor( const Color& color )
{
    defaultColor_ = color;
}

template<typename TypeId>
void ColorMapAggregator<TypeId>::setColorMap( int i, const ColorMap& colorMap, const ElementsBitSet& elementsBitSet )
{
    assert( i >= 0 && i < MaxColorMap );
    assert( !colorMap.empty() );
    assert( colorMap.size() == elementsBitSet.size() );
    if ( colorMapSize_ == 0 )
        colorMapSize_ = int( colorMap.size() );
    else
        assert( colorMap.size() == colorMapSize_ );

    dataSet_[i] = {true, colorMap, elementsBitSet};
}

template<typename TypeId>
void ColorMapAggregator<TypeId>::resetColorMap( int i )
{
    assert( i >= 0 && i < MaxColorMap );
    dataSet_[i].active = false;
    needUpdate_ = true;

    if ( !std::any_of( dataSet_.begin(), dataSet_.end(), [] ( auto e )
    {
        return e.active;
    } ) )
        colorMapSize_ = 0;
}

template<typename TypeId>
void ColorMapAggregator<TypeId>::setMode( AggregateMode mode )
{
    if ( mode == mode_ )
        return;
    mode_ = mode;
    needUpdate_ = true;
}

template<typename TypeId>
ColorMapAggregator<TypeId>::ColorMap ColorMapAggregator<TypeId>::aggregate()
{
    if ( needUpdate_ )
        updateAggregated();
    return aggregatedColorMap_;
}

template<typename TypeId>
void ColorMapAggregator<TypeId>::updateAggregated()
{
    aggregatedColorMap_.resize( colorMapSize_, defaultColor_ );

    ElementsBitSet remaining;
    remaining.resize( colorMapSize_, true );

    if ( mode_ == AggregateMode::Overlay )
    {
        for ( int i = 0; i < MaxColorMap; ++i )
        {
            if ( !dataSet_[i].active )
                continue;

            const auto& colors = dataSet_[i].colorMap;
            ElementsBitSet availableElements = remaining & dataSet_[i].elements;
            for ( const auto& e : availableElements )
            {
                aggregatedColorMap_[e] = colors[e];
            }
            availableElements -= dataSet_[i].elements;
        }
    }
    else
    {
        for ( const auto& e : remaining )
        {
            Vector4f res;
            float count = 0;
            for ( int i = 0; i < MaxColorMap; ++i )
            {
                if ( !dataSet_[i].active || !dataSet_[i].elements[e] )
                    continue;
                res += Vector4f( dataSet_[i].colorMap[e] );
                count += 1.f;
            }
            aggregatedColorMap_[e] = Color( res / count );
        }
    }

    needUpdate_ = false;
}


using VertColorMapAggregator = ColorMapAggregator<VertTag>;
using FaceColorMapAggregator = ColorMapAggregator<FaceTag>;

}
