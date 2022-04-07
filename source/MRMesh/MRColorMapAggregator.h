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
        Blending
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
    bool needUpdate_{ true };
    AggregateMode mode_{ AggregateMode::Overlay };

    void updateAggregated();
};

template<typename Tag>
void ColorMapAggregator<Tag>::setDefaultColor( const Color& color )
{
    defaultColor_ = color;
}

template<typename Tag>
void ColorMapAggregator<Tag>::setColorMap( int i, const ColorMap& colorMap, const ElementsBitSet& elementsBitSet )
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

template<typename Tag>
void ColorMapAggregator<Tag>::resetColorMap( int i )
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

template<typename Tag>
void ColorMapAggregator<Tag>::setMode( AggregateMode mode )
{
    if ( mode == mode_ )
        return;
    mode_ = mode;
    needUpdate_ = true;
}

template<typename Tag>
typename ColorMapAggregator<Tag>::ColorMap ColorMapAggregator<Tag>::aggregate()
{
    if ( needUpdate_ )
        updateAggregated();
    return aggregatedColorMap_;
}

template<typename Tag>
void ColorMapAggregator<Tag>::updateAggregated()
{
    aggregatedColorMap_.clear();
    aggregatedColorMap_.resize( colorMapSize_, defaultColor_ );

    ElementsBitSet remaining;
    remaining.resize( colorMapSize_, true );

    if ( mode_ == AggregateMode::Overlay )
    {
        for ( int i = MaxColorMap - 1; i >= 0; --i )
        {
            if ( !dataSet_[i].active )
                continue;

            const auto& colors = dataSet_[i].colorMap;
            ElementsBitSet availableElements = remaining & dataSet_[i].elements;
            for ( const auto& e : availableElements )
            {
                aggregatedColorMap_[e] = colors[e];
            }
            remaining -= dataSet_[i].elements;
        }
    }
    else
    {
        for ( int i = 0; i < MaxColorMap; ++i )
        {
            if ( !dataSet_[i].active )
                continue;
            const auto& colorMap = dataSet_[i].colorMap;
            for ( const auto& e : dataSet_[i].elements )
            {
                const Vector4f frontColor4 = Vector4f( colorMap[e] );
                const Vector3f a = Vector3f( frontColor4.x, frontColor4.y, frontColor4.z ) * frontColor4.w;
                const Vector4f backColor4 = Vector4f( aggregatedColorMap_[e] );
                const Vector3f b = Vector3f( backColor4.x, backColor4.y, backColor4.z ) * backColor4.w * ( 1 - frontColor4.w );
                const float alphaRes = frontColor4.w + backColor4.w * ( 1 - frontColor4.w );
                const Vector3f newColor = ( a + b ) / alphaRes;
                aggregatedColorMap_[e] = Color( newColor.x, newColor.y, newColor.z, alphaRes );
            }
        }
    }
    

    needUpdate_ = false;
}


using VertColorMapAggregator = ColorMapAggregator<VertTag>;
using FaceColorMapAggregator = ColorMapAggregator<FaceTag>;

}
