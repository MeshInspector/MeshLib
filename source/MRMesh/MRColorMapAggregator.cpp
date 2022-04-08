#include "MRColorMapAggregator.h"
#include "MRGTest.h"

namespace MR
{

template<typename Tag>
void ColorMapAggregator<Tag>::setDefaultColor( const Color& color )
{
    defaultColor_ = color;
}

template<typename Tag>
void ColorMapAggregator<Tag>::pushBack( const ColorMap& colorMap, const ElementBitSet& elementsBitSet )
{
    checkInputData_( colorMap, elementsBitSet );
    dataSet_.push_back( { colorMap, elementsBitSet } );
}

template<typename Tag>
void ColorMapAggregator<Tag>::insert( int i, const ColorMap& colorMap, const ElementBitSet& elementsBitSet )
{
    assert( i < dataSet_.size() );
    checkInputData_( colorMap, elementsBitSet );
    dataSet_.insert( dataSet_.begin() + i, { colorMap, elementsBitSet } );
}

template<typename Tag>
void ColorMapAggregator<Tag>::replace( int i, const ColorMap& colorMap, const ElementBitSet& elementsBitSet )
{
    assert( i >= 0 && i < dataSet_.size() );
    checkInputData_( colorMap, elementsBitSet );
    dataSet_[i] = { colorMap, elementsBitSet };
}

template<typename Tag>
void ColorMapAggregator<Tag>::erase( int i, int n /*= 1*/ )
{
    assert( i >= 0 && i + n <= dataSet_.size() );
    dataSet_.erase( dataSet_.begin() + i, dataSet_.begin() + n );

    needUpdate_ = true;
    if ( dataSet_.empty() )
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
void ColorMapAggregator<Tag>::checkInputData_( const ColorMap& colorMap, const ElementBitSet& elementsBitSet )
{
    assert( !colorMap.empty() );
    assert( colorMap.size() == elementsBitSet.size() );

    if ( colorMapSize_ == 0 )
        colorMapSize_ = int( colorMap.size() );
    else
        assert( colorMap.size() == colorMapSize_ );
}



template<typename Tag>
void ColorMapAggregator<Tag>::updateAggregated()
{
    aggregatedColorMap_.clear();
    aggregatedColorMap_.resize( colorMapSize_, defaultColor_ );

    ElementBitSet remaining;
    remaining.resize( colorMapSize_, true );

    if ( mode_ == AggregateMode::Overlay )
    {
        for ( int i = 0; i < int( dataSet_.size() ); ++i )
        {
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
        for ( int i = int( dataSet_.size() ) - 1; i >= 0; --i )
        {
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

template class ColorMapAggregator<FaceTag>;
template class ColorMapAggregator<VertTag>;


TEST( MRColorMapAggregator, FaceTag )
{
    Color cWhite = Color::white();
    Color cRed = Color( Vector4i( 255, 0, 0, 128 ) );
    Color cGreen = Color( Vector4i( 0, 255, 0, 128 ) );

    FaceColorMapAggregator cma;
    cma.setDefaultColor( cWhite );

    int size = 5;
    FaceBitSet faces;
    faces.resize( 5, true );
    cma.pushBack( Vector<Color, FaceId>( size, cRed ), FaceBitSet( std::string( "00110" ) ) );
    cma.pushBack( Vector<Color, FaceId>( size, cGreen ), FaceBitSet( std::string( "01100" ) ) );
    cma.setMode( FaceColorMapAggregator::AggregateMode::Overlay );
    Vector<Color, FaceId> res = cma.aggregate();

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == cRed );
    ASSERT_TRUE( res[2_f] == cGreen );
    ASSERT_TRUE( res[3_f] == cGreen );
    ASSERT_TRUE( res[4_f] == cWhite );


    cma.setMode( FaceColorMapAggregator::AggregateMode::Blending );
    res = cma.aggregate();

    ASSERT_TRUE( res.size() == size );
    ASSERT_TRUE( res[0_f] == cWhite );
    ASSERT_TRUE( res[1_f] == Color( Vector4i( 255, 126, 126, 255 ) ) );
    ASSERT_TRUE( res[2_f] == Color( Vector4i( 126, 190, 62, 255 ) ) );
    ASSERT_TRUE( res[3_f] == Color( Vector4i( 126, 255, 126, 255 ) ) );
    ASSERT_TRUE( res[4_f] == cWhite );
}

}
