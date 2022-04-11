#pragma once
#include "MRMeshFwd.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRId.h"
#include "MRBitSet.h"


namespace MR
{
/**
 * @brief Class for aggregate several color map in one
 * @detail Color maps are aggregated according order
 */
template<typename Tag>
class ColorMapAggregator
{
public:
    using ColorMap = Vector<Color, Id<Tag>>;
    using ElementBitSet = TaggedBitSet<Tag>;

    ColorMapAggregator() = default;

    /// set default (background) color
    MRMESH_API void setDefaultColor( const Color& color );

    /**
     * @brief add color map after all (more priority)
     * @param colorMap color map for elements
     * @param elementBitSet bitset of elements for which the color map is applied
     */
    MRMESH_API void pushBack( const ColorMap& colorMap, const ElementBitSet& elementBitSet );

    /**
     * @brief insert color map before element #i (0 - minimum priority)
     * @param colorMap color map for elements
     * @param elementBitSet bitset of elements for which the color map is applied
     */
    MRMESH_API void insert( int i, const ColorMap& colorMap, const ElementBitSet& elementBitSet );

    /**
     * @brief replace color map in #i position
     * @param colorMap color map for elements
     * @param elementBitSet bitset of elements for which the color map is applied
     */
    MRMESH_API void replace( int i, const ColorMap& colorMap, const ElementBitSet& elementBitSet );

    /// reset all accumulated color map
    MRMESH_API void reset();

    /// get number of accumulated color maps
    size_t getColorMapNumber() { return dataSet_.size(); };

    /// get color map by index
    const ColorMap& getColorMap( int i ) { return dataSet_[i].colorMap; };

    /**
     * @brief erase n color map from #i 
     */
    MRMESH_API void erase( int i, int n = 1 );

    /// color map aggregating mode
    enum class AggregateMode
    {
        Overlay, /// result element color is element color of more priority color map (or default color, if there isn't color map for this element)
        Blending /// result element color is blending colors of all color map in this element and default color (https://en.wikipedia.org/wiki/Alpha_compositing)
    };

    /// set color map aggregating mode
    MRMESH_API void setMode( AggregateMode mode );

    /// get aggregated color map for active elements
    MRMESH_API ColorMap aggregate( const ElementBitSet& elementBitSet );
private:
    Color defaultColor_;

    struct Data
    {
        ColorMap colorMap;
        ElementBitSet elements;
    };
    std::vector<Data> dataSet_;

    ColorMap aggregatedColorMap_;
    bool needUpdate_{ true };
    AggregateMode mode_{ AggregateMode::Overlay };

    void checkInputData_( const ColorMap& colorMap, const ElementBitSet& elementBitSet );
    void updateAggregated_( int newSize );
};

}
