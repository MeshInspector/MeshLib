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

    /// partial color map
    struct PartitialColorMap
    {
        ColorMap colorMap; // color map
        ElementBitSet elements; // bitset of elements for which the color map is applied
    };

    /// add color map after all (more priority)
    MRMESH_API void pushBack( const PartitialColorMap& partitialColorMap );

    /// insert color map before element #i (0 - minimum priority)
    MRMESH_API void insert( int i, const PartitialColorMap& partitialColorMap );

    /// replace color map in #i position
    MRMESH_API void replace( int i, const PartitialColorMap& partitialColorMap );

    /// reset all accumulated color map
    MRMESH_API void reset();

    /// get number of accumulated color maps
    size_t getColorMapNumber() { return dataSet_.size(); };

    /// get color map by index
    const ColorMap& getColorMap( int i ) { return dataSet_[i].colorMap; };

    /// erase n color map from #i 
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

    std::vector<PartitialColorMap> dataSet_;

    ColorMap aggregatedColorMap_;
    bool needUpdate_{ true };
    AggregateMode mode_{ AggregateMode::Overlay };

    /// return false if partitialColorMap have invalid data
    bool checkInputData_( const PartitialColorMap& partitialColorMap );
    void updateAggregated_( int newSize );
};

}
