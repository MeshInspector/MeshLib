#pragma once
#include "MRVisualObject.h"
#include "MRXfBasedCache.h"

namespace MR
{

enum class MRMESH_CLASS PointsVisualizePropertyType
{
    SelectedVertices,
    _count [[maybe_unused]],
};
template <> struct IsVisualizeMaskEnum<PointsVisualizePropertyType> : std::true_type {};

/// an object that stores a points
/// \ingroup ModelHolderGroup
class MRMESH_CLASS ObjectPointsHolder : public VisualObject
{
public:
    MRMESH_API ObjectPointsHolder();

    ObjectPointsHolder( ObjectPointsHolder&& ) noexcept = default;
    ObjectPointsHolder& operator = ( ObjectPointsHolder&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PointsHolder"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual bool hasVisualRepresentation() const override;

    [[nodiscard]] virtual bool hasModel() const override { return bool( points_ ); }

    const std::shared_ptr<const PointCloud>& pointCloud() const
    { return reinterpret_cast< const std::shared_ptr<const PointCloud>& >( points_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// gets current selected points
    const VertBitSet& getSelectedPoints() const { return selectedPoints_; }

    /// sets current selected points
    void selectPoints( VertBitSet newSelection ) { updateSelectedPoints( newSelection ); }

    /// swaps current selected points with the argument
    MRMESH_API virtual void updateSelectedPoints( VertBitSet& selection );

    /// returns selected points if any, otherwise returns all valid points
    MRMESH_API const VertBitSet& getSelectedPointsOrAll() const;

    /// returns colors of selected vertices
    const Color& getSelectedVerticesColor( ViewportId id = {} ) const
    {
        return selectedVerticesColor_.get( id );
    }
    /// sets colors of selected vertices
    MRMESH_API virtual void setSelectedVerticesColor( const Color& color, ViewportId id = {} );

    MRMESH_API const ViewportProperty<Color>& getSelectedVerticesColorsForAllViewports() const;
    MRMESH_API virtual void setSelectedVerticesColorsForAllViewports( ViewportProperty<Color> val );

    [[nodiscard]] MRMESH_API bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const override;

    /// get all visualize properties masks
    MRMESH_API AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override;

    /// sets size of points on screen in pixels
    MRMESH_API virtual void setPointSize( float size );
    /// returns size of points on screen in pixels
    virtual float getPointSize() const { return pointSize_; }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectPointsHolder( ProtectedStruct, const ObjectPointsHolder& obj ) : ObjectPointsHolder( obj )
    {}

    /// returns cached bounding box of this point object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;
    /// returns cached information about the number of valid points
    MRMESH_API size_t numValidPoints() const;
    /// returns cached information about the number of selected points
    MRMESH_API size_t numSelectedPoints() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// returns rendering discretization
    /// display each `renderDiscretization`-th point only,
    /// starting from 0 index, total number is \ref numRenderingValidPoints()
    /// \detail defined by maximum rendered points number as:
    /// \ref numValidPoints() / \ref getMaxRenderingPoints() (rounded up)
    /// updated when setting `maxRenderingPoints` or changing the cloud (setting `DIRTY_FACE` flag)
    int getRenderDiscretization() const { return renderDiscretization_; }

    /// returns count of valid points that will be rendered
    MRMESH_API size_t numRenderingValidPoints() const;

    /// default value for maximum rendered points number
    static constexpr int MaxRenderingPointsDefault = 1'000'000;
    /// recommended value for maximum rendered points number to disable discretization
    static constexpr int MaxRenderingPointsUnlimited = std::numeric_limits<int>::max();

    /// returns maximal number of points that will be rendered
    /// if actual count of valid points is greater then the points will be sampled
    MRMESH_API int getMaxRenderingPoints() const { return maxRenderingPoints_; }

    /// sets maximal number of points that will be rendered
    /// \sa \ref getRenderDiscretization, \ref MaxRenderingPointsDefault, \ref MaxRenderingPointsUnlimited
    MRMESH_API void setMaxRenderingPoints( int val );

    /// returns file extension used to serialize the points
    [[nodiscard]] const char * savePointsFormat() const { return savePointsFormat_; }

    /// sets file extension used to serialize the points: must be not null and must start from '.'
    MRMESH_API void setSavePointsFormat( const char * newFormat );

    /// signal about points selection changing, triggered in selectPoints
    using SelectionChangedSignal = Signal<void()>;
    SelectionChangedSignal pointsSelectionChangedSignal;

    /// signal about render discretization changing, triggered in setRenderDiscretization
    Signal<void()> renderDiscretizationChangedSignal;

protected:
    VertBitSet selectedPoints_;
    mutable std::optional<size_t> numValidPoints_;
    mutable std::optional<size_t> numSelectedPoints_;
    ViewportProperty<Color> selectedVerticesColor_;
    ViewportMask showSelectedVertices_ = ViewportMask::all();

    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    std::shared_ptr<PointCloud> points_;
    mutable ViewportProperty<XfBasedCache<Box3f>> worldBox_;

    /// size of point in pixels
    float pointSize_{ 5.0f };

    ObjectPointsHolder( const ObjectPointsHolder& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

    MRMESH_API virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual Expected<void> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API virtual void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual void setupRenderObject_() const override;

    /// set all visualize properties masks
    MRMESH_API void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override;

    int maxRenderingPoints_ = MaxRenderingPointsDefault;

private:

    /// this is private function to set default colors of this type (ObjectPointsHolder) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

    // update renderDiscretization_ as numValidPoints_ / maxRenderingPoints_ (rounded up)
    void updateRenderDiscretization_();

    int renderDiscretization_ = 1; // auxiliary parameter to avoid recalculation in every frame

    // falls back to the PLY format if no CTM format support is available
    // NOTE: CTM format support is available in the MRIOExtras library; make sure to load it if you prefer CTM
    const char * savePointsFormat_ = ".ctm";
};

}
