#pragma once
#include "MRVisualObject.h"
#include "MRPointCloud.h"
#include "MRXfBasedCache.h"

namespace MR
{

struct PointsVisualizePropertyType : VisualizeMaskType
{
    enum : unsigned
    {
        SelectedVertices = VisualizeMaskType::VisualizePropsCount,
        PointsVisualizePropsCount
    };
};

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

    const std::shared_ptr<const PointCloud>& pointCloud() const 
    { return reinterpret_cast< const std::shared_ptr<const PointCloud>& >( points_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;
    
    const VertBitSet& getSelectedPoints() const { return selectedPoints_; }
    MRMESH_API virtual void selectPoints( VertBitSet newSelection );
    /// returns colors of selected vertices
    const Color& getSelectedVerticesColor( ViewportId id = {} ) const
    {
        return selectedVerticesColor_.get( id );
    }
    /// sets colors of selected vertices
    MRMESH_API virtual void setSelectedVerticesColor( const Color& color, ViewportId id = {} );

    MRMESH_API const ViewportProperty<Color>& getSelectedVerticesColorsForAllViewports() const;
    MRMESH_API virtual void setSelectedVerticesColorsForAllViewports( ViewportProperty<Color> val );

    /// get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;
    
    /// sets size of points on screen in pixels
    MRMESH_API virtual void setPointSize( float size );
    /// returns size of points on screen in pixels
    float getPointSize() const { return pointSize_; }

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

    /// sets rendering discretization, each \param val -th point will be displayed on screen
    void setRenderDiscretization( int val )
    {
        if ( renderDiscretization_ == val)
            return;
        renderDiscretization_ = val;
        needRedraw_ = true;
    }

    /// returns rendering discretization, each N-th point will be displayed on screen
    int getRenderDiscretization() const
    {
        return renderDiscretization_;
    }

    /// returns file extension used to serialize the points
    [[nodiscard]] const char * savePointsFormat() const { return savePointsFormat_; }

    /// sets file extension used to serialize the points: must be not null and must start from '.'
    MRMESH_API void setSavePointsFormat( const char * newFormat );

protected:
    VertBitSet selectedPoints_;
    mutable std::optional<size_t> numValidPoints_;
    mutable std::optional<size_t> numSelectedPoints_;
    ViewportProperty<Color> selectedVerticesColor_;
    ViewportMask showSelectedVertices_ = ViewportMask::all();

    std::shared_ptr<PointCloud> points_;
    mutable ViewportProperty<XfBasedCache<Box3f>> worldBox_;

    /// size of point in pixels
    float pointSize_{ 5.0f };

    ObjectPointsHolder( const ObjectPointsHolder& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

    MRMESH_API virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual VoidOrErrStr deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API virtual void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual void setupRenderObject_() const override;

    int renderDiscretization_ = 1;


private:

    /// this is private function to set default colors of this type (ObjectPointsHolder) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

#ifndef MRMESH_NO_OPENCTM
    const char * savePointsFormat_ = ".ctm";
#else
    const char * savePointsFormat_ = ".ply";
#endif
};

}
