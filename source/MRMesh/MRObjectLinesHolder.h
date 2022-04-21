#pragma once
#include "MRVisualObject.h"
#include "MRMeshFwd.h"
#include "MRPolyline.h"
#include "MRXfBasedCache.h"

namespace MR
{

struct LinesVisualizePropertyType : VisualizeMaskType
{
    enum Type : unsigned
    {
        Points = VisualizeMaskType::VisualizePropsCount,
        Smooth,

        LinesVisualizePropsCount
    };
};

// an object that stores a lines
class MRMESH_CLASS ObjectLinesHolder : public VisualObject
{
public:
    MRMESH_API ObjectLinesHolder();
    ObjectLinesHolder( ObjectLinesHolder&& ) = default;
    ObjectLinesHolder& operator=( ObjectLinesHolder&& ) = default;
    virtual ~ObjectLinesHolder() = default;

    constexpr static const char* TypeName() noexcept { return "LinesHolder"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    const std::shared_ptr<const Polyline3>& polyline() const
    { return reinterpret_cast< const std::shared_ptr<const Polyline3>& >( polyline_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    MRMESH_API virtual void setLineWidth( float width );
    float getLineWidth() const { return lineWidth_; }
    MRMESH_API virtual void setPointSize( float size );
    float getPointSize() const { return pointSize_; }

    // this ctor is public only for std::make_shared used inside clone()
    ObjectLinesHolder( ProtectedStruct, const ObjectLinesHolder& obj ) : ObjectLinesHolder( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    const Vector<Color, UndirectedEdgeId>& getLinesColorMap() const { return linesColorMap_; }
    virtual void setLinesColorMap( Vector<Color, UndirectedEdgeId> linesColorMap )
    { linesColorMap_ = std::move( linesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    // get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    // returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;

    // returns cached bounding box of this point object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox() const override;

    // returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

protected:
    MRMESH_API ObjectLinesHolder( const ObjectLinesHolder& other );

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;
    MRMESH_API virtual Box3f computeBoundingBoxXf_() const override;

    MRMESH_API virtual void setupRenderObject_() const override;

    mutable std::optional<float> totalLength_;
    mutable XfBasedCache<Box3f> worldBox_;

    Vector<Color, UndirectedEdgeId> linesColorMap_;

    ViewportMask showPoints_;
    ViewportMask smoothConnections_;

    // width on lines on screen in pixels
    float lineWidth_{ 1.0f };
    float pointSize_{ 5.f };
    std::shared_ptr<Polyline3> polyline_;

private:
    // this is private function to set default colors of this type (ObjectLinesHolder) in constructor only
    void setDefaultColors_();
};

}
