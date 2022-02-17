#pragma once
#include "MRVisualObject.h"
#include "MRPolyline.h"

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

// This object type has not visual representation, just holder for lines in scene
class MRMESH_CLASS ObjectLines : public VisualObject
{
public:
    MRMESH_API ObjectLines();
    ObjectLines( ObjectLines&& ) = default;
    ObjectLines& operator=( ObjectLines&& ) = default;

    constexpr static const char* TypeName() noexcept { return "ObjectLines"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setPolyline( const std::shared_ptr<Polyline>& polyline );
    void setXf( const AffineXf3f& xf ) override { VisualObject::setXf( xf ); worldBox_.reset(); }

    virtual const std::shared_ptr<Polyline>& varPolyline() { return polyline_; }
    const std::shared_ptr<const Polyline>& polyline() const 
    { return reinterpret_cast< const std::shared_ptr<const Polyline>& >( polyline_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;
    
    MRMESH_API virtual void setLineWidth( float width );
    float getLineWidth() const { return lineWidth_; }
    MRMESH_API virtual void setPointSize( float size );
    float getPointSize() const { return pointSize_; }

    // swaps this object with other
    MRMESH_API virtual void swap( Object& other ) override;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectLines( ProtectedStruct, const ObjectLines& obj ) : ObjectLines( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    const Vector<Color, UndirectedEdgeId>& getLinesColorMap() const { return linesColorMap_; }
    virtual void setLinesColorMap( Vector<Color, UndirectedEdgeId> linesColorMap ) { linesColorMap_ = std::move( linesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    // get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    // returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;

    // returns cached bounding box of this point object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API const Box3f getWorldBox() const;

protected:
    ObjectLines( const ObjectLines& other ) = default;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;
    MRMESH_API virtual Box3f computeBoundingBoxXf_() const override;

    MRMESH_API virtual void setupRenderObject_() const override;

    Vector<Color, UndirectedEdgeId> linesColorMap_;

    ViewportMask showPoints_;
    ViewportMask smoothConnections_;
private:
    // this is private function to set default colors of this type (ObjectLines) in constructor only
    void setDefaultColors_();
    // width on lines on screen in pixels
    float lineWidth_{ 1.0f };
    float pointSize_{ 5.f };
    std::shared_ptr<Polyline> polyline_;

    mutable std::optional<float> totalLength_;
    mutable std::optional<Box3f> worldBox_;
};
}