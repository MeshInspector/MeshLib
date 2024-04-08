#pragma once
#include "MRVisualObject.h"
#include "MRXfBasedCache.h"

namespace MR
{

enum class MRMESH_CLASS LinesVisualizePropertyType
{
    Points,
    Smooth,
    _count [[maybe_unused]],
};
template <> struct IsVisualizeMaskEnum<LinesVisualizePropertyType> : std::true_type {};

/// an object that stores a lines
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectLinesHolder : public VisualObject
{
public:
    MRMESH_API ObjectLinesHolder();
    ObjectLinesHolder( ObjectLinesHolder&& ) = default;
    ObjectLinesHolder& operator=( ObjectLinesHolder&& ) = default;

    constexpr static const char* TypeName() noexcept { return "LinesHolder"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual bool hasVisualRepresentation() const override;

    [[nodiscard]] virtual bool hasModel() const override { return bool( polyline_ ); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    const std::shared_ptr<const Polyline3>& polyline() const
    { return reinterpret_cast< const std::shared_ptr<const Polyline3>& >( polyline_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    MRMESH_API virtual void setLineWidth( float width );
    virtual float getLineWidth() const { return lineWidth_; }
    MRMESH_API virtual void setPointSize( float size );
    virtual float getPointSize() const { return pointSize_; }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectLinesHolder( ProtectedStruct, const ObjectLinesHolder& obj ) : ObjectLinesHolder( obj ) {}

    const UndirectedEdgeColors& getLinesColorMap() const { return linesColorMap_; }
    virtual void setLinesColorMap( UndirectedEdgeColors linesColorMap )
    { linesColorMap_ = std::move( linesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }
    virtual void updateLinesColorMap( UndirectedEdgeColors& updated )
    { std::swap( linesColorMap_, updated ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    [[nodiscard]] MRMESH_API bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const override;
    /// get all visualize properties masks
    MRMESH_API AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override;

    /// returns cached bounding box of this point object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// returns cached information about the number of components in the polyline
    MRMESH_API size_t numComponents() const;

protected:
    ObjectLinesHolder( const ObjectLinesHolder& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    /// we serialize polyline as text so separate polyline serialization and base fields serialization
    /// serializeBaseFields_ serializes Parent fields and base fields of ObjectLinesHolder
    MRMESH_API void serializeBaseFields_( Json::Value& root ) const;
    /// serializeFields_: serializeBaseFields_ plus polyline serialization
    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    /// we serialize polyline as text so separate polyline serialization and base fields serialization
    /// deserializeBaseFields_ deserialize Parent fields and base fields of ObjectLinesHolder
    MRMESH_API void deserializeBaseFields_( const Json::Value& root );
    /// deserializeFields_: deserializeBaseFields_ plus polyline deserialization
    MRMESH_API virtual void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

    MRMESH_API virtual void setupRenderObject_() const override;

    /// set all visualize properties masks
    MRMESH_API void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override;

    mutable std::optional<size_t> numComponents_;
    mutable std::optional<float> totalLength_;
    mutable ViewportProperty<XfBasedCache<Box3f>> worldBox_;

    UndirectedEdgeColors linesColorMap_;

    ViewportMask showPoints_;
    ViewportMask smoothConnections_;

    /// width on lines on screen in pixels
    float lineWidth_{ 1.0f };
    float pointSize_{ 5.f };
    std::shared_ptr<Polyline3> polyline_;

private:
    /// this is private function to set default colors of this type (ObjectLinesHolder) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();
};


} // namespace MR
