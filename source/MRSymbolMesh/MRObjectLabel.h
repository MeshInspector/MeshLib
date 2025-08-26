#pragma once

#include "MRSymbolMeshFwd.h"
#include "MRSymbolMesh.h"

#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRPositionedText.h"

namespace MR
{

enum class MRSYMBOLMESH_CLASS LabelVisualizePropertyType
{
    SourcePoint,
    LeaderLine,
    Background,
    Contour,
    _count [[maybe_unused]],
};
template <> struct IsVisualizeMaskEnum<LabelVisualizePropertyType> : std::true_type {};

/// This object type renders label in scene
/// \details default pivot point = (0, 0)
/// \ingroup DataModelGroup
class MRSYMBOLMESH_CLASS ObjectLabel : public VisualObject
{
public:
    MRSYMBOLMESH_API ObjectLabel();

    ObjectLabel( ObjectLabel&& ) noexcept = default;
    ObjectLabel& operator = ( ObjectLabel&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept
    {
        return "ObjectLabel";
    }
    virtual const char* typeName() const override
    {
        return TypeName();
    }

    MRSYMBOLMESH_API virtual void applyScale( float scaleFactor ) override;

    virtual bool hasVisualRepresentation() const override { return true; }

    MRSYMBOLMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRSYMBOLMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// sets size of label font in pixels
    MRSYMBOLMESH_API virtual void setFontHeight( float size );
    /// returns size of label font on screen in pixels
    float getFontHeight() const { return fontHeight_; }

    /// sets text and position of label
    MRSYMBOLMESH_API void setLabel( const PositionedText& label );
    const PositionedText& getLabel() const { return label_; }

    /// sets path to font file
    MRSYMBOLMESH_API void setFontPath( const std::filesystem::path& pathToFont );
    const std::filesystem::path& getFontPath() const { return pathToFont_; }

    /// set pivot point
    /// \param pivotPoint - text location parameter of  relative to text position point
    /// [0, 0] - text position point is left-down corner of text
    /// [1, 1] - text position point is right-up corner
    /// can be outside range [0, 0] - [1, 1]
    MRSYMBOLMESH_API void setPivotPoint( const Vector2f& pivotPoint );

    /// get pivot point
    const Vector2f& getPivotPoint() const { return pivotPoint_; }

    /// get pivot shift (pivot point * text diagonal)
    const Vector2f& getPivotShift() const { return pivotShift_; }

    /// sets width of leader line in pixels
    MRSYMBOLMESH_API virtual void setLeaderLineWidth( float width );
    /// returns width of leader line in pixels
    float getLeaderLineWidth() const { return leaderLineWidth_; }
    /// sets size of source point in pixels
    MRSYMBOLMESH_API virtual void setSourcePointSize( float size );
    /// returns size of source point in pixels
    float getSourcePointSize() const { return sourcePointSize_; }
    /// sets background padding in pixels
    MRSYMBOLMESH_API virtual void setBackgroundPadding( float padding );
    /// returns background padding in pixels
    float getBackgroundPadding() const { return backgroundPadding_; }

    /// sets color of source point
    MRSYMBOLMESH_API virtual void setSourcePointColor( const Color& color, ViewportId id = {} );
    /// returns color of source point
    const Color& getSourcePointColor( ViewportId id = {} ) const
    {
        return sourcePointColor_.get( id );
    }
    /// sets color of leader line
    MRSYMBOLMESH_API virtual void setLeaderLineColor( const Color& color, ViewportId id = {} );
    /// return color of leader line
    const Color& getLeaderLineColor( ViewportId id = {} ) const
    {
        return leaderLineColor_.get( id );
    }
    /// sets contour color
    MRSYMBOLMESH_API void setContourColor( const Color& color, ViewportId id = {} );
    /// return contour color
    const Color& getContourColor( ViewportId id = {} ) const
    {
        return contourColor_.get( id );
    }

    MRSYMBOLMESH_API const ViewportProperty<Color>& getSourcePointColorsForAllViewports() const;
    MRSYMBOLMESH_API virtual void setSourcePointColorsForAllViewports( ViewportProperty<Color> val );

    MRSYMBOLMESH_API const ViewportProperty<Color>& getLeaderLineColorsForAllViewports() const;
    MRSYMBOLMESH_API virtual void setLeaderLineColorsForAllViewports( ViewportProperty<Color> val );

    MRSYMBOLMESH_API const ViewportProperty<Color>& getContourColorsForAllViewports() const;
    MRSYMBOLMESH_API virtual void setContourColorsForAllViewports( ViewportProperty<Color> val );

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectLabel( ProtectedStruct, const ObjectLabel& obj ) : ObjectLabel( obj )
    {}

    /// returns cached bounding box of this label object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    MRSYMBOLMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;

    /// returns mesh that represents current label
    /// only used in Render object for binding, cleared after it
    const std::shared_ptr<Mesh>& labelRepresentingMesh() const { return mesh_; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRSYMBOLMESH_API virtual size_t heapBytes() const override;

    /// get all visualize properties masks
    MRSYMBOLMESH_API AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRSYMBOLMESH_API const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override;

    /// Loads font, and converts the symbols of text into mesh;
    /// since this operation is time consuming, one can call this method in parallel for several ObjectLabels before rendering
    MRSYMBOLMESH_API void buildMeshFromText() const;

protected:
    PositionedText label_;
    std::filesystem::path pathToFont_;
    Vector2f pivotPoint_;

    /// size of label font on screen in pixels
    float fontHeight_{ 25.0f };
    /// width of leader line on screen in pixels
    float leaderLineWidth_{ 1.0f };
    /// radius of source point on screen in pixels
    float sourcePointSize_{ 5.f };
    /// padding of background on screen in pixels
    float backgroundPadding_{ 8.f };

    ViewportMask sourcePoint_;
    ViewportMask background_;
    ViewportMask contour_;
    ViewportMask leaderLine_;

    ViewportProperty<Color> sourcePointColor_;
    ViewportProperty<Color> leaderLineColor_;
    ViewportProperty<Color> contourColor_;

    ObjectLabel( const ObjectLabel& other ) = default;

    /// swaps this object with other
    MRSYMBOLMESH_API virtual void swapBase_( Object& other ) override;

    MRSYMBOLMESH_API virtual Box3f computeBoundingBox_() const override;

    MRSYMBOLMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRSYMBOLMESH_API virtual void deserializeFields_( const Json::Value& root ) override;

    MRSYMBOLMESH_API virtual void setupRenderObject_() const override;

    /// set all visualize properties masks
    MRSYMBOLMESH_API void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override;

private:
    /// this is private function to set default colors of this type (ObjectLabel) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

    void updatePivotShift_() const;

    mutable bool needRebuild_{ true };
    mutable Vector2f pivotShift_;
    mutable std::shared_ptr<Mesh> mesh_;
    mutable Box3f meshBox_; // needed for pivot update
};

}
