#pragma once

#ifndef MRMESH_NO_LABEL
#include "MRVisualObject.h"
#include "MRSymbolMesh.h"

namespace MR
{

struct LabelVisualizePropertyType : VisualizeMaskType
{
    enum Type : unsigned
    {
        SourcePoint = VisualizeMaskType::VisualizePropsCount,
        LeaderLine,
        Background,

        LabelVisualizePropsCount
    };
};

/// This object type renders label in scene
/// \details default pivot point = (0, 0)
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectLabel : public VisualObject
{
public:
    MRMESH_API ObjectLabel();

    ObjectLabel( ObjectLabel&& ) noexcept = default;
    ObjectLabel& operator = ( ObjectLabel&& ) noexcept = default;
    virtual ~ObjectLabel() = default;

    constexpr static const char* TypeName() noexcept
    {
        return "ObjectLabel";
    }
    virtual const char* typeName() const override
    {
        return TypeName();
    }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    virtual bool hasVisualRepresentation() const override { return true; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// sets size of label font in pixels
    virtual void setFontHeight( float size )
    {
        fontHeight_ = size;
    }
    /// returns size of label font on screen in pixels
    float getFontHeight() const
    {
        return fontHeight_;
    }

    /// sets text and position of label
    MRMESH_API void setLabel( const PositionedText& label );
    const PositionedText& getLabel() const { return label_; }

    /// sets path to font file
    MRMESH_API void setFontPath( const std::filesystem::path& pathToFont );
    const std::filesystem::path& getFontPath() const { return pathToFont_; }

    /// set pivot point
    /// \param pivotPoint - text location parameter of  relative to text position point
    /// [0, 0] - text position point is left-down corner of text
    /// [1, 1] - text position point is right-up corner
    /// can be outside range [0, 0] - [1, 1]
    MRMESH_API void setPivotPoint( const Vector2f& pivotPoint );

    /// get pivot point
    const Vector2f& getPivotPoint() const { return pivotPoint_; }

    /// get pivot shift (pivot point * text diagonal)
    const Vector2f& getPivotShift() const { return pivotShift_; }

    /// sets width of leader line in pixels
    MRMESH_API virtual void setLeaderLineWidth( float width );
    /// returns width of leader line in pixels
    float getLeaderLineWidth() const { return leaderLineWidth_; }
    /// sets size of source point in pixels
    MRMESH_API virtual void setSourcePointSize( float size );
    /// returns size of source point in pixels
    float getSourcePointSize() const { return sourcePointSize_; }
    /// sets background padding in pixels
    MRMESH_API virtual void setBackgroundPadding( float padding );
    /// returns background padding in pixels
    float getBackgroundPadding() const { return backgroundPadding_; }

    /// sets color of source point
    MRMESH_API virtual void setSourcePointColor( const Color& color );
    /// returns color of source point
    const Color& getSourcePointColor() const { return sourcePointColor_; }
    /// sets color of leader line
    MRMESH_API virtual void setLeaderLineColor( const Color& color );
    /// return color of leader line
    const Color& getLeaderLineColor() const { return leaderLineColor_; }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectLabel( ProtectedStruct, const ObjectLabel& obj ) : ObjectLabel( obj )
    {}

    /// returns cached bounding box of this label object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;

    /// returns mesh that represents current label
    const std::shared_ptr<Mesh>& labelRepresentingMesh() const { return mesh_; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;
protected:
    PositionedText label_;
    std::filesystem::path pathToFont_;
    Vector2f pivotPoint_;
    Vector2f pivotShift_;
    std::shared_ptr<Mesh> mesh_;

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
    ViewportMask leaderLine_;

    Color sourcePointColor_;
    Color leaderLineColor_;

    ObjectLabel( const ObjectLabel& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

#ifndef MRMESH_NO_OPENCTM
    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;
#endif

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API virtual void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API virtual void setupRenderObject_() const override;
private:

    /// this is private function to set default colors of this type (ObjectLabel) in constructor only
    void setDefaultColors_();

    void buildMesh_();

    void updatePivotShift_();
};

}
#endif
