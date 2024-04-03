#pragma once
#ifndef MRMESH_NO_LABEL

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPositionedText.h"
#include "MRViewerEventsListener.h"
#include <boost/signals2/signal.hpp>
#include <memory>

namespace MR
{
class Object;
class ObjectLabel;
struct PositionedText;

/// Helper class to manage ancillary labels used by plugins
struct MRVIEWER_CLASS AncillaryLabel
{
    std::shared_ptr<ObjectLabel> obj;

    AncillaryLabel() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryLabel( AncillaryLabel && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryLabel & operator =( AncillaryLabel && b ) { reset(); obj = std::move( b.obj ); return *this; }

    /// Make not-pickable ancillary object, link it to parent object, and set label text
    explicit AncillaryLabel( Object& parent, const PositionedText& text, bool depthTest = false )
        { make( parent, text, depthTest ); }

    /// Make not-pickable ancillary object, link it to parent object, and set label text
    MRVIEWER_API void make( Object& parent, const PositionedText& text, bool depthTest = false );

    /// Make not-pickable ancillary object without parent object, and set label text
    static MRVIEWER_API std::shared_ptr<ObjectLabel> makeDetached(
        const PositionedText& text, bool depthTest = false );

    /// detach owned object from parent, stops owning it
    MRVIEWER_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryLabel() { reset(); }

    /// Set label text
    MRVIEWER_API void setText( const PositionedText& text );

    /// Reset label text
    MRVIEWER_API void resetText();

    /// Set depth test
    MRVIEWER_API void setDepthTest( bool depthTest );

    /// Set text position
    MRVIEWER_API void setPosition( const Vector3f& pos );
};

/// Helper class that draws ImGui label
class MRVIEWER_CLASS AncillaryImGuiLabel : public PreDrawListener
{
public:
    AncillaryImGuiLabel() = default;

    /// Make label in parent space coordinates, follows parent worldXf
    MRVIEWER_API void make( Object& parent, const PositionedText& text );

    /// Make label in world space coordinates, follows parent worldXf
    MRVIEWER_API void make( const PositionedText& text );

    /// clears this instance
    MRVIEWER_API void reset();
private:
    MRVIEWER_API virtual void preDraw_() override;

    PositionedText labelData_;
    boost::signals2::connection parentXfConnection_;
};


} //namespace MR

#endif
