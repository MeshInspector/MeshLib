#pragma once
#ifndef MRMESH_NO_LABEL

#include "MRMesh/MRMeshFwd.h"
#include <memory>

namespace MR
{
class Object;
class ObjectLabel;
struct PositionedText;

/// Helper class to manage ancillary labels used by plugins
struct AncillaryLabel
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
    MRMESH_API void make( Object& parent, const PositionedText& text, bool depthTest = false );

    /// Make not-pickable ancillary object without parent object, and set label text
    static MRMESH_API std::shared_ptr<ObjectLabel> makeDetached(
        const PositionedText& text, bool depthTest = false );

    /// detach owned object from parent, stops owning it
    MRMESH_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryLabel() { reset(); }

    /// Set label text
    MRMESH_API void setText( const PositionedText& text );

    /// Reset label text
    MRMESH_API void resetText();

    /// Set depth test
    MRMESH_API void setDepthTest( bool depthTest );

    /// Set text position
    MRMESH_API void setPosition( const Vector3f& pos );
};

} //namespace MR

#endif
