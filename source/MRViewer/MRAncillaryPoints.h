#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include <memory>

namespace MR
{

/// Helper class to manage ancillary visual points used by plugins
struct MRVIEWER_CLASS AncillaryPoints
{
    std::shared_ptr<ObjectPoints> obj;

    AncillaryPoints() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryPoints( AncillaryPoints && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryPoints & operator =( AncillaryPoints && b ) { reset(); obj = std::move( b.obj ); return *this; }

    /// Make not-pickable object, link it to parent object
    explicit AncillaryPoints( Object& parent ) { make( parent ); }

    /// Make not-pickable object, link it to parent object
    MRVIEWER_API void make( Object& parent );

    /// detach owned object from parent, stops owning it
    MRVIEWER_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryPoints() { reset(); }

    /// add ancillary point
    MRVIEWER_API void addPoint( const Vector3f& point );

    /// add ancillary point with color
    MRVIEWER_API void addPoint( const Vector3f& point, const Color& color );

    /// add vector of ancillary points
    MRVIEWER_API void addPoints( const std::vector<Vector3f>& points );

    /// add vector of ancillary points with colors
    MRVIEWER_API void addPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors );

    /// Set depth test
    MRVIEWER_API void setDepthTest( bool depthTest );
};

} //namespace MR
