#pragma once

#include "MRMesh/MRMeshFwd.h"
#include <memory>

namespace MR
{

/// Helper class to manage ancillary visual points used by plugins
struct AncillaryPoints
{
    std::shared_ptr<ObjectPoints> obj;

    AncillaryPoints() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryPoints( AncillaryPoints && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryPoints & operator =( AncillaryPoints && b ) { reset(); obj = std::move( b.obj ); return *this; }

    /// Make not-pickable object, link it to parent object
    explicit AncillaryPoints( Object& parent ) { make( parent ); }

    /// Make not-pickable object, link it to parent object
    MRMESH_API void make( Object& parent );

    /// detach owned object from parent, stops owning it
    MRMESH_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryPoints() { reset(); }

    /// add ancillary point
    MRMESH_API void addPoint( const Vector3f& point );

    /// add ancillary point with color
    MRMESH_API void addPoint( const Vector3f& point, const Color& color );

    /// add vector of ancillary points
    MRMESH_API void addPoints( const std::vector<Vector3f>& points );

    /// add vector of ancillary points with colors
    MRMESH_API void addPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors );

    /// Set depth test
    MRMESH_API void setDepthTest( bool depthTest );
};

} //namespace MR
