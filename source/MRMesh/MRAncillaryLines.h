#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h" //const Contours3f& contours = {}
#include <memory>

namespace MR
{

/// Helper class to manage ancillary visual lines used by plugins
struct AncillaryLines
{
    std::shared_ptr<ObjectLines> obj;

    AncillaryLines() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryLines( AncillaryLines && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryLines & operator =( AncillaryLines && b ) { reset(); obj = std::move( b.obj ); return *this; }

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    explicit AncillaryLines( Object& parent, const Contours3f& contours = {} ) { make( parent, contours ); }

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    MRMESH_API void make( Object& parent, const Contours3f& contours = {} );

    /// detach owned object from parent, stops owning it
    MRMESH_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryLines() { reset(); }

    /// Set line geometry
    MRMESH_API void setContours( const Contours3f& contours );

    /// Reset line geometry
    MRMESH_API void resetContours();

    /// Set depth test
    MRMESH_API void setDepthTest( bool depthTest );
};

} // namespace MR
