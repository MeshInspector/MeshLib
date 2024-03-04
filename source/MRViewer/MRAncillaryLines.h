#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h" //const Contours3f& contours = {}
#include <memory>

namespace MR
{

/// Helper class to manage ancillary visual lines used by plugins
struct MRVIEWER_CLASS AncillaryLines
{
    std::shared_ptr<ObjectLines> obj;

    AncillaryLines() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryLines( AncillaryLines && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryLines & operator =( AncillaryLines && b ) { reset(); obj = std::move( b.obj ); return *this; }
    AncillaryLines( AncillaryLines& b ) = default;

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    explicit AncillaryLines( Object& parent, const Contours3f& contours = {} ) { make( parent, contours ); }

    /// Make not-pickable ancillary object, link it to parent object, and set line geometry
    MRVIEWER_API void make( Object& parent, const Contours3f& contours = {} );

    /// detach owned object from parent, stops owning it
    MRVIEWER_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryLines() { reset(); }

    /// Set line geometry
    MRVIEWER_API void setContours( const Contours3f& contours );

    /// Reset line geometry
    MRVIEWER_API void resetContours();

    /// Set depth test
    MRVIEWER_API void setDepthTest( bool depthTest );
};

} // namespace MR
