#pragma once

#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include <memory>

namespace MR
{

/// Helper class to manage ancillary visual mesh used by plugins
struct MRVIEWER_CLASS AncillaryMesh
{
    std::shared_ptr<ObjectMesh> obj;

    AncillaryMesh() = default;

    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryMesh( AncillaryMesh && b ) noexcept : obj{ std::move( b.obj ) } {}
    AncillaryMesh & operator =( AncillaryMesh && b ) { reset(); obj = std::move( b.obj ); return *this; }

    /// Make not-pickable object, link it to parent object
    explicit AncillaryMesh( Object& parent ) { make( parent ); }

    /// Make not-pickable object, link it to parent object
    MRVIEWER_API void make( Object& parent );

    /// detach owned object from parent, stops owning it
    MRVIEWER_API void reset();

    /// detach owned object from parent, stops owning it
    ~AncillaryMesh() { reset(); }
};

} //namespace MR
