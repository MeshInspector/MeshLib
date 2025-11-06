#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR
{
struct MRVIEWER_CLASS AncillaryPlane
{
    AncillaryPlane() = default;
    /// since this uniquely owns an ancillary object, we provide only move operations, not copy
    AncillaryPlane( AncillaryPlane&& b ) noexcept = default;
    MRVIEWER_API AncillaryPlane& operator =( AncillaryPlane&& b );
    MRVIEWER_API ~AncillaryPlane();

    MRVIEWER_API void make();

    MRVIEWER_API void reset();

    std::shared_ptr<PlaneObject> obj;
};
}
