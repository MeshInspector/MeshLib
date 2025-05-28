#pragma once

#include "MRObjectMesh.h"

namespace MR
{
class MRMESH_CLASS GeomDataKeeper
{
public:
    virtual ~GeomDataKeeper() = default;
    MRMESH_API virtual void beforeGeometryChange() = 0;
    MRMESH_API virtual void onGeometryChanged() = 0;
};
} // namespace MR