#pragma once

namespace MR
{

// If the Object is GeomDataKeeper please call these methods before and after changing the geometry

class MRMESH_CLASS GeomDataKeeper
{
public:
    virtual ~GeomDataKeeper() = default;

    /// Call it before changing geometry
    MRMESH_API virtual void beforeGeometryChange() = 0;

    /// Call it right after geometry changed. Notice that it creates additional history actions and store it in the history store.
    /// So you must use SCOPED_HISORY to agregate all history actions into one record
    MRMESH_API virtual void onGeometryChanged() = 0;
};
} // namespace MR