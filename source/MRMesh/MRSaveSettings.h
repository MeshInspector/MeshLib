#pragma once

#include "MRProgressCallback.h"
#include "MRAffineXf3.h"

namespace MR
{

struct SaveSettings
{
    /// optional per-vertex color to save with the geometry
    const VertColors * colors = nullptr;

    /// this transformation can optionally be applied to all vertices (points) of saved object
    const AffineXf3d * xf = nullptr;

    /// to report save progress and cancel saving if user desires
    ProgressCallback progress;
};

/// returns the point as is or after application of given transform to it in double precision
inline Vector3f applyFloat( const AffineXf3d * xf, const Vector3f & p )
{
    return xf ? Vector3f( (*xf)( Vector3d( p ) ) ) : p;
}

/// converts given point in double precision and applies given transformation to it
inline Vector3d applyDouble( const AffineXf3d * xf, const Vector3f & p )
{
    Vector3d pd( p );
    return xf ? (*xf)( pd ) : pd;
}

/// if (xf) is null then just returns (verts);
/// otherwise copies transformed points in (buf) and returns it
[[nodiscard]] MRMESH_API const VertCoords & transformPoints( const VertCoords & verts, const VertBitSet & validVerts, const AffineXf3d * xf, VertCoords & buf );

} //namespace MR
