#pragma once

#include "MRProgressCallback.h"
#include "MRAffineXf3.h"

namespace MR
{

/// determines how to save points/lines/mesh
struct SaveSettings
{
    /// true - save valid points/vertices only (pack them);
    /// false - save all points/vertices preserving their indices
    bool saveValidOnly = true;

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

/// returns the normal as is or after application of given matrix to it in double precision
inline Vector3f applyFloat( const Matrix3d * m, const Vector3f & n )
{
    return m ? Vector3f( *m * Vector3d( n ) ) : n;
}

/// converts given point in double precision and applies given transformation to it
inline Vector3d applyDouble( const AffineXf3d * xf, const Vector3f & p )
{
    Vector3d pd( p );
    return xf ? (*xf)( pd ) : pd;
}

/// converts given normal in double precision and applies given matrix to it
inline Vector3d applyDouble( const Matrix3d * m, const Vector3f & n )
{
    Vector3d nd( n );
    return m ? *m * nd : nd;
}

/// if (xf) is null then just returns (verts);
/// otherwise copies transformed points in (buf) and returns it
MRMESH_API const VertCoords & transformPoints( const VertCoords & verts, const VertBitSet & validVerts, const AffineXf3d * xf, VertCoords & buf );

/// if (m) is null then just returns (normals);
/// otherwise copies transformed normals in (buf) and returns it
MRMESH_API const VertNormals & transformNormals( const VertNormals & normals, const VertBitSet & validVerts, const Matrix3d * m, VertNormals & buf );

} //namespace MR
