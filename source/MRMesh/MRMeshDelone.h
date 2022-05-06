#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// \defgroup MeshDeloneGroup Mesh Delone
/// \details https:///en.wikipedia.org/wiki/Boris_Delaunay
/// \ingroup MeshAlgorithmGroup
/// \{

/// computes the diameter of the triangle's ABC circumcircle
template <typename T>
T circumcircleDiameter( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c );

/// given quadrangle ABCD, checks whether its edge AC satisfies Delone's condition
MRMESH_API bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d );
MRMESH_API bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d );

/// consider quadrangle formed by left and right triangles of given edge, and
/// checks whether this edge satisfies Delone's condition in the quadrangle;
/// \return false otherwise if flipping the edge does not introduce too large surface deviation
MRMESH_API bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, float maxDeviationAfterFlip,
    const FaceBitSet * region = nullptr ); /// false can be returned only for inner edge of the region

/// improves mesh triangulation by performing flipping of edges to satisfy Delone local property,
/// consider every edge at most numIters times, and allow surface deviation at most on given value during every individual flip,
/// \return the number of flips done
MRMESH_API int makeDeloneEdgeFlips( Mesh & mesh, int numIters, float maxDeviationAfterFlip,
    const FaceBitSet * region = nullptr );

/// improves mesh triangulation in a ring of vertices with common origin and represented by edge e
MRMESH_API void makeDeloneOriginRing( Mesh & mesh, EdgeId e, float maxDeviationAfterFlip,
    const FaceBitSet * region = nullptr );

/// \}

} // namespace MR

#include "MRMeshDelone.hpp"