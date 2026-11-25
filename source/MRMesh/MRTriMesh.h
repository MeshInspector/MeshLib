#pragma once

#include "MRVector.h"
#include "MRId.h"
#include "MRVector3.h"

namespace MR
{

/// very simple structure for storing mesh of triangles only,
/// without easy navigation between neighbor elements as in Mesh
struct [[nodiscard]] TriMesh
{
    Triangulation tris;
    VertCoords points;

    /// constructs TriMesh from valid triangles of given mesh; invalid faces get invalid vertex ids in the result
    [[nodiscard]] MRMESH_API static TriMesh fromMesh( const Mesh & mesh );

    /// constructs TriMesh from valid triangles of given mesh; points of the mesh will be moved in the result
    [[nodiscard]] MRMESH_API static TriMesh fromMesh( Mesh && mesh );
};

} //namespace MR
