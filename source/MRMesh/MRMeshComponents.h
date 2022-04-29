#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"

namespace MR
{

namespace MeshComponents
{

/// \defgroup MeshComponentsGroup MeshComponents
/// \ingroup ComponentsGroup
/// \{

/// Face incidence type
enum FaceIncidence
{
    PerEdge, ///< face can have neighbor only via edge
    PerVertex ///< face can have neighbor via vertex
};

/// returns one connected component containing given face, 
/// not effective to call more than once, if several components are needed use getAllComponents
MRMESH_API FaceBitSet getComponent( const MeshPart& meshPart, FaceId id, FaceIncidence incidence = FaceIncidence::PerEdge );
/// returns one connected component containing given vertex, 
/// not effective to call more than once, if several components are needed use getAllComponentsVerts
MRMESH_API VertBitSet getComponentVerts( const Mesh& mesh, VertId id, const VertBitSet* region = nullptr );

/// returns largest by surface area component
MRMESH_API FaceBitSet getLargestComponent( const MeshPart& meshPart, FaceIncidence incidence = FaceIncidence::PerEdge );
/// returns largest by number of elements component
MRMESH_API VertBitSet getLargestComponentVerts( const Mesh& mesh, const VertBitSet* region = nullptr );

/// returns union of connected components, each of which contains at least one seed face
MRMESH_API FaceBitSet getComponents( const MeshPart& meshPart, const FaceBitSet & seeds, FaceIncidence incidence = FaceIncidence::PerEdge );
/// returns union of connected components, each of which contains at least one seed vert
MRMESH_API VertBitSet getComponentsVerts( const Mesh& mesh, const VertBitSet& seeds, const VertBitSet* region = nullptr );

/// returns the number of connected components in mesh part
MRMESH_API size_t getNumComponents( const MeshPart& meshPart, FaceIncidence incidence = FaceIncidence::PerEdge );

/// gets all connected components of mesh part
MRMESH_API std::vector<FaceBitSet> getAllComponents( const MeshPart& meshPart, FaceIncidence incidence = FaceIncidence::PerEdge );
MRMESH_API std::vector<VertBitSet> getAllComponentsVerts( const Mesh& mesh, const VertBitSet* region = nullptr );
/// gets all connected components, separating vertices by given path (either closed or from boundary to boundary)
MRMESH_API std::vector<VertBitSet> getAllComponentsVertsSeparatedByPath( const Mesh& mesh, const SurfacePath& path );
/// subdivides given edges on connected components
MRMESH_API std::vector<EdgeBitSet> getAllComponentsEdges( const Mesh& mesh, const EdgeBitSet & edges );

/// gets union-find structure for given mesh part
MRMESH_API UnionFind<FaceId> getUnionFindStructureFaces( const MeshPart& meshPart, FaceIncidence incidence = FaceIncidence::PerEdge );
MRMESH_API UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const VertBitSet* region = nullptr );
/// gets union-find structure for vertices, considering connections by given edges only
MRMESH_API UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const EdgeBitSet & edges );
/// gets union-find structure for vertices, considering connections by all edges excluding given ones
MRMESH_API UnionFind<VertId> getUnionFindStructureVertsEx( const Mesh& mesh, const UndirectedEdgeBitSet & ignoreEdges );
/**
 * \brief gets union-find structure for vertices, separating vertices by given path (either closed or from boundary to boundary)
 * \param outPathVerts this set receives all vertices passed by the path
 */
MRMESH_API UnionFind<VertId> getUnionFindStructureVertsSeparatedByPath( const Mesh& mesh, const SurfacePath& path, 
    VertBitSet * outPathVerts = nullptr );

// \}

} // namespace MeshComponents

} // namespace MR
