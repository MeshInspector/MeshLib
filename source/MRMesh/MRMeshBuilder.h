#pragma once

#include "MRMeshBuilderTypes.h"
#include "MRMesh.h"

namespace MR
{

/** \namespace MR::MeshBuilder
  *
  * \brief Building topologies by triangles
  *
  * This namespace provides API for building meshes.
  * 
  * Simple example with key steps
  * \code
  * std::vector<MR::MeshBuilder::Triangle> tris;
  * // add siple plane triangles
  * tris.push_back({0_v,1_v,2_v,0_f});
  * tris.push_back({2_v,1_v,3_v,1_f});
  * // make topology
  * auto topology = MR::MeshBuilder::fromTriangles(tris);
  * \endcode
  *
  * \warning Vertices of triangles should have consistent bypass direction
  *
  * \note It is better to store topology directly in mesh
  *
  * \sa \ref MeshTopology
  */
namespace MeshBuilder
{

// construct mesh topology from a set of triangles with given ids;
// if skippedTris is given then it receives all input triangles not added in the resulting topology
MRMESH_API MeshTopology fromTriangles( const std::vector<Triangle> & tris, std::vector<Triangle> * skippedTris = nullptr );

struct VertDuplication
{
    VertId srcVert; // original vertex before duplication
    VertId dupVert; // new vertex after duplication
};

// resolve non-manifold vertices by creating duplicate vertices
// return number of duplicated vertices
MRMESH_API size_t duplicateNonManifoldVertices( std::vector<Triangle>& tris,
    std::vector<VertDuplication>* dups = nullptr );

// construct mesh topology from a set of triangles with given ids;
// unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices
MRMESH_API MeshTopology fromTrianglesDuplicatingNonManifoldVertices( const std::vector<Triangle> & tris,
    std::vector<VertDuplication> * dups = nullptr,
    std::vector<Triangle> * skippedTris = nullptr );

// construct mesh topology from vertex-index triples
MRMESH_API MeshTopology fromVertexTriples( const std::vector<VertId> & vertTriples );

// construct mesh from point triples;
// all coinciding points are given the same VertId in the result
MRMESH_API Mesh fromPointTriples( const std::vector<ThreePoints> & posTriples );

// a part of a whole mesh to be constructed
struct MeshPiece
{
    FaceMap fmap; // face of part -> face of whole mesh
    VertMap vmap; // vert of part -> vert of whole mesh
    MeshTopology topology;
    std::vector<Triangle> tris; // remaining triangles, not in topology
};

// construct mesh topology in parallel from given disjoint mesh pieces (which do not have any shared vertex)
// and some additional triangles that join the pieces
MRMESH_API MeshTopology fromDisjointMeshPieces( const std::vector<MeshPiece> & pieces, VertId maxVertId, FaceId maxFaceId,
    std::vector<Triangle> & borderTris ); //< on output borderTris will contain not added triangles

// adds triangles in the existing topology, given face indecies must be free;
// tris on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
MRMESH_API void addTriangles( MeshTopology & res, std::vector<Triangle> & tris, bool allowNonManifoldEdge = true );

// adds triangles in the existing topology, auto selecting face ids for them;
// vertTriples on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
MRMESH_API void addTriangles( MeshTopology & res, std::vector<VertId> & vertTriples,
    FaceBitSet * createdFaces = nullptr ); //< this set receives indices of added triangles

// each face is surrounded by a closed contour of vertices [fistVertex, lastVertex)
struct FaceRecord
{
    FaceId face;
    int firstVertex = 0;
    int lastVertex = 0;
};

// construct mesh topology from face soup, where each face can have arbitrary degree (not only triangles)
MRMESH_API MeshTopology fromFaceSoup( const std::vector<VertId> & verts, std::vector<FaceRecord> & faces );

/// the function finds groups of mesh vertices located closer to each other than \ref closeDist, and unites such vertices in one;
/// then the mesh is rebuilt from the remaining triangles
/// \param optionalVertOldToNew is the mapping of vertices: before -> after
/// \return the number of vertices united, 0 means no change in the mesh
MRMESH_API int uniteCloseVertices( Mesh & mesh, float closeDist, VertMap * optionalVertOldToNew = nullptr );

} //namespace MeshBuilder

} //namespace MR
