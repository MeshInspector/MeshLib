#pragma once

#include "MRMeshBuilderTypes.h"
#include "MRMesh.h"
#include "MRProgressCallback.h"

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
  * MR::Triangulation t;
  * // add simple plane triangles
  * t.push_back({0_v,1_v,2_v}); // face #0
  * t.push_back({2_v,1_v,3_v}); // face #1
  * // make topology
  * auto topology = MR::MeshBuilder::fromTriangles(t);
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

/// construct mesh topology from a set of triangles with given ids;
/// if skippedTris is given then it receives all input triangles not added in the resulting topology
MRMESH_API MeshTopology fromTriangles( const Triangulation & t, const BuildSettings & settings = {}, ProgressCallback progressCb = {} );

struct VertDuplication
{
    VertId srcVert; // original vertex before duplication
    VertId dupVert; // new vertex after duplication
};

// resolve non-manifold vertices by creating duplicate vertices in the triangulation (which is modified)
// `lastValidVert` is needed if `region` or `t` does not contain full mesh, then first duplicated vertex will have `lastValidVert+1` index
// return number of duplicated vertices
MRMESH_API size_t duplicateNonManifoldVertices( Triangulation & t, FaceBitSet * region = nullptr,
    std::vector<VertDuplication>* dups = nullptr, VertId lastValidVert = {} );

// construct mesh topology from a set of triangles with given ids;
// unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices;
// triangulation is modified to introduce duplicates
MRMESH_API MeshTopology fromTrianglesDuplicatingNonManifoldVertices( 
    Triangulation & t,
    std::vector<VertDuplication> * dups = nullptr,
    const BuildSettings & settings = {} );

// construct mesh from point triples;
// all coinciding points are given the same VertId in the result
MRMESH_API Mesh fromPointTriples( const std::vector<Triangle3f> & posTriples );

// a part of a whole mesh to be constructed
struct MeshPiece
{
    FaceMap fmap; // face of part -> face of whole mesh
    VertMap vmap; // vert of part -> vert of whole mesh
    MeshTopology topology;
    FaceBitSet rem; // remaining triangles of part, not in topology
};

// construct mesh topology in parallel from given disjoint mesh pieces (which do not have any shared vertex)
// and some additional triangles (in settings) that join the pieces
MRMESH_API MeshTopology fromDisjointMeshPieces(
    const Triangulation & t, VertId maxVertId,
    const std::vector<MeshPiece> & pieces,
    const BuildSettings & settings = {} );

// adds triangles in the existing topology, given face indecies must be free;
// settings.region on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
MRMESH_API void addTriangles( MeshTopology & res, const Triangulation & t, const BuildSettings & settings = {} );

// adds triangles in the existing topology, auto selecting face ids for them;
// vertTriples on output contain the remaining triangles that could not be added into the topology right now, but may be added later when other triangles appear in the mesh
MRMESH_API void addTriangles( MeshTopology & res, std::vector<VertId> & vertTriples,
    FaceBitSet * createdFaces = nullptr ); //< this set receives indices of added triangles

/// construct mesh topology from face soup, where each face can have arbitrary degree (not only triangles)
MRMESH_API MeshTopology fromFaceSoup( const std::vector<VertId> & verts, const Vector<VertSpan, FaceId> & faces,
    const BuildSettings & settings = {}, ProgressCallback progressCb = {} );

struct UniteCloseParams
{
    ///< vertices located closer to each other than \param closeDist will be united
    float closeDist = 0.0f;

    ///< if true then only boundary vertices can be united, all internal vertices (even close ones) will remain
    bool uniteOnlyBd = true;

    ///< if true, only vertices from this region can be affected
    VertBitSet* region = nullptr;

    ///< if true - try to duplicates non-manifold vertices instead of removing faces
    bool duplicateNonManifold = false;

    ///< is the mapping of vertices: before -> after
    VertMap* optionalVertOldToNew = nullptr;

    ///< this can be used to map attributes to duplicated vertices
    std::vector<MeshBuilder::VertDuplication>* optionalDuplications = nullptr;
};

/// the function finds groups of mesh vertices located closer to each other than \param closeDist, and unites such vertices in one;
/// then the mesh is rebuilt from the remaining triangles
/// \param optionalVertOldToNew is the mapping of vertices: before -> after
/// \param uniteOnlyBd if true then only boundary vertices can be united, all internal vertices (even close ones) will remain
/// \return the number of vertices united, 0 means no change in the mesh
MRMESH_API int uniteCloseVertices( Mesh & mesh, float closeDist, bool uniteOnlyBd = true,
    VertMap * optionalVertOldToNew = nullptr );

/// the function finds groups of mesh vertices located closer to each other than \param params.closeDist, and unites such vertices in one;
/// then the mesh is rebuilt from the remaining triangles
/// \return the number of vertices united, 0 means no change in the mesh
MRMESH_API int uniteCloseVertices( Mesh& mesh, const UniteCloseParams& params = {} );

} //namespace MeshBuilder

} //namespace MR
