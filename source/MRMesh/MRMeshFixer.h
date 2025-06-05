#pragma once

#include "MRId.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRConstants.h"
#include <cfloat>
#include <string>

namespace MR
{

/// \defgroup MeshFixerGroup Mesh Fixer
/// \ingroup MeshAlgorithmGroup
/// \{

/// Duplicates all vertices having more than two boundary edges (and returns the number of duplications);
MRMESH_API int duplicateMultiHoleVertices( Mesh & mesh );

/// finds multiple edges in the mesh
using MultipleEdge = VertPair;
[[nodiscard]] MRMESH_API Expected<std::vector<MultipleEdge>> findMultipleEdges( const MeshTopology & topology, ProgressCallback cb = {} );
[[nodiscard]] inline bool hasMultipleEdges( const MeshTopology & topology ) { return !findMultipleEdges( topology ).value().empty(); }

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESH_API void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges );
/// finds and resolves multiple edges
MRMESH_API void fixMultipleEdges( Mesh & mesh );

/// finds faces having aspect ratio >= criticalAspectRatio
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio = FLT_MAX, ProgressCallback cb = {} );

/// finds edges having length <= criticalLength
[[nodiscard]] MRMESH_API Expected<UndirectedEdgeBitSet> findShortEdges( const MeshPart& mp, float criticalLength, ProgressCallback cb = {} );

struct FixMeshDegeneraciesParams
{
    /// maximum permitted deviation from the original surface
    float maxDeviation{ 0.0f };

    /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
    float tinyEdgeLength{ 0.0f };

    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio{ 1e4f };

    /// Permit edge flips if it does not change dihedral angle more than on this value
    float maxAngleChange{ PI_F / 3 };

    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 1e-6f;

    /// degenerations will be fixed only in given region, it is updated during the operation
    FaceBitSet* region = nullptr;

    enum class Mode
    {
        Decimate, ///< use decimation only to fix degeneracies
        Remesh,   ///< if decimation does not succeed, perform subdivision too
        RemeshPatch ///< if both decimation and subdivision does not succeed, removes degenerate areas and fills occurred holes
    } mode{ Mode::Remesh };

    ProgressCallback cb;
};

/// Fixes degenerate faces and short edges in mesh (changes topology)
MRMESH_API Expected<void> fixMeshDegeneracies( Mesh& mesh, const FixMeshDegeneraciesParams& params );

/// finds vertices in region with complete ring of N edges
[[nodiscard]] MRMESH_API VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region = nullptr );

/// returns true if the edge e has both left and right triangular faces and the degree of dest( e ) is 2
[[nodiscard]] MRMESH_API bool isEdgeBetweenDoubleTris( const MeshTopology& topology, EdgeId e );

/// if the edge e has both left and right triangular faces and the degree of dest( e ) is 2,
/// then eliminates left( e ), right( e ), e, e.sym(), next( e ), dest( e ), and returns prev( e );
/// if region is provided then eliminated faces are excluded from it;
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDoubleTris( MeshTopology& topology, EdgeId e, FaceBitSet * region = nullptr );

/// eliminates all double triangles around given vertex preserving vertex valid;
/// if region is provided then eliminated triangles are excluded from it
MRMESH_API void eliminateDoubleTrisAround( MeshTopology & topology, VertId v, FaceBitSet * region = nullptr );

/// returns true if the destination of given edge has degree 3 and 3 incident triangles
[[nodiscard]] MRMESH_API bool isDegree3Dest( const MeshTopology& topology, EdgeId e );

/// if the destination of given edge has degree 3 and 3 incident triangles,
/// then eliminates the destination vertex with all its edges and all but one faces, and returns valid remaining edge with same origin as e;
/// if region is provided then eliminated triangles are excluded from it;
/// otherwise returns invalid edge
MRMESH_API EdgeId eliminateDegree3Dest( MeshTopology& topology, EdgeId e, FaceBitSet * region = nullptr );

/// eliminates from the mesh all vertices having degree 3 and 3 incident triangles from given region (which is updated);
/// if \param fs is provided then eliminated triangles are excluded from it;
/// \return the number of vertices eliminated
MRMESH_API int eliminateDegree3Vertices( MeshTopology& topology, VertBitSet & region, FaceBitSet * fs = nullptr );

/// if given vertex is present on the boundary of some hole several times then returns an edge of this hole (without left);
/// returns invalid edge otherwise (not a boundary vertex, or it is present only once on the boundary of each hole it pertains to)
[[nodiscard]] MRMESH_API EdgeId isVertexRepeatedOnHoleBd( const MeshTopology& topology, VertId v );

/// returns set bits for all vertices present on the boundary of a hole several times;
[[nodiscard]] MRMESH_API VertBitSet findRepeatedVertsOnHoleBd( const MeshTopology& topology );

/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
[[nodiscard]] MRMESH_API FaceBitSet findHoleComplicatingFaces( const Mesh & mesh );

/// Parameters structure for `fixMeshCreases` function
struct FixCreasesParams
{
    /// edges with dihedral angle sharper this will be considered as creases
    float creaseAngle = PI_F * 175.0f / 180.0f;

    /// planar check is skipped for faces with worse aspect ratio
    float criticalTriAspectRatio = 1e3f;

    /// maximum number of algorithm iterations
    int maxIters = 10;
};

/// Finds creases edges and re-triangulates planar areas around them, useful to fix double faces
MRMESH_API void fixMeshCreases( Mesh& mesh, const FixCreasesParams& params = {} );

/// Parameters for `findDisorientedFaces` function
struct FindDisorientationParams
{
    /// Mode of detecting disoriented face
    enum class RayMode
    {
        Positive, ///< positive (normal) direction of face should have even number of intersections
        Shallowest, ///< positive or negative (normal or -normal) direction (the one with lowest number of intersections) should have even/odd number of intersections
        Both ///< both direction should have correct number of intersections (positive - even; negative - odd)
    } mode{ RayMode::Shallowest };

    /// if set - copy mesh, and fills holes for better quality in case of ray going out through hole
    bool virtualFillHoles{ false };

    ProgressCallback cb;
};

/// returns all faces that are oriented inconsistently, based on number of ray intersections
[[nodiscard]] MRMESH_API Expected<FaceBitSet> findDisorientedFaces( const Mesh& mesh, const FindDisorientationParams& params = {} );


/// \}

} // namespace MR
