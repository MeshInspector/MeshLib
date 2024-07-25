#pragma once
#include "MRMeshFwd.h"

#pragma managed( push, off )
#include <climits>
#include <cfloat>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

/// Defines the order of edge collapses inside Decimate algorithm
public enum class DecimateStrategy
{
    /// the next edge to collapse will be the one that introduced minimal error to the surface
    MinimizeError,
    /// the next edge to collapse will be the shortest one
    ShortestEdgeFirst
};

public ref struct DecimateParameters
{
    DecimateStrategy strategy = DecimateStrategy::MinimizeError;
    /// for DecimateStrategy::MinimizeError:
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError = 0.001f;
    /// Maximal possible edge length created during decimation
    float maxEdgeLen = FLT_MAX;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift = FLT_MAX;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20.0f;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio = FLT_MAX;
    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    float tinyEdgeLength = -1;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;
    /// Limit on the number of deleted faces
    int maxDeletedFaces = INT_MAX;
    /// Region on mesh to be decimated, it is updated during the operation. If null then whole mesh is decimated
    FaceBitSet^ region = nullptr;
    /// Whether to allow collapse of edges incident to notFlippable edges,
   /// which can move vertices of notFlippable edges unless they are fixed
    bool collapseNearNotFlippable = false;
    // TODO: edgesToCollapse
    // TODO: twinMap
    /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
    bool touchNearBdEdges = true;
    /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
    /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
    /// this setting is ignored if touchNearBdEdges=false
    bool touchBdVerts = true;
    // TODO: bdVerts
    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    float maxAngleChange = -1;
    /// whether to pack mesh at the end
    bool packMesh = false;
    /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
    /// unlike \ref mrDecimateParallelMesh it does not create copies of mesh regions, so may take less memory to operate;
    /// IMPORTANT: please call Mesh::PackOptimally before calling decimating with subdivideParts > 1, otherwise performance will be bad
    int subdivideParts = 1;
    /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
    /// to eliminate small edges near the border of individual parts
    bool decimateBetweenParts = true;
    // TODO: partFaces
    /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
    int minFacesInPart = 0;
};

public value struct DecimateResult
{
    /// Number deleted verts. Same as the number of performed collapses
    int vertsDeleted;
    /// Number deleted faces
    int facesDeleted;
    /// for DecimateStrategy::MinimizeError:
    ///    estimated distance deviation of decimated mesh from the original mesh
    /// for DecimateStrategy::ShortestEdgeFirst:
    ///    the shortest remaining edge in the mesh
    float errorIntroduced;
};

public ref class MeshDecimate
{
public:
    static DecimateResult Decimate( Mesh^ mesh, DecimateParameters^ parameters );
};

MR_DOTNET_NAMESPACE_END