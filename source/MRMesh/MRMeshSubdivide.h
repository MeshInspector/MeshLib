#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRConstants.h"
#include <cfloat>
#include <functional>

namespace MR
{

/// \defgroup MeshSubdivideGroup Mesh Subdivide
/// \ingroup MeshAlgorithmGroup
/// \{

struct SubdivideSettings
{
    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;
    /// Maximum number of edge splits allowed
    int maxEdgeSplits = 1000;
    /// Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation
    float maxDeviationAfterFlip = 1;
    /// Improves local mesh triangulation by doing edge flips if it does change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip = FLT_MAX;
    /// If this value is less than FLT_MAX then edge flips will
    /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
    float criticalAspectRatioFlip = 1000.0f;
    /// Region on mesh to be subdivided, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;
    /// New vertices appeared during subdivision will be added here
    VertBitSet * newVerts = nullptr;
    /// If false do not touch border edges (cannot subdivide lone faces)\n
    /// use \ref MR::findRegionOuterFaces to find boundary faces
    bool subdivideBorder = true;
    /// If subdivideBorder is off subdivider can produce narrow triangles near border\n
    /// this parameter prevents subdivision of such triangles
    float critAspectRatio = 20.0f;
    /// Puts new vertices so that they form a smooth surface together with existing vertices.
    /// This option works best for natural surfaces without sharp edges in between triangles
    bool smoothMode = false;
    /// In case of activated smoothMode, the smoothness is locally deactivated at the edges having
    /// dihedral angle at least this value
    float minSharpDihedralAngle = PI_F / 6; // 30 degrees
    /// this function is called each time a new vertex has been created, but before the ring is made Delone
    std::function<void(VertId)> onVertCreated;
    /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
    std::function<void(EdgeId e1, EdgeId e)> onEdgeSplit;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/// Split edges in mesh region according to the settings;\n
/// \return The total number of edge splits performed
MRMESH_API int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings = {} );

/// \}

}
