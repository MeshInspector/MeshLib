#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRFillHoleNicely.h"

namespace MR
{

namespace SelfIntersections
{

/// Setting set for mesh self-intersections fix
struct Settings
{
    /// If true then count touching faces as self-intersections
    bool touchIsIntersection = true; 
    /// Fix method
    enum class Method
    {
        /// Relax mesh around self-intersections
        Relax,
        /// Cut and re-fill regions around self-intersections (may fall back to `Relax`)
        CutAndFill
    };
    Method method = Method::Relax;
    /// Maximum relax iterations
    int relaxIterations = 5;
    /// Maximum expand count (edge steps from self-intersecting faces), should be >= 0
    int maxExpand = 3;
    /// Edge length for subdivision of holes covers (0.0f means auto)
    /// FLT_MAX to disable subdivision
    float subdivideEdgeLen = 0.0f;
    /// trying to stay close to initial surface when patching
    bool mimicPatch = false;
    /// Callback function
    ProgressCallback callback = {};
};

/// Find all self-intersections faces component-wise
MRMESH_API Expected<FaceBitSet> getFaces( const Mesh& mesh, bool touchIsIntersection = true, ProgressCallback cb = {} );

/// Finds and fixes self-intersections per component:
MRMESH_API Expected<void> fix( Mesh& mesh, const Settings& settings );

/// splits mesh part (e.g. self-intersecting faces) on groups separated by sharp edges with dihedral angle in [angleThreshold, PI-angleThreshold];
/// nearly planar and nearly folded edges do not separate groups, so overlapping coplanar triangles stay together
/// \param angleThreshold in (0, PI/2)
/// \return the mapping FaceId -> group id (meaningful only for the faces of mesh part) and the number of groups
[[nodiscard]] MRMESH_API std::pair<Face2RegionMap, int> getGroupsMap( const MeshPart& mp, float angleThreshold = 0.5f );

/// splits given faces on groups (see getGroupsMap), then deletes each group and fills the appeared holes separately using \ref patchMesh;
/// the edges near not yet patched groups and near the patches of other groups are never split
/// \return all new faces
MRMESH_API FaceBitSet cutAndFillGroups( Mesh& mesh, const FaceBitSet& faces, float angleThreshold = 0.5f, const FillHoleNicelySettings& settings = {} );
}

}
