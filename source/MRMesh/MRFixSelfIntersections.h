#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

namespace MR
{

namespace SelfIntersections
{

/// Setting set for mesh self-intersections fix
struct Settings
{
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
    /// Maximum expand count (edge steps from self-intersecting faces), should be > 0
    int maxExpand = 3;
    /// Edge length for subdivision of holes covers (0.0f means auto)
    float subdivideEdgeLen = 0.0f;
    /// Callback function
    ProgressCallback callback = {};
};

/// Find all self-intersections faces component-wise
MRMESH_API Expected<FaceBitSet> getFaces( const Mesh& mesh, ProgressCallback cb = {} );

/// Finds and fixes self-intersections per component:
///   Relax method - simply relax area with self-intersection
///   CutAndFill method - remove area with self-intersection and fills it with new triangles
MRMESH_API VoidOrErrStr fixSimple( Mesh& mesh, const Settings& settings );
}

}
