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
    /// If true then count touching faces as self-intersections
    bool touchIsIntersection = true; 
    /// Fix method
    enum class Method
    {
        /// Relax mesh around self-intersections
        Relax,
        /// Cut and re-fill regions around self-intersections (may fall back to `Relax`)
        CutAndFill,
        /// Cut the mesh along the exact contours of its self-intersections (see selfBoolean) and keep only the largest by area part
        /// of every connected component; colliding components get united, and the holes that the contours reach are closed for the cut.
        /// The cut is made once and can leave degenerate faces and tiny self-intersections: call fixMeshDegeneracies after it,
        /// and repeat both if needed. Of the other settings only `callback` and `new2Old` are used
        SelfBoolean
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
    /// optional output of `Method::SelfBoolean`: for every face of the fixed mesh, the face of the original mesh that it is (a part of);
    /// not set by the other methods
    FaceMap* new2Old = nullptr;
};

/// Find all self-intersections faces component-wise
MRMESH_API Expected<FaceBitSet> getFaces( const Mesh& mesh, bool touchIsIntersection = true, ProgressCallback cb = {} );

/// Finds and fixes self-intersections per component:
MRMESH_API Expected<void> fix( Mesh& mesh, const Settings& settings );
}

}
