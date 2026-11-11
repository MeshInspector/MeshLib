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
        /// Cut the mesh along the exact contours of its self-intersections and reconnect its sheets there (see selfBoolean),
        /// and keep only the largest by area part of every connected component; the components that collide with one another
        /// are cut as well and taken as one, so they get united; the holes that the self-intersections reach are closed
        /// with temporary faces, which are removed after the cut, so the contours ending in a hole are cut too: the largest part is found
        /// with these faces, and it can fall apart when they are removed;
        /// the cut is made once and leaves degenerate faces along it, and tiny self-intersections can stay or appear there,
        /// so the result is not always free of them: call fixMeshDegeneracies after it, and repeat both while fewer of them are left;
        /// the faces that are not cut keep their ids, and all vertices are rounded as in selfBoolean;
        /// if the mesh cannot be cut, the error is returned and the mesh keeps its faces;
        /// all other settings but the callback are ignored (touching faces are not fixed)
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
};

/// Find all self-intersections faces component-wise
MRMESH_API Expected<FaceBitSet> getFaces( const Mesh& mesh, bool touchIsIntersection = true, ProgressCallback cb = {} );

/// Finds and fixes self-intersections per component:
MRMESH_API Expected<void> fix( Mesh& mesh, const Settings& settings );
}

}
