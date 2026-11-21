#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRAffineXf3.h"
#include "MRMeshOrPoints.h"
#include "MRVector.h"
#include <cstdint>

namespace MR
{

struct GlobalRegistrationParams
{
    /// the distance between the samples used to find corresponding points in object pairs;
    /// 0 means 0.5% of the largest bounding box diagonal of the objects
    float samplingVoxelSize = 0;

    /// two objects are aligned by their pair registration only if this fraction of the samples
    /// of the smaller object lies on the other object after it
    float minOverlap = 0.1f;

    /// the number of RANSAC runs with different seeds for each pair of objects, the one with the largest overlap is kept
    int pairAttempts = 3;

    /// the seed of the first RANSAC run, see RigidXfRansacParams::seed
    std::uint32_t seed = 0;
};

struct GlobalRegistrationResult
{
    /// new transformation of each object to world space
    Vector<AffineXf3f, ObjId> xfs;

    /// for each object, the object of its group that kept its transformation;
    /// objects not overlapping any other object form groups of one object
    Vector<ObjId, ObjId> groupRoots;
};

/// \brief Finds the transformations registering given objects (meshes or point clouds with normals) as one or several groups
/// without any initial alignment: each pair of objects is aligned by matching FPFH descriptors of their samples (see computeFPFH),
/// finding the rigid transformation by findRigidXfRansac with overlap scoring and refining it by ICP;
/// then the objects are linked by a spanning tree of the pairs with the largest overlap (at least params.minOverlap)
/// \ingroup AligningTransformGroup
[[nodiscard]] MRMESH_API Expected<GlobalRegistrationResult> globalRegistration( const Vector<MeshOrPointsXf, ObjId> & objects,
    const GlobalRegistrationParams & params = {}, const ProgressCallback & progress = {} );

} //namespace MR
