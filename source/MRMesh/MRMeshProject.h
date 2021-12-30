#pragma once

#include "MRPointOnFace.h"
#include "MRMeshTriPoint.h"
#include "MRMeshPart.h"
#include <cfloat>

namespace MR
{

struct MeshProjectionResult
{
    // the closest point on mesh, transformed by xf if it is given
    PointOnFace proj;
    // its barycentric representation
    MeshTriPoint mtp;
    // squared distance from pt to proj
    float distSq = 0;
};

// computes the closest point on mesh (or its region) to given point
MRMESH_API MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq = FLT_MAX, //< upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    const AffineXf3f * xf = nullptr,   //< mesh-to-point transformation, if not specified then identity transformation is assumed
    float loDistLimitSq = 0 );     //< low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one

struct SignedDistanceToMeshResult
{
    // the closest point on mesh
    PointOnFace proj;
    // its barycentric representation
    MeshTriPoint mtp;
    // distance from pt to proj (positive - outside, negative - inside the mesh)
    float dist = 0;
};

// computes the closest point on mesh (or its region) to given point,
// and finds the distance with sign to it (positive - outside, negative - inside the mesh)
MRMESH_API std::optional<SignedDistanceToMeshResult> findSignedDistance( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq = FLT_MAX ); //< upper limit on the distance in question, if the real distance is larger than the function exits returning nullopt

} //namespace MR
