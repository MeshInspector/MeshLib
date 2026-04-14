#pragma once
#include "MRMeshFwd.h"
#include "MRPlane3.h"
#include "MRObjectMeshData.h"

namespace MR
{

struct DivideMeshWithPlaneParams
{
    /// if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
    float eps{ 0.0f };

    /// if set - function tries to fill cut after dividing (this operation might fail leading to "params.errors")
    bool fillCut{ true };

    /// if set and filled - function subdivides filling after cut
    bool subdivideFilling{ false };

    /// optional output other part of dividing (expected to be empty)
    ObjectMeshData* otherPart{ nullptr };

    /// optional output list of errors that could possibly happen during 'divideMeshWithPlane' function call
    std::vector<std::string>* errors{ nullptr };
};

/// divide mesh \param data (with attributes) on two parts by given \param plane
/// optionally fills and subdivides cut area
MRMESH_API void divideMeshWithPlane( ObjectMeshData& data, const Plane3f& plane, const DivideMeshWithPlaneParams& divideParams = {} );

}
