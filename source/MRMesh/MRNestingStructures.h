#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"

namespace MR
{

namespace Nesting
{

struct NestingResult
{
    /// best found xf for this object (might be equal with input xf if no good nesting was found)
    AffineXf3f xf;

    /// false - means that this object does not fit the nest
    bool nested{ false };
};

struct MeshXf
{
    /// input mesh - should not be nullptr
    const Mesh* mesh{ nullptr };
    /// input mesh world transformation before nesting
    AffineXf3f xf;
};

struct NestingBaseParams
{
    /// available nest
    Box3f nest;

    /// minimum space among meshes in the nest
    float minInterval{ 0.01f };
};

}

}