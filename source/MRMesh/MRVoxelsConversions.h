#pragma once
#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRSimpleVolume.h"

namespace MR
{

struct BaseVolumeConversionParams
{
    AffineXf3f basis; // position of lowest left voxel, and axes vectors (A.transposed().x == x axis of volume)
};

struct MeshToSimpleVolumeParams : BaseVolumeConversionParams
{
    Vector3i dimensions{ 100,100,100 }; // num voxels along each axis
    float maxDistSq{ FLT_MAX }; // maximum value in voxels squared, not used in TopologyOrientation mode
    enum SignDetectionMode
    {
        Unsigned, // unsigned voxels, useful for `Shell` offset
        TopologyOrientation, // projection sign
        WindingRule // ray intersection counter, useful to fix self-intersections
    } signMode{ TopologyOrientation };
};

struct SimpleVolumeToMeshParams : BaseVolumeConversionParams
{
    float iso{ 0.0f };
};

MRMESH_API SimpleVolume meshToSimpleVolume( const Mesh& mesh, const MeshToSimpleVolumeParams& params = {} );

MRMESH_API Mesh simpleVolumeToMesh( const SimpleVolume& volume, const SimpleVolumeToMeshParams& params = {} );

}