#pragma once
#include "MRMeshFwd.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshOrPoints.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

/// represents a point cloud
public interface class MeshOrPoints
{
public:
    property VertCoordsReadOnly^ Points { VertCoordsReadOnly^ get(); }
    property VertBitSetReadOnly^ ValidPoints { VertBitSetReadOnly^ get(); }
    property Box3f^ BoundingBox { Box3f^ get(); }
};

public value struct MeshOrPointsXf
{
    MeshOrPoints^ obj;
    AffineXf3f^ xf;

    MR::MeshOrPointsXf ToNative();
};

MR_DOTNET_NAMESPACE_END
