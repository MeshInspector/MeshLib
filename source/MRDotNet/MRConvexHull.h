#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class ConvexHull
{
public:
    // computes the mesh of convex hull from given input points
    static Mesh^ MakeConvexHull( VertCoords^ vertCoords, VertBitSet^ validPoints );

    // computes the mesh of convex hull from given mesh
    static Mesh^ MakeConvexHull( Mesh^ mesh );

    // computes the mesh of convex hull from given point cloud
    static Mesh^ MakeConvexHull( PointCloud^ pointCloud );
};

MR_DOTNET_NAMESPACE_END

