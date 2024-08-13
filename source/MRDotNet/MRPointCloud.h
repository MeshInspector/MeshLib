#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// represents a point cloud
public ref class PointCloud
{
internal:
    PointCloud( MR::PointCloud* pc );

public:
    PointCloud();
    ~PointCloud();

    property VertCoordsReadOnly^ Points { VertCoordsReadOnly^ get(); }
    property VertCoordsReadOnly^ Normals { VertCoordsReadOnly^ get(); }
    property VertBitSetReadOnly^ ValidPoints { VertBitSetReadOnly^ get(); }
    property Box3f^ BoundingBox { Box3f^ get(); }

    /// loads point cloud from file of any supported format
    static PointCloud^ FromAnySupportedFormat( System::String^ path );
    /// saves point cloud to file of any supported format
    static void ToAnySupportedFormat( PointCloud^ mesh, System::String^ path );

    void AddPoint( Vector3f^ point );

    void AddPoint( Vector3f^ point, Vector3f^ normal );

private:
    MR::PointCloud* pc_;
    VertCoords^ points_;
    VertCoords^ normals_;
    VertBitSet^ validVerts_;
    Box3f^ boundingBox_;    

internal:
    MR::PointCloud* getPointCloud() { return pc_; }
    void clearManagedResources();
};

MR_DOTNET_NAMESPACE_END