#pragma once
#include "MRMeshOrPoints.h"
MR_DOTNET_NAMESPACE_BEGIN

/// represents a point cloud
public ref class PointCloud : public MeshOrPoints
{
internal:
    PointCloud( MR::PointCloud* pc );

public:
    PointCloud();
    ~PointCloud();

    virtual property VertCoordsReadOnly^ Points { VertCoordsReadOnly^ get(); }    
    virtual property VertBitSetReadOnly^ ValidPoints { VertBitSetReadOnly^ get(); }
    virtual property Box3f^ BoundingBox { Box3f^ get(); }

    property VertCoordsReadOnly^ Normals { VertCoordsReadOnly^ get(); }

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
    VertBitSet^ validPoints_;
    Box3f^ boundingBox_;    

internal:
    MR::PointCloud* getPointCloud() { return pc_; }
    void clearManagedResources();
};

MR_DOTNET_NAMESPACE_END