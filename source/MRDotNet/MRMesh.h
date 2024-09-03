#pragma once
#include "MRMeshOrPoints.h"

MR_DOTNET_NAMESPACE_BEGIN

public value struct PointOnFace
{
    FaceId faceId;
    Vector3f^ point;
};

public value struct MeshProjectionResult
{
    PointOnFace pointOnFace;
    MeshTriPoint^ meshTriPoint;
    float distanceSquared;
};

public value struct MeshPart
{
    Mesh^ mesh;
    FaceBitSet^ region;
};

/// represents a mesh, including topology (connectivity) information and point coordinates,
public ref class Mesh : public MeshOrPoints
{
internal:
    Mesh( MR::Mesh* mesh );

public:
    ~Mesh();
    /// point coordinates
    virtual property VertCoordsReadOnly^ Points { VertCoordsReadOnly^ get(); }
    /// set of all valid vertices
    virtual property VertBitSetReadOnly^ ValidPoints { VertBitSetReadOnly^ get(); }

    virtual property Box3f^ BoundingBox { Box3f^ get(); }
    /// set of all valid faces
    property FaceBitSetReadOnly^ ValidFaces { FaceBitSetReadOnly^ get(); }
    /// info about triangles
    property TriangulationReadOnly^ Triangulation { TriangulationReadOnly^ get(); }
    /// edges with no valid left face for every boundary in the mesh
    property EdgePathReadOnly^ HoleRepresentiveEdges { EdgePathReadOnly^ get(); }

    

    /// transforms all points
    void Transform( AffineXf3f^ xf );
    /// transforms all points in the region
    void Transform( AffineXf3f^ xf, VertBitSet^ region );

    /// creates mesh from point coordinates and triangulation
    static Mesh^ FromTriangles( VertCoords^ points, MR::DotNet::Triangulation^ triangles );
    /// creates mesh from point coordinates and triangulation. If some vertices are not manifold, they will be duplicated
    static Mesh^ FromTrianglesDuplicatingNonManifoldVertices( VertCoords^ points, MR::DotNet::Triangulation^ triangles );
    
    /// loads mesh from file of any supported format
    static Mesh^ FromAnySupportedFormat( System::String^ path );
    /// saves mesh to file of any supported format
    static void ToAnySupportedFormat( Mesh^ mesh, System::String^ path );

    static bool operator==( Mesh^ a, Mesh^ b );
    static bool operator!=( Mesh^ a, Mesh^ b );

    /// creates a parallelepiped with given sizes and base
    static Mesh^ MakeCube( Vector3f^ size, Vector3f^ base );
    /// creates a sphere of given radius and vertex count
    static Mesh^ MakeSphere( float radius, int vertexCount );
    /// creates a torus with given parameters
    static Mesh^ MakeTorus( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution );

    static Mesh^ MakeCylinder( float radius, float length );
    static Mesh^ MakeCylinder( float radius, float startAngle, float arcSize, float length );
    static Mesh^ MakeCylinder( float radius, float startAngle, float arcSize, float length, int resolution );
    static Mesh^ MakeCylinder( float radius0, float radius1, float startAngle, float arcSize, float length, int resolution );

    static MeshProjectionResult FindProjection( Vector3f^ point, MeshPart meshPart );
    static MeshProjectionResult FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared );
    static MeshProjectionResult FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared, AffineXf3f^ xf );
    static MeshProjectionResult FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared, AffineXf3f^ xf, float minDistanceSquared );

private:
    MR::Mesh* mesh_;

    VertCoords^ points_;

    VertBitSet^ validPoints_;
    FaceBitSet^ validFaces_;
    MR::DotNet::Triangulation^ triangulation_;
    EdgePath^ holeRepresentiveEdges_;
    Box3f^ boundingBox_;

internal:
    MR::Mesh* getMesh() { return mesh_; }
    void clearManagedResources();
};



MR_DOTNET_NAMESPACE_END