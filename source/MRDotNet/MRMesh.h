#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN


public ref class Mesh
{
internal:
    Mesh( MR::Mesh* mesh );

public:
    ~Mesh();

    property VertCoordsReadOnly^ Points { VertCoordsReadOnly^ get(); }

    property VertBitSetReadOnly^ ValidVerts { VertBitSetReadOnly^ get(); }

    property FaceBitSetReadOnly^ ValidFaces { FaceBitSetReadOnly^ get(); }

    property TriangulationReadOnly^ Triangulation { TriangulationReadOnly^ get(); }

    property EdgePathReadOnly^ HoleRepresentiveEdges { EdgePathReadOnly^ get(); }

    static Mesh^ FromTriangles( VertCoords^ points, MR::DotNet::Triangulation^ triangles );
    static Mesh^ FromTrianglesDuplicatingNonManifoldVertices( VertCoords^ points, MR::DotNet::Triangulation^ triangles );

    static Mesh^ FromFile( System::String^ path );
    static void ToFile( Mesh^ mesh, System::String^ path );

    static bool operator==( Mesh^ a, Mesh^ b );
    static bool operator!=( Mesh^ a, Mesh^ b );

    static Mesh^ MakeCube( Vector3f^ size, Vector3f^ base );
    static Mesh^ MakeSphere( float radius, int vertexCount );

internal:
    MR::Mesh* mesh_;

    VertCoords^ points_;

    VertBitSet^ validVerts_;
    FaceBitSet^ validFaces_;
    MR::DotNet::Triangulation^ triangulation_;
    EdgePath^ holeRepresentiveEdges_;

    MR::Mesh* getMesh() { return mesh_; }
};

MR_DOTNET_NAMESPACE_END