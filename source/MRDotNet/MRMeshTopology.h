#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class MeshTopology
{
internal:
    MeshTopology( MR::MeshTopology* mtop );

public:
    ~MeshTopology();

    property VertBitSetReadOnly^ ValidVerts { VertBitSetReadOnly^ get(); }
    property FaceBitSetReadOnly^ ValidFaces { FaceBitSetReadOnly^ get(); }
    property TriangulationReadOnly^ Triangulation { TriangulationReadOnly^ get(); }
    property EdgePathReadOnly^ HoleRepresentiveEdges { EdgePathReadOnly^ get(); }

internal:
    VertBitSet^ validVerts_;
    FaceBitSet^ validFaces_;
    MR::DotNet::Triangulation^ triangulation_;
    EdgePath^ holeRepresentiveEdges_;

private:
    MR::MeshTopology* mtop_;

    bool isLoneEdge_( MR::EdgeId edge );
};

MR_DOTNET_NAMESPACE_END

