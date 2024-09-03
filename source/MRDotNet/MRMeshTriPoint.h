#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public value struct TriPoint
{
    float a;
    float b;

    Vector3f^ Interpolate( Vector3f^ p0, Vector3f^ p1, Vector3f^ p2 );
};

public ref struct MeshTriPoint
{
public:
    ~MeshTriPoint();

    MR::DotNet::EdgeId e;
    MR::DotNet::TriPoint bary;

internal:
    MeshTriPoint( MR::MeshTriPoint* mtp );
private:
    MR::MeshTriPoint* mtp_;
};

MR_DOTNET_NAMESPACE_END

