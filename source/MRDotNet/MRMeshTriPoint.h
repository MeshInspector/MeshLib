#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref struct TriPoint
{
    float a;
    float b;
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

