#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public enum class BooleanOperation
{
    InsideA,
    InsideB,
    OutsideA,
    OutsideB,
    Union,
    Intersection,
    DifferenceBA,
    DifferenceAB,
    Count
};

public value struct BooleanParameters
{
    AffineXf3f^ rigidB2A;
    bool mergeAllNonIntersectingComponents;
};

public value struct BooleanResult
{
    Mesh^ mesh;
};

public ref class MeshBoolean
{
public:
    static BooleanResult Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op );
    static BooleanResult Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op, BooleanParameters params );
};

MR_DOTNET_NAMESPACE_END