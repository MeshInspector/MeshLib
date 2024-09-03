#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

enum class VariantIndex
{
    Face,
    Edge,
    Vertex
};

public value struct OneMeshIntersection
{
    VariantIndex variantIndex;
    int index;
    Vector3f^ coordinate;
};

public value struct OneMeshContour
{
    List<OneMeshIntersection>^ intersections;
    bool closed;
};

using OneMeshContours = List<OneMeshContour>;

public ref class ContoursCut
{
public:
    static OneMeshContours^ GetOneMeshIntersectionContours( Mesh^ meshA, Mesh^ meshB, ContinousContours^ contours, bool getMeshAIntersections,
    CoordinateConverters^ converters, AffineXf3f^ rigidB2A );
};

MR_DOTNET_NAMESPACE_END
