#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN
/// represents primitive type
enum class VariantIndex
{
    Face,
    Edge,
    Vertex
};
/// simple point on mesh, represented by primitive id and coordinate in mesh space
public value struct OneMeshIntersection
{
    VariantIndex variantIndex;
    int index;
    Vector3f^ coordinate;
};
/// one contour on mesh
public value struct OneMeshContour
{
    List<OneMeshIntersection>^ intersections;
    bool closed;
};
/// list of contours on mesh
using OneMeshContours = List<OneMeshContour>;

public ref class ContoursCut
{
public:
    /// converts ordered continuous contours of two meshes to OneMeshContours
    /// converters is required for better precision in case of degenerations
    /// note that contours should not have intersections
    static OneMeshContours^ GetOneMeshIntersectionContours( Mesh^ meshA, Mesh^ meshB, ContinousContours^ contours, bool getMeshAIntersections,
    CoordinateConverters^ converters );
    /// converts ordered continuous contours of two meshes to OneMeshContours
    /// converters is required for better precision in case of degenerations
    /// note that contours should not have intersections
    static OneMeshContours^ GetOneMeshIntersectionContours( Mesh^ meshA, Mesh^ meshB, ContinousContours^ contours, bool getMeshAIntersections,
    CoordinateConverters^ converters, AffineXf3f^ rigidB2A );
};

MR_DOTNET_NAMESPACE_END
