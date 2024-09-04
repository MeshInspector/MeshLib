#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class IntersectionContour
{
public:
    /// combines individual intersections into ordered contours with the properties:
    /// a. left  of contours on mesh A is inside of mesh B,
    /// b. right of contours on mesh B is inside of mesh A,
    /// c. each intersected edge has origin inside meshes intersection and destination outside of it
    static ContinousContours^ OrderIntersectionContours( Mesh^ meshA, Mesh^ meshB, PreciseCollisionResult^ intersections );

};

MR_DOTNET_NAMESPACE_END