#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class IntersectionContour
{
public:
    static ContinousContours^ OrderIntersectionContours( Mesh^ meshA, Mesh^ meshB, PreciseCollisionResult^ intersections );

};

MR_DOTNET_NAMESPACE_END