#pragma once

#include "MRPointOnFace.h"

#include <variant>
#include "MRMesh/MRMeshTriPoint.h"   // TODO REMOVE IT 
#include "MRMesh/MREdgePoint.h" // TODO REMOVE IT  

namespace MR
{

// point located on either
// 1. face of ObjectMesh
// 2. line of ObjectLines
// 3. point of ObjectPoints
struct PointOnObject
{
    PointOnObject() {} //default ctor is required by Clang
    /// 3D location on the object in local coordinates
    Vector3f point;
    /// z buffer value
    float zBuffer{ 1.0f };
    /// to which primitive that point pertains
    union
    {
        int primId = -1;
        FaceId face;             //for ObjectMesh
        UndirectedEdgeId uedge;  //for ObjectLines
        VertId vert;             //for ObjectPoints
    };
    [[nodiscard]] operator PointOnFace() const { return { .face = face, .point = point }; }
};

using PickedPoint = std::variant<MeshTriPoint, EdgePoint, VertId, int>;

MRMESH_API MR::Vector3f pickedPointToVector3( const VisualObject* surface, const PickedPoint& point );
MRMESH_API PickedPoint pointOnObjectToPickedPoint( const VisualObject* surface, const PointOnObject& pos );


} //namespace MR
