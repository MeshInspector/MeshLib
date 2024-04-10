#pragma once

#include "MRPointOnFace.h"
#include "MRMeshTriPoint.h"
#include "MREdgePoint.h"
#include <variant>

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

// For long-term storage of picked points on an object, such as point and contour widgets,
// it is more convenient to use the local coordinates of the object rather than 3D coordinates,
// which can change depending on the xf of the object.
// --- MeshTriPoint for ObjectMeshHolder 
// --- EdgePoint for ObjectPointsHolder 
// --- VertId for ObjectLinesHolder (polylines)
// --- int value (eq. -1) means not valid pick (pick in empty space). 
using PickedPoint = std::variant<MeshTriPoint, EdgePoint, VertId, int>;

// Converts pickedPoint coordinates depending on the object type into a 3D Vector3 
MRMESH_API MR::Vector3f pickedPointToVector3( const VisualObject* object, const PickedPoint& point );

// Converts PointOnObject coordinates depending on the object type to the PickedPoint variant
MRMESH_API PickedPoint pointOnObjectToPickedPoint( const VisualObject* object, const PointOnObject& pos );

// Checks that the picked point presents in the object's topology
MRMESH_API bool isPickedPointValid( const VisualObject* object, const PickedPoint& point );

} //namespace MR
