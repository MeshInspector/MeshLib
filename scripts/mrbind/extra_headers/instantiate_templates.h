#pragma once

#ifdef MR_PARSING_FOR_PB11_BINDINGS

#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRVoxels/MRObjectVoxels.h"

#define INST_CAT(x, y) INST_CAT_(x, y)
#define INST_CAT_(x, y) x##y
#define INST_IF(cond) INST_CAT(INST_IF_, cond)
#define INST_IF_0(...)
#define INST_IF_1(...) __VA_ARGS__

namespace MR
{

#define VEC3(T, isFloatingPoint) \
    template struct Vector3<T>; \
    template Vector3<T>& operator+=( Vector3<T>& a, const Vector3<T>& b ); \
    template Vector3<T>& operator-=( Vector3<T>& a, const Vector3<T>& b ); \
    template Vector3<T>& operator*=( Vector3<T>& a, T b ); \
    template Vector3<T>& operator/=( Vector3<T>& a, T b ); \
    template Vector3<T> operator-( const Vector3<T> & a ); \
    template const Vector3<T> & operator+( const Vector3<T> & a ); \
    template Vector3<T> cross( const Vector3<T> & a, const Vector3<T> & b ); \
    template T dot( const Vector3<T> & a, const Vector3<T> & b ); \
    template T sqr( const Vector3<T> & a ); \
    template T mixed( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c ); \
    template Vector3<T> mult( const Vector3<T>& a, const Vector3<T>& b ); \
    template T angle( const Vector3<T> & a, const Vector3<T> & b ); \
    template bool operator==( const Vector3<T> & a, const Vector3<T> & b ); \
    template bool operator!=( const Vector3<T> & a, const Vector3<T> & b ); \
    template Vector3<T> operator+( const Vector3<T> & a, const Vector3<T> & b ); \
    template Vector3<T> operator-( const Vector3<T> & a, const Vector3<T> & b ); \
    template Vector3<T> operator*( T a, const Vector3<T> & b ); \
    template Vector3<T> operator*( const Vector3<T> & b, T a ); \
    template Vector3<T> operator/( Vector3<T> b, T a ); \
    INST_IF(isFloatingPoint)( \
        template Vector3<T> unitVector3( T azimuth, T altitude ); \
    ) \

#define VEC2(T) \
    template struct Vector2<T>; \
    template T cross( const Vector2<T> & a, const Vector2<T> & b ); \
    template T dot( const Vector2<T> & a, const Vector2<T> & b ); \
    template T sqr( const Vector2<T> & a ); \
    template Vector2<T> mult( const Vector2<T>& a, const Vector2<T>& b ); \
    template T angle( const Vector2<T> & a, const Vector2<T> & b ); \
    template bool operator ==( const Vector2<T> & a, const Vector2<T> & b ); \
    template bool operator !=( const Vector2<T> & a, const Vector2<T> & b ); \
    template Vector2<T> operator +( const Vector2<T> & a, const Vector2<T> & b ); \
    template Vector2<T> operator -( const Vector2<T> & a, const Vector2<T> & b ); \
    template Vector2<T> operator *( T a, const Vector2<T> & b ); \
    template Vector2<T> operator *( const Vector2<T> & b, T a ); \
    template Vector2<T> operator /( Vector2<T> b, T a ); \

VEC3(float, 1)
VEC3(double, 1)
VEC3(int, 0)

VEC2(float)
VEC2(double)
VEC2(int)

#undef VEC3
#undef VEC2

#define OBJTYPE(...) \
    template std::vector<std::shared_ptr<__VA_ARGS__>> getAllObjectsInTree( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable ); \
    template std::vector<std::shared_ptr<__VA_ARGS__>> getTopmostVisibleObjects( Object* root, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable ); \
    template std::shared_ptr<__VA_ARGS__> getDepthFirstObject( Object* root, const ObjectSelectivityType& type );

OBJTYPE(Object)
OBJTYPE(VisualObject)
OBJTYPE(ObjectPoints)
OBJTYPE(ObjectPointsHolder)
OBJTYPE(ObjectLines)
OBJTYPE(ObjectLinesHolder)
OBJTYPE(ObjectMesh)
OBJTYPE(ObjectMeshHolder)
OBJTYPE(ObjectDistanceMap)
OBJTYPE(ObjectVoxels)
#undef OBJTYPE

// Those are needed for mrviewerpy:

#define FORCE_INSTANTIATE_TYPE(...) using INST_CAT(_mrbind_inst_,__LINE__) __attribute__((__annotate__("mrbind::instantiate_only"))) = __VA_ARGS__
FORCE_INSTANTIATE_TYPE( std::vector<MR::DistanceMap> );
FORCE_INSTANTIATE_TYPE( std::vector<MR::Mesh> );
FORCE_INSTANTIATE_TYPE( std::vector<std::shared_ptr<MR::Object>> );
FORCE_INSTANTIATE_TYPE( std::vector<MR::PointCloud> );
FORCE_INSTANTIATE_TYPE( std::vector<MR::Polyline3> );
#undef FORCE_INSTANTIATE_TYPE

}

#endif
