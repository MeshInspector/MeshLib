#pragma once

#ifdef MR_PARSING_FOR_PB11_BINDINGS

#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRVoxels/MRFloatGrid.h"
#include "MRVoxels/MRObjectVoxels.h"

#define INST_IF(cond) MR_CONCAT(INST_IF_, cond)
#define INST_IF_0(...)
#define INST_IF_1(...) __VA_ARGS__

namespace MR
{

#define VEC3(T, isFloatingPoint) \
    template struct Vector3<T>; \
    template Vector3<T> cross( const Vector3<T> & a, const Vector3<T> & b ); \
    template T dot( const Vector3<T> & a, const Vector3<T> & b ); \
    template T sqr( const Vector3<T> & a ); \
    template T mixed( const Vector3<T> & a, const Vector3<T> & b, const Vector3<T> & c ); \
    template Vector3<T> mult( const Vector3<T>& a, const Vector3<T>& b ); \
    template T angle( const Vector3<T> & a, const Vector3<T> & b ); \
    INST_IF(isFloatingPoint)( \
        template Vector3<T> unitVector3( T azimuth, T altitude ); \
    )

#define VEC2(T) \
    template struct Vector2<T>; \
    template T cross( const Vector2<T> & a, const Vector2<T> & b ); \
    template T dot( const Vector2<T> & a, const Vector2<T> & b ); \
    template T sqr( const Vector2<T> & a ); \
    template Vector2<T> mult( const Vector2<T>& a, const Vector2<T>& b ); \
    template T angle( const Vector2<T> & a, const Vector2<T> & b );

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


// Here you can force type registration for certain standard templates and similar things.
// The lack of automatic registration in rare cases is not a defect, we do this intentionally to speed up compilation.
// See `MB_PB11_NO_REGISTER_TYPE_DEPS` for more details.

// For generic types. `...` is a type that needs to be instantiated.
#define FORCE_REGISTER_TYPE(...) using MR_CONCAT(_mrbind_inst_,__LINE__) __attribute__((__annotate__("mrbind::instantiate_only"))) = __VA_ARGS__
// Specifically for parameter types. `...` is a type that needs to be instantiated.
// This is needed for some types, for which using them as parameters automatically adjusts them to different types. E.g. for pointers to scalars `T *`,
//   the parameters become `mrmeshpy.output_T` classes. Since this happens only in parameters, scalar pointers must be registered using this macro.
#define FORCE_REGISTER_PARAM_TYPE(...) __attribute__((__annotate__("mrbind::instantiate_only"))) void MR_CONCAT(_mrbind_inst_,__LINE__)(__VA_ARGS__)
// Specifically for return types. `...` is a type that needs to be instantiated.
// This is similar to `FORCE_REGISTER_PARAM_TYPE`. Some types get adjusted only when used as return types.
#define FORCE_REGISTER_RETURN_TYPE(...) __attribute__((__annotate__("mrbind::instantiate_only"))) __VA_ARGS__ MR_CONCAT(_mrbind_inst_,__LINE__)()

// Those are needed for mrviewerpy:
FORCE_REGISTER_TYPE( std::vector<MR::DistanceMap> );
FORCE_REGISTER_TYPE( std::vector<MR::Mesh> );
FORCE_REGISTER_TYPE( std::vector<std::shared_ptr<MR::Object>> );
FORCE_REGISTER_TYPE( std::vector<MR::PointCloud> );
FORCE_REGISTER_TYPE( std::vector<MR::Polyline3> );
// Those are needed directly for mrmeshpy:
FORCE_REGISTER_TYPE( std::monostate );
FORCE_REGISTER_TYPE( Expected<MR::VoxelsLoad::DicomVolumeT<MR::VoxelsVolumeMinMax<FloatGrid>>> );
FORCE_REGISTER_TYPE( Expected<MR::VoxelsLoad::DicomVolumeT<MR::VoxelsVolumeMinMax<Vector<float, MR::VoxelId>>>> );
FORCE_REGISTER_PARAM_TYPE( double * );
// ---

#undef FORCE_REGISTER_TYPE
#undef FORCE_REGISTER_PARAM_TYPE
#undef FORCE_REGISTER_RETURN_TYPE

}

#endif
