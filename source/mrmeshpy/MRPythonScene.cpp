#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeta.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"

#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#endif

// NOTE: see the disclaimer in the header file
#include "MRPython/MRPython.h"

namespace
{

template <typename T, auto M, typename ReturnType = std::remove_cvref_t<decltype((std::declval<T>().*M)())>>
auto extractModel( const MR::Object& object ) -> std::unique_ptr<typename MR::Meta::SharedPtrTraits<ReturnType>::elemType>
{
    if ( auto base = dynamic_cast<const T*>( &object ) )
    {
        if constexpr ( MR::Meta::SharedPtrTraits<ReturnType>::isSharedPtr )
        {
            if ( auto ret = (base->*M)() )
                return std::make_unique<std::remove_cvref_t<decltype( *ret )>>( *ret );
        }
        else
        {
            return std::make_unique<ReturnType>( (base->*M)() );
        }
    }

    return nullptr;
}

} // namespace

MR_ADD_PYTHON_CUSTOM_CLASS_DECL( mrmeshpy, SceneObject, MR::Object, std::shared_ptr<MR::Object> )
MR_ADD_PYTHON_CUSTOM_CLASS_INST( mrmeshpy, SceneObject )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorSceneObject, std::shared_ptr<MR::Object> )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SceneObject, []( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( SceneObject )
        .def( "extractMesh", extractModel<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::mesh>, "Mesh of this object, or None." )
        .def( "extractPoints", extractModel<MR::ObjectPointsHolder, &MR::ObjectPointsHolder::pointCloud>, "Pointcloud of this object, or None." )
        .def( "extractLines", extractModel<MR::ObjectLinesHolder, &MR::ObjectLinesHolder::polyline>, "Polyline of this object, or None." )
        .def( "extractDistanceMap", extractModel<MR::ObjectDistanceMap, &MR::ObjectDistanceMap::getDistanceMap>, "Distance map of this object, or None." )
        .def( "xf", []( const MR::Object& o, MR::ViewportId v ){ return o.xf( v ); }, pybind11::arg_v( "viewport", MR::ViewportId{}, "meshlib.mrmeshpy.ViewportId()" ), "Mapping from object space to parent object space." )
        .def( "worldXf", []( const MR::Object& o, MR::ViewportId v ){ return o.worldXf( v ); }, pybind11::arg_v( "viewport", MR::ViewportId{}, "meshlib.mrmeshpy.ViewportId()" ), "Mapping from object space to world space." )
        .def( "children", []( MR::Object& o ) -> auto & { return o.children(); }, "Retruns the child objects of an object." )
    ;
} )

#ifndef MESHLIB_NO_VOXELS
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SceneObjectVoxels, []( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( SceneObject )
        .def( "extractVoxels", extractModel<MR::ObjectVoxels, &MR::ObjectVoxels::vdbVolume>, "Voxels of this object, or None." )
    ;
} )
#endif
