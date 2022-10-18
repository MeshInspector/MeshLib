#include "MRMesh/MRPython.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRFloatGrid.h"
#include "MRMesh/MRVDBConversions.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf3.h"
#include <tl/expected.hpp>
#include <pybind11/functional.h>


#define MR_ADD_PYTHON_VOXELS_VOLUME( Type, TypeText ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Type, []( pybind11::module_& m ) \
{\
    pybind11::class_<MR::Type>( m, #Type, "Voxels representation as " #TypeText ).\
        def( pybind11::init<>() ).\
        def_readwrite( "data", &MR::Type::data ).\
        def_readwrite( "dims", &MR::Type::dims, "Size of voxels space" ).\
        def_readwrite( "voxelSize", &MR::Type::voxelSize ).\
        def_readwrite( "min", &MR::Type::min, "Minimum value from all voxels" ).\
        def_readwrite( "max", &MR::Type::max, "Maximum value from all voxels" );\
} )

MR_ADD_PYTHON_VOXELS_VOLUME( VdbVolume, "VDB FloatGrid" )
MR_ADD_PYTHON_VOXELS_VOLUME( SimpleVolume, "vector of float" )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Voxels, []( pybind11::module_& m )
{
    pybind11::class_<MR::OpenVdbFloatGrid, MR::FloatGrid>( m, "FloatGrid", "Smart pointer to OpenVdbFloatGrid" ).
        def( pybind11::init<>() );

    m.def( "meshToLevelSet", &MR::meshToLevelSet,
        pybind11::arg( "mp" ), pybind11::arg( "xf" ), pybind11::arg( "voxelSize" ), pybind11::arg( "surfaceOffset" ) = 3, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Closed surface is required.\n"
        "SurfaceOffset - number voxels around surface to calculate distance in (should be positive)." );

    m.def( "meshToDistanceField", &MR::meshToDistanceField,
        pybind11::arg( "mp" ), pybind11::arg( "xf" ), pybind11::arg( "voxelSize" ), pybind11::arg( "surfaceOffset" ) = 3, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Does not require closed surface, resulting grid cannot be used for boolean operations.\n"
        "SurfaceOffset - the number voxels around surface to calculate distance in (should be positive)." );

    m.def( "simpleVolumeToDenseGrid", ( MR::FloatGrid( * )( const MR::SimpleVolume&, const MR::ProgressCallback& ) )& MR::simpleVolumeToDenseGrid,
        pybind11::arg( "simpleVolue" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make FloatGrid from SimpleVolume. Make copy of data.\n"
        "Grid can be used to make iso-surface later with gridToMesh function." );

    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::FloatGrid&, const MR::Vector3f&, float, float, const MR::ProgressCallback& ) )& MR::gridToMesh,
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg("cb") = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "isoValue - layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::FloatGrid&, const MR::Vector3f&, int, float, float, const MR::ProgressCallback& ) )& MR::gridToMesh,
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "maxFaces"), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "maxFaces - If mesh faces exceed this value error returns.\n"
        "isoValue - Layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );
} )

MR_ADD_PYTHON_EXPECTED( mrmeshpy, ExpectedVdbVolume, MR::VdbVolume, std::string )