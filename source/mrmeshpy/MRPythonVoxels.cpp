#include "MRMesh/MRPython.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRVoxelsSave.h"
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

    m.def( "simpleVolumeToDenseGrid", ( MR::FloatGrid( * )( const MR::SimpleVolume&, MR::ProgressCallback ) )& MR::simpleVolumeToDenseGrid,
        pybind11::arg( "simpleVolume" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make FloatGrid from SimpleVolume. Make copy of data.\n"
        "Grid can be used to make iso-surface later with gridToMesh function." );
    m.def( "simpleVolumeToVdbVolume", ( MR::VdbVolume( * )( const MR::SimpleVolume&, MR::ProgressCallback ) )& MR::simpleVolumeToVdbVolume,
        pybind11::arg( "simpleVolume" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make VdbVolume from SimpleVolume. Make copy of data.\n"
        "Grid can be used to make iso-surface later with gridToMesh function." );

    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::FloatGrid&, const MR::Vector3f&, float, float, MR::ProgressCallback ) )& MR::gridToMesh,
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg("cb") = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "isoValue - layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );
    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::VdbVolume&, float, float, MR::ProgressCallback ) )& MR::gridToMesh,
        pybind11::arg( "vdbVolume" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from VdbVolume.\n"
        "isoValue - layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::FloatGrid&, const MR::Vector3f&, int, float, float, MR::ProgressCallback ) )& MR::gridToMesh,
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "maxFaces"), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "maxFaces - If mesh faces exceed this value error returns.\n"
        "isoValue - Layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );
    m.def( "gridToMesh", ( tl::expected<MR::Mesh, std::string>( * )( const MR::VdbVolume&, int, float, float, MR::ProgressCallback ) )& MR::gridToMesh,
        pybind11::arg( "vdbVolume" ), pybind11::arg( "maxFaces" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from VdbVolume.\n" 
        "maxFaces - If mesh faces exceed this value error returns.\n"
        "isoValue - Layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    pybind11::enum_<MR::SlicePlain>( m, "SlicePlain" ).
        value( "XY", MR::SlicePlain::XY, "XY plain" ).
        value( "YZ", MR::SlicePlain::YZ, "YZ plain" ).
        value( "XZ", MR::SlicePlain::YZ, "XZ plain" ).
        value( "None", MR::SlicePlain::None, "None" );

    m.def( "saveSliceToImage", ( tl::expected<void, std::string>( * )( const std::filesystem::path&, const MR::VdbVolume&, const MR::SlicePlain&, int, MR::ProgressCallback ) )& MR::VoxelsSave::saveSliceToImage,
        pybind11::arg( "path" ), pybind11::arg( "vdbVolume" ), pybind11::arg( "slicePlain" ), pybind11::arg( "sliceNumber" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Save the slice by the active plane through the sliceNumber to an image file.\n" );

    m.def( "saveAllSlicesToImage", ( tl::expected<void, std::string>( * )( const std::filesystem::path&, const MR::VdbVolume&, const MR::SlicePlain&, MR::ProgressCallback ) )& MR::VoxelsSave::saveAllSlicesToImage,
       pybind11::arg( "path" ), pybind11::arg( "vdbVolume" ), pybind11::arg( "slicePlain" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
       "Save the slice by the active plane through the sliceNumber to an image file.\n" );
} )

MR_ADD_PYTHON_EXPECTED( mrmeshpy, ExpectedVdbVolume, MR::VdbVolume, std::string )
