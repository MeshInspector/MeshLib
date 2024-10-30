#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRVoxelsSave.h"
#include "MRVoxels/MRVoxelsLoad.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRVoxels/MRVDBConversions.h"
// NOTE: see the disclaimer in the header file
#include "MRPython/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRExpected.h"
#include "MRVoxels/MRMeshToDistanceVolume.h"
#include "MRVoxels/MRTeethMaskToDirectionVolume.h"
#include "MRVoxels/MRVoxelsApplyTransform.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

#define MR_ADD_PYTHON_VOXELS_VOLUME( Type, TypeText ) \
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Type, MR::Type ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Type, [] ( pybind11::module_& ) \
{                                                     \
    MR_PYTHON_CUSTOM_CLASS( Type ).doc() =                                       \
        "Voxels representation as " #TypeText;        \
    MR_PYTHON_CUSTOM_CLASS( Type ).                                              \
        def( pybind11::init<>() ).\
        def_readwrite( "data", &MR::Type::data ).\
        def_readwrite( "dims", &MR::Type::dims, "Size of voxels space" ).\
        def_readwrite( "voxelSize", &MR::Type::voxelSize );\
} )

MR_ADD_PYTHON_VOXELS_VOLUME( SimpleVolume, "vector of float" )

#define MR_ADD_PYTHON_VOXELS_VOLUME_MINMAX( Type, TypeText ) \
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Type, [] ( pybind11::module_& ) \
{                                                     \
    MR_PYTHON_CUSTOM_CLASS( Type ).doc() =                                       \
        "Voxels representation as " #TypeText;        \
    MR_PYTHON_CUSTOM_CLASS( Type ).                                              \
        def( pybind11::init<>() ).\
        def_readwrite( "data", &MR::Type::data ).\
        def_readwrite( "dims", &MR::Type::dims, "Size of voxels space" ).\
        def_readwrite( "voxelSize", &MR::Type::voxelSize ).\
        def_readwrite( "min", &MR::Type::min, "Minimum value from all voxels" ).\
        def_readwrite( "max", &MR::Type::max, "Maximum value from all voxels" );\
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VdbVolume, MR::VdbVolume )
MR_ADD_PYTHON_VOXELS_VOLUME_MINMAX( VdbVolume, "VDB FloatGrid" )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, SimpleVolumeMinMax, MR::SimpleVolumeMinMax, MR::SimpleVolume )
MR_ADD_PYTHON_VOXELS_VOLUME_MINMAX( SimpleVolumeMinMax, "vector of float" )

MR_ADD_PYTHON_CUSTOM_CLASS_DECL( mrmeshpy, FloatGrid, MR::OpenVdbFloatGrid, MR::FloatGrid )
MR_ADD_PYTHON_CUSTOM_CLASS_INST( mrmeshpy, FloatGrid )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorVdbVolume, MR::VdbVolume )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Voxels, []( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( FloatGrid ).doc() =
        "Smart pointer to OpenVdbFloatGrid";
    MR_PYTHON_CUSTOM_CLASS( FloatGrid ).
        def( pybind11::init<>() );

    m.def( "meshToLevelSet", &MR::meshToLevelSet,
        pybind11::arg( "mp" ), pybind11::arg( "xf" ), pybind11::arg( "voxelSize" ), pybind11::arg( "surfaceOffset" ) = 3, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Closed surface is required.\n"
        "SurfaceOffset - number voxels around surface to calculate distance in (should be positive)." );

    m.def( "meshToDistanceField", &MR::meshToDistanceField,
        pybind11::arg( "mp" ), pybind11::arg( "xf" ), pybind11::arg( "voxelSize" ), pybind11::arg( "surfaceOffset" ) = 3, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Does not require closed surface, resulting grid cannot be used for boolean operations.\n"
        "SurfaceOffset - the number voxels around surface to calculate distance in (should be positive)." );

    m.def( "simpleVolumeToDenseGrid", &MR::simpleVolumeToDenseGrid,
        pybind11::arg( "simpleVolume" ), pybind11::arg( "background" ) = 0.0f, pybind11::arg("cb") = MR::ProgressCallback{},
        "Make FloatGrid from SimpleVolume. Make copy of data.\n"
        "Grid can be used to make iso-surface later with gridToMesh function." );
    m.def( "simpleVolumeToVdbVolume", &MR::simpleVolumeToVdbVolume,
        pybind11::arg( "simpleVolumeMinMax" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make VdbVolume from SimpleVolumeMinMax. Make copy of data.\n"
        "Grid can be used to make iso-surface later with gridToMesh function." );
    m.def( "vdbVolumeToSimpleVolume",
        MR::decorateExpected(
            []( const MR::VdbVolume& volume, MR::ProgressCallback cb ) { return vdbVolumeToSimpleVolume( volume, {}, cb ); } ),
        pybind11::arg( "vdbVolume" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make SimpleVolume from VdbVolume. Make copy of data." );

    m.def( "gridToMesh",
        MR::decorateExpected( []( const MR::FloatGrid& grid, const MR::Vector3f& voxelSize, float isoValue, float adaptivity, MR::ProgressCallback cb )
        {
            return gridToMesh( grid, MR::GridToMeshSettings{
                .voxelSize = voxelSize,
                .isoValue = isoValue,
                .adaptivity = adaptivity,
                .cb = cb
            } );
        } ),
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg("cb") = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "isoValue - layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    m.def( "gridToMesh",
        MR::decorateExpected( []( const MR::VdbVolume& vdbVolume, float isoValue, float adaptivity, MR::ProgressCallback cb )
        {
            return gridToMesh( vdbVolume.data, MR::GridToMeshSettings{
                .voxelSize = vdbVolume.voxelSize,
                .isoValue = isoValue,
                .adaptivity = adaptivity,
                .cb = cb
            } );
        } ),
        pybind11::arg( "vdbVolume" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from VdbVolume.\n"
        "isoValue - layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    m.def( "gridToMesh",
        MR::decorateExpected( []( const MR::FloatGrid& grid, const MR::Vector3f& voxelSize, int maxFaces, float isoValue, float adaptivity, MR::ProgressCallback cb )
        {
            return gridToMesh( grid, MR::GridToMeshSettings{
                .voxelSize = voxelSize,
                .isoValue = isoValue,
                .adaptivity = adaptivity,
                .maxFaces = maxFaces,
                .cb = cb
            } );
        } ),
        pybind11::arg( "grid" ), pybind11::arg( "voxelSize" ), pybind11::arg( "maxFaces"), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from FloatGrid.\n"
        "maxFaces - If mesh faces exceed this value error returns.\n"
        "isoValue - Layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    m.def( "gridToMesh",
        MR::decorateExpected( []( const MR::VdbVolume& vdbVolume, int maxFaces, float isoValue, float adaptivity, MR::ProgressCallback cb )
        {
            return gridToMesh( vdbVolume.data, MR::GridToMeshSettings{
                .voxelSize = vdbVolume.voxelSize,
                .isoValue = isoValue,
                .adaptivity = adaptivity,
                .maxFaces = maxFaces,
                .cb = cb
            } );
        } ),
        pybind11::arg( "vdbVolume" ), pybind11::arg( "maxFaces" ), pybind11::arg( "isoValue" ) = 0.f, pybind11::arg( "adaptivity" ) = 0.f, pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Make mesh from VdbVolume.\n"
        "maxFaces - If mesh faces exceed this value error returns.\n"
        "isoValue - Layer of grid with this value would be converted in mesh.\n"
        "isoValue can be negative only in level set grids.\n"
        "adaptivity - [0.0;1.0] Ratio of combining small triangles into bigger ones.\n"
        "(Curvature can be lost on high values.)" );

    pybind11::enum_<MR::SlicePlane>( m, "SlicePlane" ).
        value( "XY", MR::SlicePlane::XY, "XY plane" ).
        value( "YZ", MR::SlicePlane::YZ, "YZ plane" ).
        value( "ZX", MR::SlicePlane::ZX, "XZ plane" ).
        value( "None", MR::SlicePlane::None, "None" );

    m.def( "saveSliceToImage",
        MR::decorateExpected( ( MR::Expected<void>( * )( const std::filesystem::path&, const MR::VdbVolume&, const MR::SlicePlane&, int, MR::ProgressCallback ) )& MR::VoxelsSave::saveSliceToImage ),
        pybind11::arg( "path" ), pybind11::arg( "vdbVolume" ), pybind11::arg( "slicePlane" ), pybind11::arg( "sliceNumber" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Save the slice by the active plane through the sliceNumber to an image file.\n" );

    pybind11::class_<MR::VoxelsSave::SavingSettings>( m, "VoxelsSaveSavingSettings",
        "stores together all data for save voxel object as a group of images" ).
        def( pybind11::init<>() ).
        def_readwrite( "path", &MR::VoxelsSave::SavingSettings::path, "path to directory where you want to save images" ).
        def_readwrite( "format", &MR::VoxelsSave::SavingSettings::format,
            "format for file names, you should specify a placeholder for number and extension, e.g. \"slice_{ 0:0{1} }.tif\"" ).
        def_readwrite( "slicePlane", &MR::VoxelsSave::SavingSettings::slicePlane, "Plane which the object is sliced by. XY, XZ, or YZ" );

    m.def( "saveAllSlicesToImage",
       MR::decorateExpected( &MR::VoxelsSave::saveAllSlicesToImage ),
       pybind11::arg( "vdbVolume" ), pybind11::arg( "settings"),
       "save all slices by the active plane through all voxel planes along the active axis to an image file" );

    pybind11::enum_<MR::VoxelsLoad::GridType>( m, "GridType" ).
        value( "DenseGrid", MR::VoxelsLoad::GridType::DenseGrid, "Represents dense volume" ).
        value( "LevelSet", MR::VoxelsLoad::GridType::LevelSet, "Represents distances volume" );

    pybind11::class_<MR::VoxelsLoad::LoadingTiffSettings>( m, "LoadingTiffSettings", "Settings structure for loadTiffDir function" ).
        def( pybind11::init<>() ).
        def_readwrite( "dir", &MR::VoxelsLoad::LoadingTiffSettings::dir, "Path to directory" ).
        def_readwrite( "voxelSize", &MR::VoxelsLoad::LoadingTiffSettings::voxelSize, "Size of voxel" ).
        def_readwrite( "gridType", &MR::VoxelsLoad::LoadingTiffSettings::gridType, "Type of the grid: DenseGrid or LevelSet" ).
        def_readwrite( "progressCallback", &MR::VoxelsLoad::LoadingTiffSettings::cb, "Callback to report progress" );

    m.def( "loadTiffDir",
        MR::decorateExpected( ( MR::Expected<MR::VdbVolume>( * ) ( const MR::VoxelsLoad::LoadingTiffSettings& ) )& MR::VoxelsLoad::loadTiffDir ),
        pybind11::arg( "settings" ),
        "Load voxels from a directory with TIFF images.\n",
        "settings - Settings structure for loadTiffDir function\n" );

    m.def( "floatGridToVdbVolume", &MR::floatGridToVdbVolume, pybind11::arg( "grid" ),
        "fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)" );

    pybind11::enum_<MR::MeshToVolumeParams::Type>( m, "MeshToVolumeParamsType", "Conversion type" ).
        value( "Signed", MR::MeshToVolumeParams::Type::Signed, "only closed meshes can be converted with signed type" ).
        value( "Unsigned", MR::MeshToVolumeParams::Type::Unsigned, "this type leads to shell like iso-surfaces" );

    pybind11::class_<MR::MeshToVolumeParams>( m, "MeshToVolumeParams", "Parameters structure for meshToVolume function" ).
        def( pybind11::init<>() ).
        def_readwrite( "type", &MR::MeshToVolumeParams::type, "Conversion type" ).
        def_readwrite( "surfaceOffset", &MR::MeshToVolumeParams::surfaceOffset, "the number of voxels around surface to calculate distance in (should be positive)" ).
        def_readwrite( "voxelSize", &MR::MeshToVolumeParams::voxelSize, "Conversion type" ).
        def_readwrite( "worldXf", &MR::MeshToVolumeParams::worldXf, "mesh initial transform" ).
        def_readwrite( "outXf", &MR::MeshToVolumeParams::outXf, "optional output: xf to original mesh (respecting worldXf)" );

    m.def( "meshToVolume",
        MR::decorateExpected( &MR::meshToVolume ),
        pybind11::arg( "mesh" ),
        pybind11::arg_v( "params", MR::MeshToVolumeParams(), "MeshToVolumeParams()" ),
        "convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box" );

    pybind11::class_<MR::MeshToDistanceVolumeParams>( m, "MeshToDistanceVolumeParams" ).
        def( pybind11::init<>() ).
        def_property( "origin", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.vol.origin; },
            [] ( MR::MeshToDistanceVolumeParams & p, const MR::Vector3f & v ) { p.vol.origin = v; }, "origin point of voxels box" ).
        def_property( "voxelSize", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.vol.voxelSize; },
            [] ( MR::MeshToDistanceVolumeParams & p, const MR::Vector3f & v ) { p.vol.voxelSize = v; }, "size of voxel on each axis" ).
        def_property( "dimensions", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.vol.dimensions; },
            [] ( MR::MeshToDistanceVolumeParams & p, const MR::Vector3i & v ) { p.vol.dimensions = v; }, "num voxels along each axis" ).
        def_property( "minDistSq", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.dist.minDistSq; },
            [] ( MR::MeshToDistanceVolumeParams & p, float v ) { p.dist.minDistSq = v; }, "minimum squared value in a voxel" ).
        def_property( "maxDistSq", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.dist.maxDistSq; },
            [] ( MR::MeshToDistanceVolumeParams & p, float v ) { p.dist.maxDistSq = v; }, "maximum squared value in a voxel" ).
        def_property( "signMode", [] ( const MR::MeshToDistanceVolumeParams & p ) { return p.dist.signMode; },
            [] ( MR::MeshToDistanceVolumeParams & p, MR::SignDetectionMode v ) { p.dist.signMode = v; }, "the method to compute distance sign" );

    m.def( "meshToDistanceVolume", MR::decorateExpected( &MR::meshToDistanceVolume ),
        pybind11::arg( "mesh" ), pybind11::arg_v( "params", MR::MeshToDistanceVolumeParams(), "MeshToDistanceVolumeParams()" ),
        "makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings" );

    m.def( "transformVdbVolume", &MR::transformVdbVolume,
           pybind11::arg( "volume" ), pybind11::arg( "xf" ), pybind11::arg_v( "fixBox", false ), pybind11::arg_v( "box", MR::Box3f{}, "Box3f()" ) );

    m.def( "teethMaskToDirectionVolume", MR::decorateExpected( &MR::teethMaskToDirectionVolume ),
           pybind11::arg( "volume" ), pybind11::arg_v( "additional ids", std::vector<int>{} ), "Convert 3d teeth mask into directional volume" );
} )
#endif
