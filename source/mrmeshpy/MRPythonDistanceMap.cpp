#include "MRMesh/MRPython.h"
#include "MRMesh/MRDistanceMap.h"
#include <MRMesh/MRMesh.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

// Distance Map
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DistanceMap, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::DistanceMap>( m, "DistanceMap" ).
        def( pybind11::init<>() ).
        def( "get", static_cast< std::optional<float>( MR::DistanceMap::* )( size_t, size_t ) const >( &MR::DistanceMap::get ), "read X,Y value" ).
        def( "get", static_cast< std::optional<float>( MR::DistanceMap::* )( size_t ) const >( &MR::DistanceMap::get ), "read value by index" ).
        def( "getInterpolated", ( std::optional<float>( MR::DistanceMap::* )( float, float ) const )& MR::DistanceMap::getInterpolated, "bilinear interpolation between 4 pixels" ).
        def( "isValid", ( bool( MR::DistanceMap::* )( size_t, size_t ) const )& MR::DistanceMap::isValid, "check if X,Y pixel is valid" ).
        def( "isValid", ( bool( MR::DistanceMap::* )( size_t ) const )& MR::DistanceMap::isValid, "check if index pixel is valid").
        def( "resX", &MR::DistanceMap::resX, "X resolution" ).
        def( "resY", &MR::DistanceMap::resY, "Y resolution" ).
        def( "clear", &MR::DistanceMap::clear, "clear all values, set resolutions to zero" ).
        def( "invalidateAll", &MR::DistanceMap::invalidateAll, "invalidate all pixels" ).
        def( "set", static_cast< void( MR::DistanceMap::* )( size_t, float ) >( &MR::DistanceMap::set ), "write value by index" ).
        def( "set", static_cast< void( MR::DistanceMap::* )( size_t, size_t, float ) >( &MR::DistanceMap::set ), "write X,Y value" ).
        def( "unset", static_cast< void( MR::DistanceMap::* )( size_t, size_t ) >( &MR::DistanceMap::unset), "invalidate X,Y pixel" ).
        def( "unset", static_cast< void( MR::DistanceMap::* )( size_t ) >( &MR::DistanceMap::unset), "invalidate by index" );

    pybind11::class_<MR::MeshToDistanceMapParams>( m, "MeshToDistanceMapParams" ).
        def( pybind11::init<>(), "Default constructor. Manual params initialization is required" ).
        def( "setDistanceLimits", &MR::MeshToDistanceMapParams::setDistanceLimits, pybind11::arg( "min" ), pybind11::arg( "max" ),
             "if distance is not in set range, pixel became invalid\n"
             "default value: false. Any distance will be applied (include negative)" ).
        def_readwrite( "xRange", &MR::MeshToDistanceMapParams::xRange, "Cartesian range vector between distance map borders in X direction" ).
        def_readwrite( "yRange", &MR::MeshToDistanceMapParams::yRange, "Cartesian range vector between distance map borders in Y direction" ).
        def_readwrite( "direction", &MR::MeshToDistanceMapParams::direction, "direction of intersection ray" ).
        def_readwrite( "orgPoint", &MR::MeshToDistanceMapParams::orgPoint, "location of (0,0) pixel with value 0.f" ).
        def_readwrite( "useDistanceLimits", &MR::MeshToDistanceMapParams::useDistanceLimits, "out of limits intersections will be set to non-valid" ).
        def_readwrite( "allowNegativeValues", &MR::MeshToDistanceMapParams::allowNegativeValues, "allows to find intersections in backward to direction vector with negative values" ).
        def_readwrite( "minValue", &MR::MeshToDistanceMapParams::minValue, "Using of this parameter depends on useDistanceLimits" ).
        def_readwrite( "maxValue", &MR::MeshToDistanceMapParams::maxValue, "Using of this parameter depends on useDistanceLimits" ).
        def_readwrite( "resolution", &MR::MeshToDistanceMapParams::resolution, "resolution of distance map" );

    pybind11::class_<MR::ContourToDistanceMapParams>( m, "ContourToDistanceMapParams" ).
        def( pybind11::init<>() );

    pybind11::class_<MR::DistanceMapToWorld>( m, "DistanceMapToWorld", "This structure store data to transform distance map to world coordinates" ).
        def( pybind11::init<>(), "Default ctor init all fields with zeros, make sure to fill them manually" ).
        def( pybind11::init<const MR::MeshToDistanceMapParams&>(), "Init fields by `MeshToDistanceMapParams` struct" ).
        def( pybind11::init<const MR::ContourToDistanceMapParams&>(), "Init fields by `ContourToDistanceMapParams` struct" ).
        def( "toWorld", &MR::DistanceMapToWorld::toWorld, pybind11::arg( "x" ), pybind11::arg( "y" ), pybind11::arg( "depth" ),
             "Get world coordinate by depth map info.\n"
             "x - float X coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)\n"
             "y - float Y coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)\n"
             "depth - value in distance map, represent depth in world" ).
        def_readwrite( "orgPoint", &MR::DistanceMapToWorld::orgPoint, "World coordinates of distance map origin corner" ).
        def_readwrite( "pixelXVec", &MR::DistanceMapToWorld::pixelXVec, "Vector in world space of pixel x positive direction.\n"
                                                                        "Note! Length is equal to pixel size. Typically it should be orthogonal to `pixelYVec`." ).
        def_readwrite( "pixelYVec", &MR::DistanceMapToWorld::pixelYVec, "Vector in world space of pixel y positive direction.\n"
                                                                        "Note! Length is equal to pixel size. Typically it should be orthogonal to `pixelXVec`." ).
        def_readwrite( "direction", &MR::DistanceMapToWorld::direction, "Vector of depth direction."
                                                                        "Note! Typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane." );

    m.def( "computeDistanceMapD", []( const MR::MeshPart& mp, const MR::MeshToDistanceMapParams& params, MR::ProgressCallback cb )
           { return MR::computeDistanceMapD( mp, params, cb ); },
           pybind11::arg( "mp" ), pybind11::arg( "params" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
           "computes distance map for presented projection parameters\n"
           "use MeshToDistanceMapParams constructor instead of overloads of this function\n"
           "MeshPart - input 3d model\n"
           "general call. You could customize params manually" );

    m.def( "distanceMapToMesh", 
        MR::decorateExpected( &MR::distanceMapToMesh ),
        pybind11::arg( "mp" ), 
        pybind11::arg( "toWorldStruct" ),
        pybind11::arg( "cb" ) = MR::ProgressCallback{},
           "converts distance map back to the mesh fragment with presented params" );

    m.def( "saveDistanceMapToImage",
           MR::decorateExpected( &MR::saveDistanceMapToImage ),
           pybind11::arg( "distMap" ), pybind11::arg( "filename" ), pybind11::arg( "threshold" ) = 1.0f / 255.0f,
           "saves distance map to monochrome image in scales of gray:\n"
           "\tthreshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)\n"
           "minimum (close): 1.0 (white)\n"
           "maximum (far): threshold\n"
           "invalid (infinity): 0.0 (black)" );

    m.def( "loadDistanceMapFromImage",
           MR::decorateExpected( &MR::loadDistanceMapFromImage ),
           pybind11::arg( "filename" ), pybind11::arg( "threshold" ) = 1.0f / 255.0f,
           "load distance map from monochrome image file\n"
           "\tthreshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid" );

    m.def( "distanceMapTo2DIsoPolyline", ( MR::Polyline2( * )(const MR::DistanceMap&, float) ) &MR::distanceMapTo2DIsoPolyline,
    pybind11::arg( "dm" ), pybind11::arg( "isoValue" ),
    "Converts distance map to 2d iso-lines:\n"
    "Iso-lines are created in space DistanceMap ( plane OXY with pixelSize = (1, 1) )" );

    m.def( "distanceMapTo2DIsoPolyline", ( std::pair<MR::Polyline2, MR::AffineXf3f>( * )( const MR::DistanceMap&, const MR::DistanceMapToWorld&, float, bool ) )& MR::distanceMapTo2DIsoPolyline,
           pybind11::arg( "dm" ), pybind11::arg( "params" ), pybind11::arg( "isoValue" ), pybind11::arg( "useDepth" ),
           "Iso-lines are created in real space.\n"
           "( contours plane with parameters according DistanceMapToWorld )\n"
           "Return: pair contours in OXY & transformation from plane OXY to real contours plane" );

} )
