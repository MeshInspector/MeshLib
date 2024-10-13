#include "MRPython/MRPython.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRImageSave.h"
#include <MRMesh/MRMesh.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

namespace
{

using namespace MR;

Expected<void> saveDistanceMapToImage( const DistanceMap& dm, const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    const auto image = convertDistanceMapToImage( dm, threshold );
    return ImageSave::toAnySupportedFormat( image, filename );
}

Expected<MR::DistanceMap> loadDistanceMapFromImage( const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    auto resLoad = ImageLoad::fromAnySupportedFormat( filename );
    if ( !resLoad.has_value() )
        return unexpected( resLoad.error() );
    return convertImageToDistanceMap( *resLoad, threshold );
}

}

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, DistanceMap, MR::DistanceMap )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DistanceMap, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( DistanceMap ).
        def( pybind11::init<>() ).
        def( "get",
            []( const MR::DistanceMap& m, std::size_t x, std::size_t y ) -> std::optional<float>
            {
                if ( !m.isInBounds( x, y ) )
                    throw std::out_of_range( "Out of bounds!" );
                return m.get( x, y );
            },
            "Read value at (X,Y), returns Null if out of bounds or if that pixel is invalid (aka infinite distance)." ).
        def( "get",
            []( const MR::DistanceMap& m, std::size_t i ) -> std::optional<float>
            {
                if ( !m.isInBounds( i ) )
                    throw std::out_of_range( "Out of bounds!" );
                return m.get( i );
            },
            "Read value at the flattened index I, returns Null if out of bounds or if that pixel is invalid (aka infinite distance)." ).
        // This validates the coords automatically:
        def( "getInterpolated", ( std::optional<float>( MR::DistanceMap::* )( float, float ) const )& MR::DistanceMap::getInterpolated, "bilinear interpolation between 4 pixels" ).
        def( "isInBounds", ( bool( MR::DistanceMap::* )( std::size_t, std::size_t ) const )& MR::DistanceMap::isInBounds, "Check if coordinates X,Y are not out of bounds." ).
        def( "isInBounds", ( bool( MR::DistanceMap::* )( std::size_t ) const )& MR::DistanceMap::isInBounds, "Check if the flattened coordinate is not out of bounds.").
        def( "isValid", []( const MR::DistanceMap& m, std::size_t x, std::size_t y ) { return m.isInBounds( x, y ) && m.isValid( x, y ); }, "Check if pixel at X,Y is in bounds and valid (i.e. not at infinite distance)" ).
        def( "isValid", []( const MR::DistanceMap& m, std::size_t i ) { return m.isInBounds( i ) && m.isValid( i ); }, "Check if pixel at a flattened coordinate is in bounds and valid (i.e. not at infinite distance)" ).
        def( "resX", &MR::DistanceMap::resX, "X resolution" ).
        def( "resY", &MR::DistanceMap::resY, "Y resolution" ).
        def( "clear", &MR::DistanceMap::clear, "clear all values, set resolutions to zero" ).
        def( "invalidateAll", &MR::DistanceMap::invalidateAll, "invalidate all pixels" ).
        def( "set",
            []( MR::DistanceMap& m, std::size_t x, std::size_t y, float value )
            {
                if ( !m.isInBounds( x, y ) )
                    throw std::out_of_range( "Out of bounds!" );
                m.set( x, y, value );
            },
            "Sets a pixel, throws if the coordinates are out of bounds." ).
        def( "set",
            []( MR::DistanceMap& m, std::size_t i, float value )
            {
                if ( !m.isInBounds( i ) )
                    throw std::out_of_range( "Out of bounds!" );
                m.set( i, value );
            },
            "Sets a pixel at a flattened coordinate, throws if the coordinate is out of bounds." ).
        def( "unset",
            []( MR::DistanceMap& m, std::size_t x, std::size_t y )
            {
                if ( !m.isInBounds( x, y ) )
                    throw std::out_of_range( "Out of bounds!" );
                m.unset( x, y );
            },
            "Sets a pixel to the invalid value (see `isValid()`), throws if the coordinates are out of bounds." ).
        def( "unset",
            []( MR::DistanceMap& m, std::size_t i )
            {
                if ( !m.isInBounds( i ) )
                    throw std::out_of_range( "Out of bounds!" );
                m.unset( i );
            },
            "Sets a pixel to the invalid value (see `isValid()`), throws if the coordinate is out of bounds." );

    pybind11::class_<MR::MeshToDistanceMapParams>( m, "MeshToDistanceMapParams" ).
        def( pybind11::init<>(), "Default constructor. Manual params initialization is required" ).
        def( "setDistanceLimits", &MR::MeshToDistanceMapParams::setDistanceLimits, pybind11::arg( "min" ), pybind11::arg( "max" ),
             "if distance is not in set range, pixel became invalid\n"
             "default value: false. Any distance will be applied (include negative)" ).
        def( "xf", &MR::MeshToDistanceMapParams::xf, "converts in AffineXf3f" ).
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
        def( pybind11::init<>() ).
        def( "xf", &MR::ContourToDistanceMapParams::xf, "converts in AffineXf3f" );

    pybind11::class_<MR::DistanceMapToWorld>( m, "DistanceMapToWorld", "This structure store data to transform distance map to world coordinates" ).
        def( pybind11::init<>(), "Default ctor init all fields with zeros, make sure to fill them manually" ).
        def( pybind11::init<const MR::MeshToDistanceMapParams&>(), "Init fields by `MeshToDistanceMapParams` struct" ).
        def( pybind11::init<const MR::ContourToDistanceMapParams&>(), "Init fields by `ContourToDistanceMapParams` struct" ).
        def( pybind11::init<const MR::AffineXf3f&>(), "Converts from AffineXf3f" ).
        def( "toWorld", &MR::DistanceMapToWorld::toWorld, pybind11::arg( "x" ), pybind11::arg( "y" ), pybind11::arg( "depth" ),
             "Get world coordinate by depth map info.\n"
             "x - float X coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)\n"
             "y - float Y coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)\n"
             "depth - value in distance map, represent depth in world" ).
        def( "xf", &MR::DistanceMapToWorld::xf, "converts in AffineXf3f" ).
        def_readwrite( "orgPoint", &MR::DistanceMapToWorld::orgPoint, "World coordinates of distance map origin corner" ).
        def_readwrite( "pixelXVec", &MR::DistanceMapToWorld::pixelXVec, "Vector in world space of pixel x positive direction.\n"
                                                                        "Note! Length is equal to pixel size. Typically it should be orthogonal to `pixelYVec`." ).
        def_readwrite( "pixelYVec", &MR::DistanceMapToWorld::pixelYVec, "Vector in world space of pixel y positive direction.\n"
                                                                        "Note! Length is equal to pixel size. Typically it should be orthogonal to `pixelXVec`." ).
        def_readwrite( "direction", &MR::DistanceMapToWorld::direction, "Vector of depth direction."
                                                                        "Note! Typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane." );

    pybind11::implicitly_convertible<const MR::AffineXf3f&, MR::DistanceMapToWorld>();

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
        pybind11::arg( "toWorld" ),
        pybind11::arg( "cb" ) = MR::ProgressCallback{},
           "converts distance map back to the mesh fragment with presented params" );

    m.def( "saveDistanceMapToImage",
           MR::decorateExpected( &::saveDistanceMapToImage ),
           pybind11::arg( "distMap" ), pybind11::arg( "filename" ), pybind11::arg( "threshold" ) = 1.0f / 255.0f,
           "saves distance map to a grayscale image file\n"
           "\tthreshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)\n"
           "minimum (close): 1.0 (white)\n"
           "maximum (far): threshold\n"
           "invalid (infinity): 0.0 (black)" );

    m.def( "loadDistanceMapFromImage",
           MR::decorateExpected( &::loadDistanceMapFromImage ),
           pybind11::arg( "filename" ), pybind11::arg( "threshold" ) = 1.0f / 255.0f,
           "load distance map from a grayscale image file\n"
           "\tthreshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid" );

    m.def( "distanceMapTo2DIsoPolyline", ( MR::Polyline2( * )(const MR::DistanceMap&, float) ) &MR::distanceMapTo2DIsoPolyline,
    pybind11::arg( "dm" ), pybind11::arg( "isoValue" ),
    "Converts distance map to 2d iso-lines:\n"
    "Iso-lines are created in space DistanceMap ( plane OXY with pixelSize = (1, 1) )" );

    m.def( "distanceMapTo2DIsoPolyline", ( std::pair<MR::Polyline2, MR::AffineXf3f>( * )( const MR::DistanceMap&, const MR::AffineXf3f&, float, bool ) )& MR::distanceMapTo2DIsoPolyline,
           pybind11::arg( "dm" ), pybind11::arg( "xf" ), pybind11::arg( "isoValue" ), pybind11::arg( "useDepth" ),
           "computes iso-lines of distance map corresponding to given iso-value; "
           "in second returns the transformation from 0XY plane to world; "
           "param useDepth true - the isolines will be located on distance map surface, false - isolines for any iso-value will be located on the common plane xf(0XY)" );

} )

// Distance Map
MR_ADD_PYTHON_VEC( mrmeshpy, vectorDistanceMap, MR::DistanceMap )
