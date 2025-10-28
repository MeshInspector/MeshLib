#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// determines the weight or mass of each vertex in applications like Laplacian
enum class VertexMass
{
    /// all vertices have same mass=1
    Unit = 0,

    /// vertex mass depends on local geometry and proportional to the area of first-ring triangles
    NeiArea
};

/// determines the weight of each edge in applications like Laplacian
enum class EdgeWeights
{
    /// all edges have same weight=1
    Unit = 0,

    /// edge weight depends on local geometry and uses cotangent values
    Cotan

    /// cotangent edge weights and equation weights inversely proportional to square root of local area
    // CotanWithAreaEqWeight => use EdgeWeights::Cotan and VertexMass::NeiArea instead
};

/// typically returned from callbacks to control the behavior of main algorithm
enum class Processing : bool
{
    Continue,
    Stop
};

/// the method how to choose between two opposite normal orientations
enum class OrientNormals
{
    TowardOrigin,
    AwayFromOrigin,
    Smart
};

enum class OffsetMode : int
{
    Smooth,     ///< create mesh using dual marching cubes from OpenVDB library
    Standard,   ///< create mesh using standard marching cubes implemented in MeshLib
    Sharpening  ///< create mesh using standard marching cubes with additional sharpening implemented in MeshLib
};

/// Type of object coloring,
/// \note that texture are applied over main coloring
enum class ColoringType
{
    SolidColor,   ///< Use one color for whole object
    PrimitivesColorMap, ///< Use different color (taken from faces colormap) for each primitive
    FacesColorMap = PrimitivesColorMap, ///< Use different color (taken from faces colormap) for each face (primitive for object mesh)
    LinesColorMap = PrimitivesColorMap, ///< Use different color (taken from faces colormap) for each line (primitive for object lines)
    VertsColorMap  ///< Use different color (taken from verts colormap) for each vertex
};

/// returns string representation of enum values
[[nodiscard]] MRMESH_API const char * asString( ColoringType ct );

enum class UseAABBTree : char
{
    No,  // AABB-tree of the mesh will not be used, even if it is available
    Yes, // AABB-tree of the mesh will be used even if it has to be constructed
    YesIfAlreadyConstructed, // AABB-tree of the mesh will be used if it was previously constructed and available, and will not be used otherwise
};

/// the algorithm to compute approximately geodesic path
enum class GeodesicPathApprox : char
{
    /// compute edge-only path by building it from start and end simultaneously
    DijkstraBiDir,
    /// compute edge-only path using A*-search algorithm
    DijkstraAStar,
    /// use Fast Marching algorithm
    FastMarching
};

// A stub measurement unit representing no unit.
enum class NoUnit
{
    _count [[maybe_unused]]
};

// Measurement units of length.
enum class LengthUnit
{
    microns,
    millimeters,
    centimeters,
    meters,
    inches,
    feet,
    _count [[maybe_unused]],
};

// Measurement units of angle.
enum class AngleUnit
{
    radians,
    degrees,
    _count [[maybe_unused]],
};

// Measurement units of screen sizes.
enum class PixelSizeUnit
{
    pixels,
    _count [[maybe_unused]],
};

// Measurement units for factors / ratios.
enum class RatioUnit
{
    factor, // 0..1 x
    percents, // 0..100 %
    _count [[maybe_unused]],
};

// Measurement units for time.
enum class TimeUnit
{
    seconds,
    milliseconds,
    _count [[maybe_unused]],
};

// Measurement units for movement speed.
enum class MovementSpeedUnit
{
    micronsPerSecond,
    millimetersPerSecond,
    centimetersPerSecond,
    metersPerSecond,
    inchesPerSecond,
    feetPerSecond,
    _count [[maybe_unused]],
};

// Measurement units for surface area.
enum class AreaUnit
{
    microns2,
    millimeters2,
    centimeters2,
    meters2,
    inches2,
    feet2,
    _count [[maybe_unused]],
};

// Measurement units for body volume.
enum class VolumeUnit
{
    microns3,
    millimeters3,
    centimeters3,
    meters3,
    inches3,
    feet3,
    _count [[maybe_unused]],
};

// Measurement units for 1/length.
enum class InvLengthUnit
{
    inv_microns,
    inv_millimeters,
    inv_centimeters,
    inv_meters,
    inv_inches,
    inv_feet,
    _count [[maybe_unused]],
};

} //namespace MR
