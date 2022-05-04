#pragma once

#include "MRMatrix.h"
#include "MRBitSet.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRBox.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRVector2.h"

namespace MR
{

/// \addtogroup DistanceMapGroup
/// \{

struct MeshToDistanceMapParams
{
    /// default constructor. Manual params initialization is required:
    MeshToDistanceMapParams() = default;

    /// direction vector shows the projections vector to the distance map for points on model
    /// yRange and xRange directions make orthonormal basis with direction
    /// see Vector3<T>::perpendicular() for more details
    /// All Output Distance map values will be positive
    /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
    /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
    MRMESH_API MeshToDistanceMapParams( const Vector3f& direction, const Vector2i& resolution, const MeshPart& mp, bool usePreciseBoundingBox = false );

    /// input matrix should be orthonormal!
    /// rotation.z - direction
    /// rotation.x * (box X length) - xRange
    /// rotation.y * (box Y length) - yRange
    /// All Output Distance map values will be positive
    MRMESH_API MeshToDistanceMapParams( const Matrix3f& rotation, const Vector3f& origin, const Vector2i& resolution, const Vector2f& size );
    MRMESH_API MeshToDistanceMapParams( const Matrix3f& rotation, const Vector3f& origin, const Vector2f& pixelSize, const Vector2i& resolution );

    /// input matrix should be orthonormal!
    /// rotation.z - direction
    /// rotation.x * (box X length) - xRange
    /// rotation.y * (box Y length) - yRange
    /// All Output Distance map values will be positive
    /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
    /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
    MRMESH_API MeshToDistanceMapParams( const Matrix3f& rotation, const Vector2i& resolution, const MeshPart& mp, bool usePreciseBoundingBox = false );

    /// The most general constructor. Use it if you have to find special view, resolution,
    /// distance map with visual the part of the model etc.
    /// All params match is in the user responsibility
    /// xf.b - origin point: pixel(0,0) with value 0.
    /// xf.A.z - direction
    /// xf.A.x - xRange
    /// xf.A.y - yRange
    /// All Output Distance map values could be positive and negative by default. Set allowNegativeValues to false if negative values are not required
    MRMESH_API MeshToDistanceMapParams( const AffineXf3f& xf, const Vector2i& resolution, const Vector2f& size );
    MRMESH_API MeshToDistanceMapParams( const AffineXf3f& xf, const Vector2f& pixelSize, const Vector2i& resolution );

    Vector3f xRange = Vector3f( 1.f, 0.f, 0.f ); ///< Cartesian range vector between distance map borders in X direction
    Vector3f yRange = Vector3f( 0.f, 1.f, 0.f ); ///< Cartesian range vector between distance map borders in Y direction
    Vector3f direction = Vector3f( 0.f, 0.f, 1.f ); ///< direction of intersection ray
    Vector3f orgPoint = Vector3f( 0.f, 0.f, 0.f ); ///< location of (0,0) pixel with value 0.f

    /// if distance is not in set range, pixel became invalid
    /// default value: false. Any distance will be applied (include negative)
    void setDistanceLimits( float min, float max )
    {
        useDistanceLimits = true;
        minValue = min;
        maxValue = max;
    }
    bool useDistanceLimits = false; ///< out of limits intersections will be set to non-valid
    bool allowNegativeValues = false; ///< allows to find intersections in backward to direction vector with negative values
    float minValue = 0.f; ///< Using of this parameter depends on useDistanceLimits
    float maxValue = 0.f; ///< Using of this parameter depends on useDistanceLimits

    Vector2i resolution; ///< resolution of distance map

private:
    std::pair<Vector3f,Vector2f> orgSizeFromMeshPart_( const Matrix3f& rotation, const MeshPart& mp, bool presiceBox ) const;
    void initFromSize_( const AffineXf3f& worldOrientation, const Vector2i& resolution, const Vector2f& size );
};

struct DistanceMapToWorld;

/// Structure with parameters to generate DistanceMap by Contours
struct ContourToDistanceMapParams {
    /// Default ctor, make sure to fill all fields manually
    ContourToDistanceMapParams() = default;

    /// Ctor, calculating pixelSize by areaSize & dmapSize
    MRMESH_API ContourToDistanceMapParams( const Vector2i& resolution, const Vector2f& oriPoint,
        const Vector2f& areaSize, bool withSign = false );

    /// Ctor, calculating pixelSize & oriPoint by box parameters
    MRMESH_API ContourToDistanceMapParams( const Vector2i& resolution, const Box2f& box, bool withSign = false );

    /// Ctor, calculating pixelSize & oriPoint by contours box + offset
    MRMESH_API ContourToDistanceMapParams( const Vector2i& resolution, const Contours2f& contours,
        float offset, bool withSign = false );

    /// Ctor, calculating resolution & oriPoint by contours box + offset
    MRMESH_API ContourToDistanceMapParams( float pixelSize, const Contours2f& contours,
        float offset, bool withSign = false );

    MRMESH_API explicit ContourToDistanceMapParams( const DistanceMapToWorld& toWorld );
    /// get world 2d coordinate (respects origin point and pixel size)
    /// point - coordinate on distance map
    Vector2f toWorld( Vector2f point ) const
    {
        return orgPoint + Vector2f{ pixelSize.x * point.x, pixelSize.y * point.y };
    }

    Vector2f pixelSize{ 1.F, 1.F }; ///< pixel size
    Vector2i resolution{ 1, 1 }; ///< distance map size
    Vector2f orgPoint{ 0.F, 0.F }; ///< coordinates of origin area corner
    bool withSign{ false }; ///< allows calculate negative values of distance (inside closed and correctly oriented (CW) contours)
};

/// This structure store data to transform distance map to world coordinates
struct DistanceMapToWorld
{
    /// Default ctor init all fields with zeros, make sure to fill them manually
    DistanceMapToWorld() = default;

    /// Init fields by `MeshToDistanceMapParams` struct
    MRMESH_API DistanceMapToWorld( const MeshToDistanceMapParams& params );

    /// Init fields by `ContourToDistanceMapParams` struct
    MRMESH_API DistanceMapToWorld( const ContourToDistanceMapParams& params );
    

    /// get world coordinate by depth map info
    /// x - float X coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
    /// y - float Y coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
    /// float depth value (value in distance map, represent depth in world)
    Vector3f toWorld( float x, float y, float depth ) const
    {
        return orgPoint + x * pixelXVec + y * pixelYVec + depth * direction;
    }


    /// world coordinates of distance map origin corner
    Vector3f orgPoint;
    /// vector in world space of pixel x positive direction
    /// length is equal to pixel size
    /// \note typically it should be orthogonal to `pixelYVec`
    Vector3f pixelXVec;
    /// vector in world space of pixel y positive direction
    /// length is equal to pixel size
    /// \note typically it should be orthogonal to `pixelXVec`
    Vector3f pixelYVec;
    /// vector of depth direction
    /// \note typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane
    Vector3f direction;
};

/// \}

} // namespace MR
