#pragma once

#include "MRRectIndexer.h"
#include "MRBitSet.h"
#include "MRAffineXf3.h"
#include "MRDistanceMapParams.h"
#include "MRPolyline.h"
#include "MRHeapBytes.h"
#include "MRImage.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <vector>

namespace MR
{

/// \defgroup DistanceMapGroup Distance Map group
/// \{

/// this class allows to store distances from the plane in particular pixels
/// validVerts keeps only pixels with mesh-intersecting rays from them
class [[nodiscard]] DistanceMap : public RectIndexer
{
public:
    DistanceMap() = default;

    /// Preferable constructor with resolution arguments
    /// Access by the index (i) is equal to (y*resX + x)
    [[nodiscard]] MRMESH_API DistanceMap( size_t resX, size_t resY );

    /// make from 2d array
    [[nodiscard]] MRMESH_API DistanceMap( const MR::Matrix<float>& m );

    /// checks if X,Y element is valid
    [[nodiscard]] MRMESH_API bool isValid( size_t x, size_t y ) const;
    /// checks if index element is valid
    [[nodiscard]] MRMESH_API bool isValid( size_t i ) const;

    /// returns value in (X,Y) element, returns nullopt if not valid
    [[nodiscard]] MRMESH_API std::optional<float> get( size_t x, size_t y ) const;
    /// returns value of index element, returns nullopt if not valid
    [[nodiscard]] MRMESH_API std::optional<float> get( size_t i ) const;
    /// returns value in (X,Y) element without check on valid
    /// use this only if you sure that distance map has no invalid values or for serialization
    [[nodiscard]] float& getValue( size_t x, size_t y )
    {
        return data_[toIndex( { int( x ), int( y ) } )];
    }
    [[nodiscard]] float  getValue( size_t x, size_t y ) const
    {
        return data_[toIndex( { int( x ), int( y ) } )];
    }
    [[nodiscard]] float& getValue( size_t i )
    {
        return data_[i];
    }
    [[nodiscard]] float  getValue( size_t i ) const
    {
        return data_[i];
    }


    /**
     * \brief finds interpolated value.
     * \details \ref https:///en.wikipedia.org/wiki/Bilinear_interpolation
     * getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
     * see \ref https:///docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
     * all 4 elements around this point should be valid, returns nullopt if at least one is not valid
     * \param x,y should be in resolution range [0;resX][0;resY].
     */
    [[nodiscard]] MRMESH_API std::optional<float> getInterpolated( float x, float y ) const;

    /// finds 3d coordinates of the Point on the model surface for the (x,y) pixel
    /// Use the same params with distance map creation
    [[nodiscard]] MRMESH_API std::optional<Vector3f> unproject( size_t x, size_t y, const DistanceMapToWorld& toWorldStruct ) const;

    /**
     * \brief finds 3d coordinates of the Point on the model surface for the (x,y) interpolated value
     * \param x,y should be in resolution range [0;resX][0;resY].
     * \details getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
     * see \ref https: *docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
     * all 4 elements around this point should be valid, returns nullopt if at least one is not valid
     */
    [[nodiscard]] MRMESH_API std::optional<Vector3f> unprojectInterpolated( float x, float y, const DistanceMapToWorld& toWorldStruct ) const;

    /// replaces every valid element in the map with its negative value
    MRMESH_API void negate();

    /// boolean operators
    /// returns new Distance Map with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
    [[nodiscard]] MRMESH_API DistanceMap max( const DistanceMap& rhs ) const;
    /// replaces values with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& mergeMax( const DistanceMap& rhs );
    /// returns new Distance Map with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
    [[nodiscard]] MRMESH_API DistanceMap min( const DistanceMap& rhs ) const;
    /// replaces values with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& mergeMin( const DistanceMap& rhs );
    /// returns new Distance Map with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
    [[nodiscard]] MRMESH_API DistanceMap operator- ( const DistanceMap& rhs ) const;
    /// replaces values with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& operator-= ( const DistanceMap& rhs );

    /// sets value in (X,Y) element
    MRMESH_API void set( size_t x, size_t y, float val );
    /// sets value in index element
    MRMESH_API void set( size_t i, float val );
    /// invalidates value in (X,Y) element
    MRMESH_API void unset( size_t x, size_t y );
    /// invalidates value in index element
    MRMESH_API void unset( size_t i );

    /// invalidates all elements
    MRMESH_API void invalidateAll();
    /// clears data, sets resolutions to zero
    MRMESH_API void clear();

    /// returns new derivatives map without directions
    [[nodiscard]] MRMESH_API DistanceMap getDerivativeMap() const;
    /// returns new derivative maps with X and Y axes direction
    [[nodiscard]] MRMESH_API std::pair< DistanceMap, DistanceMap > getXYDerivativeMaps() const;

    /// computes single derivative map from XY spaces combined. Returns local maximums then
    [[nodiscard]] MRMESH_API std::vector< std::pair<size_t, size_t> > getLocalMaximums() const;

    ///returns X resolution
    [[nodiscard]] size_t resX() const
    {
        return dims_.x;
    }
    ///returns Y resolution
    [[nodiscard]] size_t resY() const
    {
        return dims_.y;
    }

    ///returns the number of pixels
    [[nodiscard]] size_t numPoints() const
    {
        return size();
    }

    /// finds minimum and maximum values
    /// returns min_float and max_float if all values are invalid
    [[nodiscard]] MRMESH_API std::pair<float, float> getMinMaxValues() const;
    /// finds minimum value X,Y
    /// returns [-1.-1] if all values are invalid
    [[nodiscard]] MRMESH_API std::pair<size_t, size_t> getMinIndex() const;
    /// finds maximum value X,Y
    /// returns [-1.-1] if all values are invalid
    [[nodiscard]] MRMESH_API std::pair<size_t, size_t> getMaxIndex() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const
    {
        return MR::heapBytes( data_ );
    }

private:
    std::vector<float> data_;
};

/// fill another distance map pair with gradients across X and Y axes of the argument map
[[nodiscard]] MRMESH_API DistanceMap combineXYderivativeMaps( std::pair<DistanceMap, DistanceMap> XYderivativeMaps );

/// computes distance (height) map for given projection parameters
/// using float-precision for finding ray-mesh intersections, which is faster but less reliable
[[nodiscard]] MRMESH_API DistanceMap computeDistanceMap( const MeshPart& mp, const MeshToDistanceMapParams& params, ProgressCallback cb = {} );

/// computes distance (height) map for given projection parameters
/// using double-precision for finding ray-mesh intersections, which is slower but more reliable
[[nodiscard]] MRMESH_API DistanceMap computeDistanceMapD( const MeshPart& mp, const MeshToDistanceMapParams& params, ProgressCallback cb = {} );

/// Structure with parameters for optional offset in `distanceMapFromContours` function
struct [[nodiscard]] ContoursDistanceMapOffset
{
    /// offset values for each undirected edge of given polyline
    const Vector<float, UndirectedEdgeId>& perEdgeOffset;
    enum class OffsetType
    {
        Normal, ///< distance map from given polyline with values offset
        Shell   ///< distance map from shell of given polyline (perEdgeOffset should not have negative values )
    } type{ OffsetType::Shell };
};

struct [[nodiscard]] ContoursDistanceMapOptions
{
    /// method to calculate sign
    enum SignedDetectionMethod
    {
        /// detect sign of distance based on closest contour's edge turn\n
        /// (recommended for good contours with no self-intersections)
        /// \note that polyline topology should be consistently oriented \n
        ContourOrientation,
        /// detect sign of distance based on number of ray intersections with contours\n
        /// (recommended for contours with self-intersections)
        WindingRule
    } signMethod{ ContourOrientation };
    /// optional input offset for each edges of polyline, find more on `ContoursDistanceMapOffset` structure description
    const ContoursDistanceMapOffset* offsetParameters{ nullptr };
    /// if pointer is valid, then only these pixels will be filled
    const PixelBitSet* region{ nullptr };
    /// optional output vector of closest polyline edge per each pixel of distance map
    std::vector<UndirectedEdgeId>* outClosestEdges{ nullptr };
};

/**
 * \brief Computes distance of 2d contours according ContourToDistanceMapParams
 * \param options - optional input and output options for distance map calculation, find more \ref ContoursDistanceMapOptions
 */
[[nodiscard]] MRMESH_API DistanceMap distanceMapFromContours( const Polyline2& contours, const ContourToDistanceMapParams& params,
    const ContoursDistanceMapOptions& options = {} );

/// Makes distance map and filter out pixels with large (>threshold) distance between closest points on contour in neighbor pixels
/// Converts such points back in 3d space and return
/// \note that polyline topology should be consistently oriented
[[nodiscard]] MRMESH_API std::vector<Vector3f> edgePointsFromContours( const Polyline2& contour, float pixelSize, float threshold );

/// converts distance map to 2d iso-lines:
/// iso-lines are created in space DistanceMap ( plane OXY with pixelSize = (1, 1) )
[[nodiscard]] MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float isoValue );

/// iso-lines are created in real space ( plane OXY with parameters according ContourToDistanceMapParams )
[[nodiscard]] MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const ContourToDistanceMapParams& params, float isoValue );

/// iso-lines are created in real space
/// ( contours plane with parameters according DistanceMapToWorld )
/// \return pair contours in OXY & transformation from plane OXY to real contours plane
[[nodiscard]] MRMESH_API std::pair<Polyline2, AffineXf3f> distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const DistanceMapToWorld& params, float isoValue, bool useDepth = false );
[[nodiscard]] MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float pixelSize, float isoValue );

/**
 * \brief computes the union of the shapes bounded by input 2d contours
 * \return the boundary of the union
 * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
 * the value of params.withSign must be true (checked with assert() inside the function)
 * \note that polyline topology should be consistently oriented
 */
[[nodiscard]] MRMESH_API Polyline2 contourUnion( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0 );

/**
 * \brief computes the intersection of the shapes bounded by input 2d contours
 * \return the boundary of the intersection
 * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
 * the value of params.withSign must be true (checked with assert() inside the function)
 * \note that polyline topology should be consistently oriented
 */
[[nodiscard]] MRMESH_API Polyline2 contourIntersection( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

/**
 * \brief computes the difference between the shapes bounded by contoursA and the shapes bounded by contoursB
 * \return the boundary of the difference
 * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
 * the value of params.withSign must be true (checked with assert() inside the function)
 * \note that polyline topology should be consistently oriented
 */
[[nodiscard]] MRMESH_API Polyline2 contourSubtract( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

/// converts distance map back to the mesh fragment with presented params
[[nodiscard]] MRMESH_API Mesh distanceMapToMesh( const DistanceMap& distMap, const DistanceMapToWorld& toWorldStruct );

/// saves distance map to monochrome image in scales of gray:
/// \param threshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)
/// minimum (close): 1.0 (white)
/// maximum (far): threshold
/// invalid (infinity): 0.0 (black)
MRMESH_API tl::expected<void, std::string> saveDistanceMapToImage( const DistanceMap& distMap, const std::filesystem::path& filename, float threshold = 1.f / 255 );

/// load distance map from monochrome image in scales of gray:
/// \param threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid
[[nodiscard]] MRMESH_API tl::expected<DistanceMap, std::string> convertImageToDistanceMap( const Image& image, float threshold = 1.f / 255 );

/// load distance map from monochrome image file
/// \param threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid
[[nodiscard]] MRMESH_API tl::expected<DistanceMap, std::string> loadDistanceMapFromImage( const std::filesystem::path& filename, float threshold = 1.f / 255 );

/// \}

} // namespace MR
