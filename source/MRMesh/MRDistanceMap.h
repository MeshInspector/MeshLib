#pragma once

#include "MRRectIndexer.h"
#include "MRBitSet.h"
#include "MRAffineXf3.h"
#include "MRDistanceMapParams.h"
#include "MRPolyline2.h"
#include <tl/expected.hpp>
#include <filesystem>
#include <vector>

namespace MR
{

// this class allows to store distances from the plane in particular pixels
// validVerts keeps only pixels with mesh-intersecting rays from them
class DistanceMap : public RectIndexer
{
public:
    DistanceMap() = default;

    // Preferable constructor with resolution arguments
    // Access by the index (i) is equal to (y*resX + x)
    MRMESH_API DistanceMap( size_t resX, size_t resY );

    // make from 2d array
    MRMESH_API DistanceMap( const MR::Matrix<float>& m );

    // checks if X,Y element is valid
    MRMESH_API bool isValid( size_t x, size_t y ) const;
    // checks if index element is valid
    MRMESH_API bool isValid( size_t i ) const;

    // returns value in (X,Y) element, returns nullopt if not valid
    MRMESH_API std::optional<float> get( size_t x, size_t y ) const;
    // returns value of index element, returns nullopt if not valid
    MRMESH_API std::optional<float> get( size_t i ) const;
    // returns value in (X,Y) element without check on valid
    // use this only if you sure that distance map has no invalid values or for serialization
    float& getValue( size_t x, size_t y )       { return data_[ toIndex( { int( x ), int( y ) } ) ]; }
    float  getValue( size_t x, size_t y ) const { return data_[ toIndex( { int( x ), int( y ) } ) ]; }
    float& getValue( size_t i )       { return data_[i]; }
    float  getValue( size_t i ) const { return data_[i]; }


    // finds interpolated value.
    // https://en.wikipedia.org/wiki/Bilinear_interpolation
    // X,Y should be in resolution range [0;resX][0;resY].
    // getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
    // see https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
    // all 4 elements around this point should be valid, returns nullopt if at least one is not valid
    MRMESH_API std::optional<float> getInterpolated( float x, float y ) const;

    // finds 3d coordinates of the Point on the model surface for the (x,y) pixel
    // Use the same params with distance map creation
    MRMESH_API std::optional<Vector3f> unproject( size_t x, size_t y, const DistanceMapToWorld& toWorldStruct ) const;

    // finds 3d coordinates of the Point on the model surface for the (x,y) interpolated value
    // X,Y should be in resolution range [0;resX][0;resY].
    // getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
    // see https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
    // all 4 elements around this point should be valid, returns nullopt if at least one is not valid
    MRMESH_API std::optional<Vector3f> unprojectInterpolated( float x, float y, const DistanceMapToWorld& toWorldStruct ) const;

    // replaces every valid element in the map with its negative value
    MRMESH_API void negate();

    // boolean operators
    // returns new Distance Map with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API DistanceMap max( const DistanceMap& rhs) const;
    // replaces values with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& mergeMax( const DistanceMap& rhs );
    // returns new Distance Map with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API DistanceMap min( const DistanceMap& rhs ) const;
    // replaces values with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& mergeMin ( const DistanceMap& rhs );
    // returns new Distance Map with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API DistanceMap operator- ( const DistanceMap& rhs ) const;
    // replaces values with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
    MRMESH_API const DistanceMap& operator-= ( const DistanceMap& rhs );

    // sets value in (X,Y) element
    MRMESH_API void set( size_t x, size_t y, float val );
    // sets value in index element
    MRMESH_API void set( size_t i, float val );
    // invalidates value in (X,Y) element
    MRMESH_API void unset( size_t x, size_t y );
    // invalidates value in index element
    MRMESH_API void unset( size_t i );

    // invalidates all elements
    MRMESH_API void invalidateAll();
    // clears data, sets resolutions to zero
    MRMESH_API void clear();

    // returns new derivatives map without directions
    MRMESH_API DistanceMap getDerivativeMap() const;
    // returns new derivative maps with X and Y axes direction
    MRMESH_API std::pair< DistanceMap, DistanceMap > getXYDerivativeMaps() const;

    // computes single derivative map from XY spaces combined. Returns local maximums then
    MRMESH_API std::vector< std::pair<size_t, size_t> > getLocalMaximums() const;

    //returns X resolution
    size_t resX() const { return dims_.x; }
    //returns Y resolution
    size_t resY() const { return dims_.y; }

    //returns the number of pixels
    size_t numPoints() const { return size(); }

    // finds minimum and maximum values
    // returns min_float and max_float if all values are invalid
    MRMESH_API std::pair<float, float> getMinMaxValues() const;
    // finds minimum value X,Y
    // returns [-1.-1] if all values are invalid
    MRMESH_API std::pair<size_t, size_t> getMinIndex() const;
    // finds maximum value X,Y
    // returns [-1.-1] if all values are invalid
    MRMESH_API std::pair<size_t, size_t> getMaxIndex() const;

    // returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return data_.capacity() * sizeof( data_[0] ); }

private:
    std::vector<float> data_;
};

// fill another distance map pair with gradients across X and Y axes of the argument map
MRMESH_API DistanceMap combineXYderivativeMaps( std::pair<DistanceMap, DistanceMap> XYderivativeMaps );

// computes distance map for presented projection parameters
// use MeshToDistanceMapParams constructor instead of overloads of this function
// MeshPart - input 3d model
// general call. You could customize params manually
MRMESH_API DistanceMap computeDistanceMap( const MeshPart& mp, const MeshToDistanceMapParams& params );
MRMESH_API DistanceMap computeDistanceMapD( const MeshPart& mp, const MeshToDistanceMapParams& params );

// Structure with parameters for optional offset in `distanceMapFromContours` function
struct ContoursDistanceMapOffset
{
    // offset values for each undirected edge of given polyline
    const Vector<float, UndirectedEdgeId>& perEdgeOffset;
    enum class OffsetType
    {
        Normal, // distance map from given polyline with values offset
        Shell   // distance map from shell of given polyline (perEdgeOffset should not have negative values )
    } type{ OffsetType::Shell };
};

// Computes distance of 2d contours according ContourToDistanceMapParams
// offsetParameters - optional input offset for each edges of polyline, find more on `ContoursDistanceMapOffset` structure description
// outClosestEdges - optional output vector of closest polyline edge per each pixel of distance map
// !note that polyline topology should be consistently oriented
MRMESH_API DistanceMap distanceMapFromContours( const Polyline2& contours, const ContourToDistanceMapParams& params,
    const ContoursDistanceMapOffset* offsetParameters = nullptr,
    std::vector<UndirectedEdgeId>* outClosestEdges = nullptr,
    const PixelBitSet * region = nullptr ); //< if pointer is valid, then only these pixels will be filled

// Makes distance map and filter out pixels with large (>threshold) distance between closest points on contour in neighbor pixels
// Converts such points back in 3d space and return
// !note that polyline topology should be consistently oriented
MRMESH_API std::vector<Vector3f> edgePointsFromContours( const Polyline2& contour, float pixelSize, float threshold );

// converts distance map to 2d iso-lines:
// iso-lines are created in space DistanceMap ( plane OXY with pixelSize = (1, 1) )
MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float isoValue );

// iso-lines are created in real space ( plane OXY with parameters according ContourToDistanceMapParams )
MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const ContourToDistanceMapParams& params, float isoValue );

// iso-lines are created in real space
// ( contours plane with parameters according DistanceMapToWorld )
// return pair contours in OXY & transformation from plane OXY to real contours plane
MRMESH_API std::pair<Polyline2, AffineXf3f> distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const DistanceMapToWorld& params, float isoValue, bool useDepth = false );
MRMESH_API Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float pixelSize, float isoValue );

// computes the union of the shapes bounded by input 2d contours;
// returns the boundary of the union;
// input contours must be closed within the area of distance map and be consistently oriented (counterclockwise, that is leaving the bounded shapes from the left);
// the value of params.withSign must be true (checked with assert() inside the function)
// !note that polyline topology should be consistently oriented
MRMESH_API Polyline2 contourUnion( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0 );

// computes the intersection of the shapes bounded by input 2d contours;
// returns the boundary of the intersection;
// input contours must be closed within the area of distance map and be consistently oriented (counterclockwise, that is leaving the bounded shapes from the left);
// the value of params.withSign must be true (checked with assert() inside the function)
// !note that polyline topology should be consistently oriented
MRMESH_API Polyline2 contourIntersection( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

// computes the difference between the shapes bounded by contoursA and the shapes bounded by contoursB;
// returns the boundary of the difference;
// input contours must be closed within the area of distance map and be consistently oriented (counterclockwise, that is leaving the bounded shapes from the left);
// the value of params.withSign must be true (checked with assert() inside the function)
// !note that polyline topology should be consistently oriented
MRMESH_API Polyline2 contourSubtract( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

// converts distance map back to the mesh fragment with presented params
MRMESH_API Mesh distanceMapToMesh( const DistanceMap& distMap, const DistanceMapToWorld& toWorldStruct );

// saves distance map to image in scales of gray:
// far: 0.3 (dark-gray)
// close: 1.0 (white)
MRMESH_API tl::expected<void, std::string> saveDistanceMapToImage( const DistanceMap& distMap, const std::filesystem::path& filename );


// !!! DEPRECATED !!!

// Deprecated, use `distanceMapFromContours( const Polyline2& contours, const ContourToDistanceMapParams& params )`
[[deprecated]] MRMESH_API DistanceMap distanceMapFromContours( const Contours2f& contours, const ContourToDistanceMapParams& params );

// Deprecated, use `edgePointsFromContours( const Polyline2& contour, float pixelSize, float threshold )`
[[deprecated]] MRMESH_API std::vector<Vector3f> edgePointsFromContours( const Contours2f& contour, float pixelSize, float threshold );

// Deprecated, use `distanceMapTo2DIsoPolyline`
[[deprecated]] MRMESH_API Contours2f distanceMapTo2DIsoLine( const DistanceMap& distMap, float isoValue );

// Deprecated, use `distanceMapTo2DIsoPolyline`
[[deprecated]] MRMESH_API Contours2f distanceMapTo2DIsoLine( const DistanceMap& distMap,
    const ContourToDistanceMapParams& params, float isoValue );

// Deprecated, use `distanceMapTo2DIsoPolyline`
[[deprecated]] MRMESH_API std::pair<Contours2f, AffineXf3f> distanceMapTo2DIsoLine( const DistanceMap& distMap,
    const DistanceMapToWorld& params, float isoValue, bool useDepth = false );

// Deprecated, use `distanceMapTo2DIsoPolyline`
[[deprecated]] MRMESH_API Contours2f distanceMapTo2DIsoLine( const DistanceMap& distMap, float pixelSize, float isoValue );

// Deprecated, use `Polyline2 contourUnion(...)`
[[deprecated]] MRMESH_API Contours2f contourUnion( const Contours2f& contoursA, const Contours2f& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0 );

// Deprecated, use `Polyline2 contourIntersection(...)`
[[deprecated]] MRMESH_API Contours2f contourIntersection( const Contours2f& contoursA, const Contours2f& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

// Deprecated, use `Polyline2 contourSubtract(...)`
[[deprecated]] MRMESH_API Contours2f contourSubtract( const Contours2f& contoursA, const Contours2f& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside = 0.f );

} //namespace MR
