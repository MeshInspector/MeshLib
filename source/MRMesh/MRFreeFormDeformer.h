#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRBox.h"

namespace MR
{
// Class for deforming mesh using trilinear interpolation
class FreeFormDeformer
{
public:
    // Only set mesh ref
    MRMESH_API FreeFormDeformer( Mesh& mesh );
    // Parallel calculates all points normed positions
    // sets ref grid by initialBox, if initialBox is invalid use mesh bounding box instead 
    MRMESH_API void init( const Vector3i& resolution = Vector3i::diagonal( 2 ), const Box3f& initialBox = Box3f() );
    // Updates ref grid point position
    MRMESH_API void setRefGridPointPosition( const Vector3i& coordOfPointInGrid, const Vector3f& newPos );
    // Gets ref grid point position
    MRMESH_API const Vector3f& getRefGridPointPosition( const Vector3i& coordOfPointInGrid ) const;
    // Parallel apply updated grid to all mesh points
    // ensure updating render object after using it
    MRMESH_API void apply() const;
    // Apply updated grid to given point
    MRMESH_API Vector3f applySinglePoint( const Vector3f& point ) const;
    // Get one dimension index by grid coord
    MRMESH_API int getIndex( const Vector3i& coordOfPointInGrid ) const;
    // Get grid coord by index
    MRMESH_API Vector3i getCoord( int index ) const;

    const std::vector<Vector3f>& getAllRefGridPositions() const { return refPointsGrid_; }
    const void setAllRefGridPositions( const std::vector<Vector3f>& refPoints ) { refPointsGrid_ = refPoints; }
    const Vector3i& getResolution() const { return resolution_; }
private:
    Mesh& mesh_;
    std::vector<Vector3f> refPointsGrid_;
    Box3f initialBox_;
    std::vector<Vector3f> meshPointsNormedPoses_;
    Vector3i resolution_;

    Vector3f applyToNormedPoint_( const Vector3f& normedPoint, std::vector<Vector3f>& xPlaneCache, std::vector<Vector3f>& yLineCache ) const;
};

// Calculates best Free Form transform to fit given source->target deformation
// origin ref grid as box corners ( resolution parameter specifies how to divide box )
// returns new positions of ref grid
MRMESH_API std::vector<Vector3f> findBestFreeformDeformation( const Box3f& box, const std::vector<Vector3f>& source, const std::vector<Vector3f>& target,
                                                              const Vector3i& resolution = Vector3i::diagonal( 2 ) );

}
