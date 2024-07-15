#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRBox.h"
#include "MRVector.h"
#include "MRMeshEigen.h"

namespace MR
{
// Class for deforming mesh using trilinear interpolation
class FreeFormDeformer
{
public:
    // Only set mesh ref
    MRMESH_API FreeFormDeformer( VertCoords& coords, const VertBitSet& valid );
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
    void setAllRefGridPositions( const std::vector<Vector3f>& refPoints ) { refPointsGrid_ = refPoints; }
    const Vector3i& getResolution() const { return resolution_; }
private:
    VertCoords& coords_;
    const VertBitSet& validPoints_;
    std::vector<Vector3f> refPointsGrid_;
    Box3f initialBox_;
    VertCoords normedCoords_;
    Vector3i resolution_;

    Vector3f applyToNormedPoint_( const Vector3f& normedPoint, std::vector<Vector3f>& xPlaneCache, std::vector<Vector3f>& yLineCache, std::vector<Vector3f>& tempPoints ) const;
};

/// Class to accumulate source and target points for free form alignment
/// Calculates best Free Form transform to fit given source->target deformation
/// origin ref grid as box corners ( resolution parameter specifies how to divide box )
class MRMESH_CLASS FreeFormBestFit
{
public:
    /// initialize the class, compute cached values and reserve space for matrices
    MRMESH_API FreeFormBestFit( const Box3d& box, const Vector3i& resolution = Vector3i::diagonal( 2 ) );

    /// add pair of source and target point to accumulator
    MRMESH_API void addPair( const Vector3d& src, const Vector3d& tgt, double w = 1.0 );
    void addPair( const Vector3f& src, const Vector3f& tgt, float w = 1.0f ) { addPair( Vector3d( src ), Vector3d( tgt ), double( w ) ); }

    /// adds other instance of FreeFormBestFit if it has same ref grid
    MRMESH_API void addOther( const FreeFormBestFit& other );

    /// stabilizer adds additional weights to keep result grid closer to origins
    /// recommended values (0;1], but it can be higher
    void setStabilizer( double stabilizer ) { stabilizer_ = stabilizer; }
    double getStabilizer() const { return stabilizer_; }

    /// finds best grid points positions to align source points to target points
    [[nodiscard]] MRMESH_API std::vector<Vector3f> findBestDeformationReferenceGrid();
private:
    Box3d box_;
    Vector3i resolution_;
    size_t resXY_{ 0 };
    size_t size_{ 0 };
    double sumWeight_{ 0.0 };

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> accumA_;
    Eigen::Matrix<double, Eigen::Dynamic, 3> accumB_;

    std::vector<int> pascalLineX_;
    std::vector<int> pascalLineY_;
    std::vector<int> pascalLineZ_;

    Vector3d reverseDiagonal_;

    double stabilizer_{ 0.1 };
    void stabilize_();
};

/// Returns positions of grid points in given box with given resolution 
MRMESH_API std::vector<Vector3f> makeFreeFormOriginGrid( const Box3f& box, const Vector3i& resolution );

// Calculates best Free Form transform to fit given source->target deformation
// origin ref grid as box corners ( resolution parameter specifies how to divide box )
// samplesToBox - if set used to transform source and target points to box space
// returns new positions of ref grid
MRMESH_API std::vector<Vector3f> findBestFreeformDeformation( const Box3f& box, const std::vector<Vector3f>& source, const std::vector<Vector3f>& target,
                                                              const Vector3i& resolution = Vector3i::diagonal( 2 ), const AffineXf3f* samplesToBox = nullptr );

}
