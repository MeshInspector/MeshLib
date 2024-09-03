#pragma once
#include "MRBox.h"
#include "MRAffineXf3.h"
#include "MRBestFit.h"
#include "MRPointCloud.h"

namespace MR
{
/// Structure to hold and work with dense box
/// \details Scalar operations that are not provided in this struct can be called via `box()`
/// For example `box().size()`, `box().diagonal()` or `box().volume()`
/// Non const operations are not allowed for dense box because it can spoil density
/// \ingroup MathGroup
struct DenseBox
{
    /// Include given points into this dense box
    MRMESH_API DenseBox( const std::vector<Vector3f>& points, const AffineXf3f* xf = nullptr );
    /// Include given weighed points into this dense box
    MRMESH_API DenseBox( const std::vector<Vector3f>& points, const std::vector<float>& weights, const AffineXf3f* xf = nullptr );
    /// Include mesh part into this dense box
    MRMESH_API DenseBox( const MeshPart& meshPart, const AffineXf3f* xf = nullptr );
    /// Include point into this dense box
    MRMESH_API DenseBox( const PointCloud& points, const AffineXf3f* xf = nullptr );
    /// Include line into this dense box
    MRMESH_API DenseBox( const Polyline3& line, const AffineXf3f* xf = nullptr );
    
    /// returns center of dense box
    MRMESH_API Vector3f center() const;
    /// returns corner of dense box, each index value means: false - min, true - max
    /// example: {false, false, flase} - min point, {true, true, true} - max point
    MRMESH_API Vector3f corner( const Vector3b& index ) const;
    /// returns true if dense box contains given point
    MRMESH_API bool contains( const Vector3f& pt ) const;

    // Access members

    /// return box in its space
    const Box3f& box() const { return box_; }
    /// transform box space to world space 
    const AffineXf3f& basisXf() const { return basisXf_; }
    /// transform world space to box space
    const AffineXf3f& basisXfInv() const { return basisXfInv_; }

private:
    /// Include given points into this dense box
    void init_( const std::vector<Vector3f>& points, const std::vector<float>* weights = nullptr, const AffineXf3f* xf = nullptr );
    /// Include mesh part into this dense box
    void init_( const MeshPart& meshPart, const AffineXf3f* xf = nullptr );
    /// Include point into this dense box
    void init_( const PointCloud& points, const AffineXf3f* xf = nullptr );
    /// Include line into this dense box
    void init_( const Polyline3& line, const AffineXf3f* xf = nullptr );

    Box3f box_;
    AffineXf3f basisXf_;
    AffineXf3f basisXfInv_;
};

}