#pragma once

#include "MRVoxelsFwd.h"

#include "MRVoxelsVolumeAccess.h"
#include "MRVolumeInterpolation.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRBestFitParabola.h"
#include "MRMesh/MRBestFitPolynomial.h"


namespace MR
{

struct MoveMeshToVoxelMaxDerivSettings
{
    /// number of iterations. Each iteration moves vertex only slightly and smooths the vector field of shifts.
    int iters = 30;

    /// number of points to sample for each vertex. Samples are used to get the picewice-linear function of density and
    /// estimate the derivative based on it
    int samplePoints = 6;

    /// degree of the polynomial used to fit sampled points. Must be in range [3; 6]
    int degree = 3;

    /// for each iteration, if target position of the vertex is greater than this threshold, it is disregarded
    /// for small degrees, this value should be small, for large degrees it may be larger
    /// measured in number of voxels
    float outlierThreshold = 1.f;

    /// force of the smoothing (relaxation) of vector field of shifts on each iteration
    float intermediateSmoothForce = 0.3f;

    /// force of initial smoothing of vertices, before applying the algorithm
    float preparationSmoothForce = 0.1f;
};


/// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
/// of voxels.
/// @return Vertices that were moved by the algorithm
MRVOXELS_API Expected<VertBitSet> moveMeshToVoxelMaxDeriv(
        Mesh& mesh, const AffineXf3f& meshXf,
        const VdbVolume& volume, const AffineXf3f& volumeXf,
        const MoveMeshToVoxelMaxDerivSettings& settings,
        ProgressCallback callback = {}
    );



// Helper class to organize mesh and voxels volume access and build point sequences
// Note: this class is not thread-safe but accessing same volume from different instances is ok
template <typename MeshType>
class MeshOnVoxelsT
{
public:
    MRVOXELS_API MeshOnVoxelsT( MeshType& mesh, const AffineXf3f& meshXf, const VdbVolume& volume, const AffineXf3f& volumeXf );
    MRVOXELS_API MeshOnVoxelsT( const MeshOnVoxelsT& other );

    // Access to base data
    MRVOXELS_API MeshType& mesh() const;

    MRVOXELS_API const VdbVolume& volume() const;


    // Cached number of valid vertices
    MRVOXELS_API int numVerts() const;

    // Voxel size as scalar
    MRVOXELS_API float voxelSize() const;


    // Transformation mesh to volume
    // All points are in voxels volume space, unless otherwise is implied
    MRVOXELS_API AffineXf3f xf() const;

    MRVOXELS_API Vector3f xf( const Vector3f& pt ) const;

    MRVOXELS_API AffineXf3f xfInv() const;

    MRVOXELS_API Vector3f xfInv( const Vector3f &pt ) const;


    // Vertex position
    MRVOXELS_API Vector3f point( VertId v ) const;

    // Volume value
    MRVOXELS_API float getValue( const Vector3f& pos ) const;

    // Get offset vector (mesh normal for a vertex with `voxelSize` length)
    MRVOXELS_API Vector3f getOffsetVector( VertId v ) const;

    // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
    // Pseudo-index is a signed number; for whole index, is is whole or half-whole
    MRVOXELS_API static float pseudoIndex( float index, int count );

    MRVOXELS_API static float pseudoIndex( int index, int count );

    MRVOXELS_API static float indexFromPseudoIndex( float pseudoIndex, int count );

    // Get row of points with `offset` stride
    MRVOXELS_API void getPoints( std::vector<Vector3f>& result, const Vector3f& pos, const Vector3f& offset ) const;

    // Get volume values for a row of points
    MRVOXELS_API void getValues( std::vector<float>& result, const Vector3f& pos, const Vector3f& offset ) const;

    // Get derivatives from result of `getValues`
    MRVOXELS_API static void getDerivatives( std::vector<float>& result, const std::vector<float>& values );

    // Get best fit parabola in pseudo-index space for a zero-centered array
    static Parabolaf getBestParabola( auto begin, auto end )
    {
        BestFitParabola<float> bestFitParabola;
        auto size = std::distance( begin, end );
        for ( auto it = begin; it != end; ++it )
            bestFitParabola.addPoint( pseudoIndex( int( it - begin ), int( size ) ), *it );
        return bestFitParabola.getBestParabola();
    }

    template <size_t degree>
    static Polynomialf<degree> getBestPolynomial( const std::vector<float>& values )
    {
        BestFitPolynomial<double, degree> bestFit( 0.f );
        for ( size_t i = 0; i < values.size(); ++i )
            bestFit.addPoint( pseudoIndex( int( i ), int( values.size() ) ), values[i] );
        auto poly = bestFit.getBestPolynomial().template cast<float>();
        return poly;
    }

    MRVOXELS_API static PolynomialWrapperf getBestPolynomial( const std::vector<float>& values, size_t degree );

private:
    MeshType& mesh_;
    const VdbVolume& volume_;
    float voxelSize_;
    VoxelsVolumeAccessor<VdbVolume> accessor_;
    VoxelsVolumeInterpolatedAccessor<VoxelsVolumeAccessor<VdbVolume>> interpolator_;
    AffineXf3f xf_, xfInv_;
    Matrix3f xfNormal_;
    bool noXf_; // Xf is unit or translation
    int numVerts_;
};


using MeshOnVoxels = MeshOnVoxelsT<Mesh>;
using MeshOnVoxelsC = MeshOnVoxelsT<const Mesh>;


}
