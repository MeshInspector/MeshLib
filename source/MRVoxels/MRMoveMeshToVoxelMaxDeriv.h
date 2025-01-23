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

/**
\defgroup SubvoxelMeshCorrection Subvoxel Mesh Correction
\ingroup VoxelGroup
\brief Precise automatic mesh correction or/and smoothing based on reference voxel (volume) data.

This group provides highlevel interface of the algorithm: \ref moveMeshToVoxelMaxDeriv configurable through
\ref MoveMeshToVoxelMaxDerivSettings and the low-level structure \ref MeshOnVoxelsT that allows creation of custom correction strategy.

\paragraph description Description of the algorithm.

-# **Input**: Mesh and VdbVolume with corresponding transforms. The objects must be aligned.
-# First step is to apply preparation smoothing with force specified by \ref MoveMeshToVoxelMaxDerivSettings::preparationSmoothForce.
   This step might be needed if the initial surface is very noisy.
-# Then the algorithm performs correction iteratively for the total number of iterations
   specified by \ref MoveMeshToVoxelMaxDerivSettings::iters.

    -# Each iteration consist of moving each vertex of the mesh towards its optimal position.
       More formally, let's denote, \f$ V = \left\{ ( v_i, n_i ) \right\} \f$ -- set of vertices and corresponding normals of the mesh,
       and \f$ F: \mathbb{R}^3 \rightarrow \mathbb{R} \f$ -- scalar field of volume -- for each position in the scene, it gives the density
       of the voxels in this position. For points with non-integer coordinates, the density is interpolated linearly (for details see \ref MeshOnVoxelsT).

       The optimal position \f$ p_i \f$ of the vertex \f$ v_i \f$ is given by fitting the polynomial \f$ Q_i \f$ of degree \ref MoveMeshToVoxelMaxDerivSettings::degree
       to the set of points \f$ \left\{ F\left( v_i + n_i \cdot k \cdot \text{voxel size} \right) \mid k \in -l \dots l \right\} \f$ where \f$ l = \frac{\text{sample points}}{2} \f$
       (see \ref MoveMeshToVoxelMaxDerivSettings::samplePoints); and finding the minimum of the derivative of this polynomial.

       Since the \ref MoveMeshToVoxelMaxDerivSettings::degree is limited to \f$ 6 \f$, we know that the degree of \f$ \frac{d^2 Q_i}{dx^2} \f$
       is limited to \f$ 4 \f$ and thus the equation \f$ \frac{d^2 Q_i}{dx^2} \left( x \right) = 0 \f$ has analytical solution and
       it gives us the extrema of the derivative.
       Comparing the evaluations of the polynomial at extrema and on the border of the interval we obtain the optimal position \f$ p_i \f$.
       To sum up:
       \f{align*}{
            &l = \frac{\text{sample points}}{2} \\
            &N_i \leftarrow \left\{ v_i + n_i \cdot k \cdot \text{voxel size} \mid k = -l \dots l \right\} \\
            &\overline{N_i} \leftarrow \left\{ v_i + n_i \cdot x \cdot \text{voxel size} \mid x \in \left[-l; l\right] \right\} \\
            &Q_i \leftarrow \text{fit polynomial from } \left\{ \left( p, F(p) \right) \mid p \in N_i \right\} \\
            &p_i \leftarrow \min_{x \in \overline{N_i}} \frac{dQ_i}{dx}
       \f}
       Note that we are finding the minimum of \f$ \frac{dQ_i}{dx} \f$ analytically over the real interval \f$ \overline{N_i} \f$ and not over \f$ N_i \f$.
       This is exactly where the subvoxel precision comes from.

    -# However, directly using the optimal position has proven to be error-prone, as both voxels and mesh could contain noise.
       Therefore, we use the following heuristic: instead of moving the vertex \f$ v_i \f$ to position \f$ p_i \f$, we construct a field of shifts
       with domain on the mesh vertices \f$ S : v_i \mapsto \text{clamp}\left( p_i - v_i \right) \f$ and smooth it with the force specified by
       \ref MoveMeshToVoxelMaxDerivSettings::intermediateSmoothForce for 15 iterations. Then we update the vertices according to the smoothed field:
       \f[
            v_i \leftarrow v_i + S_{smooth}\left( v_i \right)
       \f]
        and proceed to the next interation.
-# The algorithm modifies the mesh inplace and returns the set of vertices that were moved during the correction.

A nice visualization of \a why it works could be found in MeshInspector's plugin "Voxels Inspector". Just click on any vertex and you will see
the values of \f$ F, Q_i \f$ on the interval \f$ \overline{N_i} \f$, as well as points of \f$ N_i \f$.

\paragraph usage Usage

The algorithm can be used in different cases related to the CT-scanning workflows:
- To increase precision of the mesh retrieved by marching cubes meshing from a noisy scan.
- To smooth a noisy mesh taking into account extra information from voxels. In this case, even if both mesh and voxels are noisy, the result
  is expected to be better than after a simple smoothing. For this use-case, you might need to increase the force of both preparation and intermidiate
  smoothings.

*/


/// \ingroup SubvoxelMeshCorrection
struct MoveMeshToVoxelMaxDerivSettings
{
    /// number of iterations. Each iteration moves vertex only slightly and smooths the vector field of shifts.
    int iters = 30;

    /// number of points to sample for each vertex. Samples are used to get the picewice-linear function of density and
    /// estimate the derivative based on it
    int samplePoints = 6;

    /// degree of the polynomial used to fit sampled points. Must be in range [3; 6]
    int degree = 3;

    /// for each iteration, if target position of the vertex is greater than this threshold, it is disregarded.
    /// For small degrees, this value should be small, for large degrees it may be larger.
    /// Measured in number of voxels.
    float outlierThreshold = 1.f;

    /// force of the smoothing (relaxation) of vector field of shifts on each iteration
    float intermediateSmoothForce = 0.3f;

    /// force of initial smoothing of vertices, before applying the algorithm
    float preparationSmoothForce = 0.1f;
};


/// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
/// of voxels.
/// \ingroup SubvoxelMeshCorrection
/// @return Vertices that were moved by the algorithm
template <typename VolumeType = VdbVolume>
MRVOXELS_API Expected<VertBitSet> moveMeshToVoxelMaxDeriv(
        Mesh& mesh, const AffineXf3f& meshXf,
        const VolumeType& volume, const AffineXf3f& volumeXf,
        const MoveMeshToVoxelMaxDerivSettings& settings,
        ProgressCallback callback = {}
);

extern template MRVOXELS_API Expected<VertBitSet> moveMeshToVoxelMaxDeriv<VdbVolume>( Mesh& mesh, const AffineXf3f& meshXf,
    const VdbVolume& volume, const AffineXf3f& volumeXf,
    const MoveMeshToVoxelMaxDerivSettings& settings,
    ProgressCallback callback );
extern template MRVOXELS_API Expected<VertBitSet> moveMeshToVoxelMaxDeriv<SimpleVolumeMinMax>( Mesh& mesh, const AffineXf3f& meshXf,
    const SimpleVolumeMinMax& volume, const AffineXf3f& volumeXf,
    const MoveMeshToVoxelMaxDerivSettings& settings,
    ProgressCallback callback );
extern template MRVOXELS_API Expected<VertBitSet> moveMeshToVoxelMaxDeriv<FunctionVolume>( Mesh& mesh, const AffineXf3f& meshXf,
    const FunctionVolume& volume, const AffineXf3f& volumeXf,
    const MoveMeshToVoxelMaxDerivSettings& settings,
    ProgressCallback callback );


/// Helper class to organize mesh and voxels volume access and build point sequences
/// \note this class is not thread-safe but accessing same volume from different instances is ok
/// \ingroup SubvoxelMeshCorrection
template <typename MeshType, typename VolumeType>
class MeshOnVoxelsT
{
public:
    MRVOXELS_API MeshOnVoxelsT( MeshType& mesh, const AffineXf3f& meshXf, const VolumeType& volume, const AffineXf3f& volumeXf );
    MRVOXELS_API MeshOnVoxelsT( const MeshOnVoxelsT& other );

    // Access to base data
    MRVOXELS_API MeshType& mesh() const;

    MRVOXELS_API const VolumeType& volume() const;


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
    const VolumeType& volume_;
    float voxelSize_;
    VoxelsVolumeAccessor<VolumeType> accessor_;
    VoxelsVolumeInterpolatedAccessor<VoxelsVolumeAccessor<VolumeType>> interpolator_;
    AffineXf3f xf_, xfInv_;
    Matrix3f xfNormal_;
    bool noXf_; // Xf is unit or translation
    int numVerts_;
};


using MeshOnVoxelsVdb = MeshOnVoxelsT<Mesh, VdbVolume>;
using MeshOnVoxelsVdbC = MeshOnVoxelsT<const Mesh, VdbVolume>;

using MeshOnVoxelsSimple = MeshOnVoxelsT<Mesh, SimpleVolumeMinMax>;
using MeshOnVoxelsSimpleC = MeshOnVoxelsT<const Mesh, SimpleVolumeMinMax>;

using MeshOnVoxelsFunction = MeshOnVoxelsT<Mesh, FunctionVolume>;
using MeshOnVoxelsFunctionC = MeshOnVoxelsT<const Mesh, FunctionVolume>;
}
