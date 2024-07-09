#pragma once

#include "MRMeshFwd.h"
#include "MRVoxelsVolumeAccess.h"
#include "MRVolumeInterpolation.h"
#include "MRMatrix3.h"
#include "MRAffineXf.h"
#include "MRTimer.h"
#include "MRBestFitParabola.h"


namespace MR
{

struct MoveMeshToVoxelMaxDerivSettings
{
    // number of iterations. Each iteration moves vertex only slightly and smooths the vector field of shifts.
    int iters = 30;

    // number of points to sample for each vertex. Samples are used to get the picewice-linear function of density and
    // estimate the derivative based on it
    int samplePoints = 6;

    // force of the smoothing (relaxation) of vector field of shifts on each iteration
    float intermediateSmoothForce = 0.3f;

    // force of initial smoothing of vertices, before applying the algorithm
    float preparationSmoothForce = 0.1f;
};


/// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
/// of voxels.
/// @return Vertices that were moved by the algorithm
MRMESH_API VertBitSet moveMeshToVoxelMaxDeriv(
        Mesh& mesh, const AffineXf3f& meshXf,
        const VdbVolume& volume, const AffineXf3f& volumeXf,
        const MoveMeshToVoxelMaxDerivSettings& settings,
        ProgressCallback callback = {}
    );



// Helper class to organize mesh and voxels volume access and build point sequences
template <typename MeshType>
class MRMESH_API MeshOnVoxelsT
{
public:
    MeshOnVoxelsT( MeshType& mesh, const AffineXf3f& meshXf, const VdbVolume& volume, const AffineXf3f& volumeXf );

    // Access to base data
    MeshType& mesh() const;
    const VdbVolume& volume() const;
    // Cached number of valid vertices
    int numVerts() const;
    // Voxel size as scalar
    float voxelSize() const;
    // Transformation mesh to volume
    // All points are in voxels volume space, unless otherwise is implied
    AffineXf3f xf() const;
    Vector3f xf( const Vector3f& pt ) const;
    AffineXf3f xfInv() const;
    Vector3f xfInv( const Vector3f &pt ) const;
    // Vertex position
    Vector3f point( VertId v ) const;
    // Volume value
    float getValue( const Vector3f& pos ) const;
    // Get offset vector (mesh normal for a vertex with `voxelSize` length)
    Vector3f getOffsetVector( VertId v ) const;
    // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
    // Pseudo-index is a signed number; for whole index, is is whole or half-whole
    static float pseudoIndex( float index, int count );
    static float pseudoIndex( int index, int count );
    static float indexFromPseudoIndex( float pseudoIndex, int count );
    // Get row of points with `offset` stride
    void getPoints( std::vector<Vector3f>& result, const Vector3f& pos, const Vector3f& offset ) const;
    // Get volume values for a row of points
    void getValues( std::vector<float>& result, const Vector3f& pos, const Vector3f& offset ) const;
    // Get derivatives from result of `getValues`
    static void getDerivatives( std::vector<float>& result, const std::vector<float>& values );
    // Get best fit parabola in pseudo-index space for a zero-centered array
    static Parabolaf getBestParabola( auto begin, auto end )
    {
        BestFitParabola<float> bestFitParabola;
        auto size = std::distance( begin, end );
        for ( auto it = begin; it != end; ++it )
            bestFitParabola.addPoint( pseudoIndex( int( it - begin ), int( size ) ), *it );
        return bestFitParabola.getBestParabola();
    }

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