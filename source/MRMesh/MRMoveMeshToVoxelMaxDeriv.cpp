#include "MRMoveMeshToVoxelMaxDeriv.h"
#include "MRTimer.h"
#include "MRBestFitParabola.h"
#include "MRMesh.h"
#include "MRMeshRelax.hpp"
#include <MRMesh/MRParallelFor.h>


namespace MR
{



template <typename MeshType>
MeshOnVoxelsT<MeshType>::MeshOnVoxelsT( MeshType& mesh, const AffineXf3f& meshXf, const VdbVolume& volume, const AffineXf3f& volumeXf ):
    mesh_( mesh ), volume_( volume ),
    voxelSize_( std::min( { volume_.voxelSize.x, volume_.voxelSize.y, volume_.voxelSize.z } ) ),
    accessor_( volume_ ), interpolator_( volume_, accessor_ ),
    xf_( volumeXf.inverse() * meshXf ),
    xfInv_( xf_.inverse() ), xfNormal_( xfInv_.A.transposed() ),
    noXf_( xf_.A == Matrix3f() ),
    numVerts_( mesh_.topology.numValidVerts() )
{}

template <typename MeshType>
MeshOnVoxelsT<MeshType>::MeshOnVoxelsT( const MeshOnVoxelsT& other ) :
    mesh_( other.mesh_ ), volume_( other.volume_ ),
    voxelSize_( other.voxelSize_ ),
    accessor_( other.accessor_ ), interpolator_( volume_, accessor_ ), // Note: accessor copy is created here
    xf_( other.xf_ ), xfInv_( other.xfInv_ ), xfNormal_( other.xfNormal_ ), noXf_( other.noXf_ ),
    numVerts_( other.numVerts_ )
{}


template <typename MeshType>
MeshType& MeshOnVoxelsT<MeshType>::mesh() const
{ return mesh_; }

template <typename MeshType>
const VdbVolume& MeshOnVoxelsT<MeshType>::volume() const
{ return volume_; }

template <typename MeshType>
int MeshOnVoxelsT<MeshType>::numVerts() const
{ return numVerts_; }

template <typename MeshType>
float MeshOnVoxelsT<MeshType>::voxelSize() const
{ return voxelSize_; }

template <typename MeshType>
AffineXf3f MeshOnVoxelsT<MeshType>::xf() const
{ return xf_; }

template <typename MeshType>
Vector3f MeshOnVoxelsT<MeshType>::xf( const Vector3f& pt ) const
{ return noXf_ ? pt + xf_.b : xf_( pt ); }

template <typename MeshType>
AffineXf3f MeshOnVoxelsT<MeshType>::xfInv() const
{ return xfInv_; }

template <typename MeshType>
Vector3f MeshOnVoxelsT<MeshType>::xfInv( const Vector3f& pt ) const
{ return noXf_ ? pt + xfInv_.b : xfInv_( pt ); }

template <typename MeshType>
Vector3f MeshOnVoxelsT<MeshType>::point( VertId v ) const
{ return xf( mesh_.points[v] ); }

template <typename MeshType>
float MeshOnVoxelsT<MeshType>::getValue( const Vector3f& pos ) const
{ return interpolator_.get( pos ); }

template <typename MeshType>
Vector3f MeshOnVoxelsT<MeshType>::getOffsetVector( VertId v ) const
{ return ( noXf_ ? mesh_.normal( v ) : ( xfNormal_ * mesh_.dirDblArea( v ) ).normalized() ) * voxelSize_; }

template <typename MeshType>
float MeshOnVoxelsT<MeshType>::pseudoIndex( float index, int count )
{ return index - ( count - 1 ) * 0.5f; }

template <typename MeshType>
float MeshOnVoxelsT<MeshType>::pseudoIndex( int index, int count )
{ return pseudoIndex( ( float )index, count ); }

template <typename MeshType>
float MeshOnVoxelsT<MeshType>::indexFromPseudoIndex( float pseudoIndex, int count )
{ return pseudoIndex + ( count - 1 ) * 0.5f; }

template <typename MeshType>
void MeshOnVoxelsT<MeshType>::getPoints( std::vector<Vector3f>& result, const Vector3f& pos, const Vector3f& offset ) const
{
    Vector3f p = pos - ( offset * ( ( result.size() - 1 ) * 0.5f ) );
    for ( auto& v : result )
    {
        v = p;
        p += offset;
    }
}

template <typename MeshType>
void MeshOnVoxelsT<MeshType>::getValues( std::vector<float>& result, const Vector3f& pos, const Vector3f& offset ) const
{
    Vector3f p = pos - ( offset * ( ( result.size() - 1 ) * 0.5f ) );
    for ( auto& v : result )
    {
        v = getValue( p );
        p += offset;
    }
}

template <typename MeshType>
void MeshOnVoxelsT<MeshType>::getDerivatives( std::vector<float>& result, const std::vector<float>& values )
{
    assert( result.size() == values.size() - 1 );
    for ( size_t i = 0; i < result.size(); i++ )
        result[i] = values[i + 1] - values[i];
}


template class MeshOnVoxelsT<Mesh>;
template class MeshOnVoxelsT<const Mesh>;


namespace
{

VertBitSet adjustOneIter( MeshOnVoxels& mv, int samplePoints, float intermediateSmoothSpeed )
{
    MR_TIMER
    auto& meshCopy = mv.mesh();
    VertCoords& newPoints = meshCopy.points;

    VertBitSet correctedPoints( mv.mesh().points.size() );
    Vector<Vector3f, VertId> shifts( meshCopy.points.size(), Vector3f{ 0.f, 0.f, 0.f } );

    struct ThreadSpecific {
        MeshOnVoxels mv;            // Volume accessors are copied for thread safety
        std::vector<float> values;  // Pre-allocate working vectors
        std::vector<float> derivatives;
    } threadSpecificExemplar {
        mv,
        std::vector<float>( samplePoints ),
        std::vector<float>( samplePoints - 1 )
    };
    tbb::enumerable_thread_specific<ThreadSpecific> threadSpecific( std::move( threadSpecificExemplar ) );
    BitSetParallelFor( mv.mesh().topology.getValidVerts(), threadSpecific,
        [&correctedPoints, &shifts] ( VertId v, ThreadSpecific &local )
        {
            std::vector<float> &values = local.values;
            std::vector<float> &derivatives = local.derivatives;

            // Calculate values
            Vector3f pt = local.mv.point( v ), offset = local.mv.getOffsetVector( v );
            local.mv.getValues( values, pt, offset );
            local.mv.getDerivatives( derivatives, values );

            auto parabola = local.mv.getBestParabola( derivatives.begin(), derivatives.end() );
            if ( parabola.a < 0.f )
                return;

            float minX = parabola.extremArg();
            // Adjust point
            if ( minX >= -1.f && minX <= 1.f )
            {
                correctedPoints.set( v );
                shifts[v] = std::clamp( minX, -0.1f, 0.1f ) * offset;
            }
        } );

    Vector<float, VertId> weights( meshCopy.points.size(), 1.f );
    BitSetParallelFor( correctedPoints,
        [weights = weights.data()] ( VertId v )
        {
            weights[v] = 10.f;
        } );

    constexpr int smoothIters = 15;

    relax(
        meshCopy.topology,
        shifts,
        MeshRelaxParams{ {
                .iterations = smoothIters,
                .force = intermediateSmoothSpeed
            },
            false, &weights }
    );

    BitSetParallelFor( mv.mesh().topology.getValidVerts(),
        [newPoints = newPoints.data(), shifts = shifts.data()] ( VertId v )
        {
            newPoints[v] += shifts[v];
        } );

    relax(
        meshCopy,
        MeshRelaxParams{ {
                .iterations = smoothIters,
                .force = 0.01f
            },
            false, &weights }
    );

    meshCopy.invalidateCaches();
    return correctedPoints;
}


}

VertBitSet moveMeshToVoxelMaxDeriv(
        Mesh& mesh, const AffineXf3f& meshXf,
        const VdbVolume& volume, const AffineXf3f& volumeXf,
        const MoveMeshToVoxelMaxDerivSettings& settings,
        ProgressCallback callback
    )
{
    MR_TIMER
    MeshOnVoxels mv( mesh, meshXf, volume, volumeXf );

    relax( mesh, { { .iterations = 1, .force = settings.preparationSmoothForce } } );

    VertBitSet correctedPoints;
    for ( int i = 0; i < settings.iters; ++i )
    {
        correctedPoints |= adjustOneIter( mv, settings.samplePoints, settings.intermediateSmoothForce );
        if ( !reportProgress( callback, (float)i / (float)settings.iters ) )
            break;
    }

    return correctedPoints;
}

}
