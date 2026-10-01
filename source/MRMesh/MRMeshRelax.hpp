#pragma once

#include "MRMeshRelax.h"
#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRMeshFixer.h"
#include "MRTimer.h"

namespace MR
{

/// set the field in the vertices with exactly three neighbor vertices as the average value of the field in the neighbors
template<typename T>
void hardSmoothTetrahedronsT( const MeshTopology & topology, Vector<T, VertId> & field, const VertBitSet *region = nullptr )
{
    MR_TIMER;
    auto tetrahedrons = findInnerVertsOfDegree( topology, 3, region );
    // in normal mesh two vertices from tetrahedrons cannot be neighbors, so it is safe to run it in parallel
    BitSetParallelFor( tetrahedrons, [&] ( VertId v )
    {
        T center{};
        for ( auto e : orgRing( topology, v ) )
            center += field[ topology.dest( e ) ];
        field[v] = center / 3.0f;
    } );
}

/// This class is responsible for limiting vertex movement during relaxation according to parameters
template<typename T>
class VertLimiter
{
public:
    /// initialization
    VertLimiter( const Vector<T, VertId> & initialPos, const MeshRelaxParams& params ) : params_( params )
    {
        maxInitialDistSq_ = sqr( params.maxInitialDist );
        if ( params.limitNearInitial )
            initialPos_ = initialPos;
    }

    /// returns limited coordinates of the vertex (v) given its target unlimited position
    T operator()( VertId v, T targetPos ) const
    {
        if ( params_.limitNearInitial )
            targetPos = getLimitedPos( targetPos, initialPos_[v], maxInitialDistSq_ );
        return targetPos;
    }

private:
    const MeshRelaxParams& params_;
    Vector<T, VertId> initialPos_;
    float maxInitialDistSq_ = 0;
};

/// applies given number of relaxation iterations to given field on mesh vertices;
/// \return true if was finished successfully, false if was interrupted by progress callback
template<typename T>
bool relaxT( const MeshTopology & topology, Vector<T, VertId> & field, const MeshRelaxParams& params = {}, const ProgressCallback& cb = {} )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER;

    const auto getWeightOrDefault = [w = params.weights] ( VertId v ) -> float
    {
        if ( w )
            return ( *w )[v];
        else
            return 1.f;
    };

    VertLimiter limiter( field, params );

    // computes new value of vertex (v) from the values of its neighbors
    const auto relaxedValue = [&]( VertId v )
    {
        T sum{};
        float sumWeight = 0.f;
        for ( auto e : orgRing( topology, v ) )
        {
            const auto dst = topology.dest( e );
            const auto w = getWeightOrDefault( dst );
            sum += w * field[dst];
            sumWeight += w;
        }
        auto np = field[v];
        auto pushForce = params.force * ( sum / sumWeight - np );
        np += pushForce;
        return limiter( v, np );
    };

    const VertBitSet& zone = topology.getVertIds( params.region );
    if ( 2 * zone.count() > field.size() )
    {
        // dense zone: double-buffer the whole field
        Vector<T, VertId> newField;
        for ( int i = 0; i < params.iterations; ++i )
        {
            auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
            newField = field;
            if ( !BitSetParallelFor( zone, [&]( VertId v )
            {
                if ( topology.edgeWithOrg( v ) )
                    newField[v] = relaxedValue( v );
            }, internalCb ) )
                return false;
            field.swap( newField );
        }
    }
    else
    {
        // sparse zone: store new values only for zone vertices
        std::vector<VertId> zoneVerts;
        zoneVerts.reserve( zone.count() );
        for ( auto v : zone )
            if ( topology.edgeWithOrg( v ) )
                zoneVerts.push_back( v );
        std::vector<T> newValues( zoneVerts.size() );
        for ( int i = 0; i < params.iterations; ++i )
        {
            auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
            if ( !ParallelFor( zoneVerts, [&]( size_t j )
            {
                newValues[j] = relaxedValue( zoneVerts[j] );
            }, internalCb ) )
                return false;
            ParallelFor( zoneVerts, [&]( size_t j )
            {
                field[zoneVerts[j]] = newValues[j];
            } );
        }
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedronsT( topology, field, params.region );
    return true;
}

} //namespace MR
