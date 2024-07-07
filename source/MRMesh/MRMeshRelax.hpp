#pragma once

#include "MRMeshRelax.h"
#include "MRMeshTopology.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshFixer.h"
#include "MRTimer.h"

namespace MR
{

/// set the field in the vertices with exactly three neighbor vertices as the average value of the field in the neighbors
template<typename T>
void hardSmoothTetrahedrons( const MeshTopology & topology, Vector<T, VertId> & field, const VertBitSet *region = nullptr )
{
    MR_TIMER
    auto tetrahedrons = findNRingVerts( topology, 3, region );
    // in normal mesh two vertices from tetrahedrons cannot be neighbors, so it is safe to run it in parallel
    BitSetParallelFor( tetrahedrons, [&] ( VertId v )
    {
        T center{};
        for ( auto e : orgRing( topology, v ) )
            center += field[ topology.dest( e ) ];
        field[v] = center / 3.0f;
    } );
}

/// applies given number of relaxation iterations to given field on mesh vertices;
/// \return true if was finished successfully, false if was interrupted by progress callback
template<typename T>
bool relax( const MeshTopology & topology, Vector<T, VertId> & field, const MeshRelaxParams& params = {}, ProgressCallback cb = {} )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    Vector<T, VertId> initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = field;

    Vector<T, VertId> newField;
    const VertBitSet& zone = topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newField = field;
        if ( !BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            T sum{};
            int count = 0;
            for ( auto e : orgRing( topology, e0 ) )
            {
                sum += field[topology.dest( e )];
                ++count;
            }
            auto np = newField[v];
            auto pushForce = params.force * ( sum / float( count ) - np );
            np += pushForce;
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newField[v] = np;
        }, internalCb ) )
            return false;
        field.swap( newField );
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedrons( topology, field, params.region );
    return true;
}

} //namespace MR
