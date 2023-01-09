#include "MRSmoothLineViaVertices.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRLaplacian.h"
#include "MRTimer.h"

namespace MR
{

MeshLines getSmoothLinesViaVertices( const Mesh & mesh, const VertBitSet & vs )
{
    MR_TIMER
    MeshLines res;
    if ( vs.empty() )
        return res;

    Vector<float,VertId> scalarField( mesh.topology.vertSize(), 0 );
    VertBitSet freeVerts;
    for ( const auto & cc : MeshComponents::getAllComponentsVerts( mesh ) )
    {
        auto freeCC = cc - vs;
        auto numfree = freeCC.count();
        if ( numfree <= 0 )
            continue; // too small connected component
        if ( numfree == cc.count() )
            continue; // no single fixed vertex in the component

        // fix one additional vertex in each connected component with the value 1
        // (to avoid constant 0 solution)
        VertId fixedV = *begin( freeCC );
        scalarField[fixedV] = 1;
        freeCC.reset( fixedV );
        freeVerts |= freeCC;
    }

    Laplacian lap( const_cast<Mesh&>( mesh ) ); //mesh will not be changed
    lap.init( freeVerts, Laplacian::EdgeWeights::Unit, Laplacian::RememberShape::No );
    lap.applyToScalar( scalarField );
    res = extractIsolines( mesh.topology, scalarField, 0 );

    return res;
}

} //namespace MR
