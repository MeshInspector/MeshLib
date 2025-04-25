#include <MRMesh/MRLaplacian.h>
#include <MRMesh/MRMesh.h>

int main()
{
    MR::Mesh mesh;
    MR::VertBitSet region;
    MR::VertId v0, v1;
    MR::Vector3f newPos0, newPos1;

//! [0]    
    // Construct deformer on the mesh vertices
    MR::Laplacian lDeformer( mesh );

    // Initialize laplacian
    lDeformer.init( region, MR::EdgeWeights::Cotan, MR::VertexMass::NeiArea );

    // Fix the anchor vertices in the required position
    lDeformer.fixVertex( v0, newPos0 );
    lDeformer.fixVertex( v1, newPos1 );

    // Move the free vertices according to the anchor ones
    lDeformer.apply();

    // Invalidate the mesh because of the external vertex changes
    mesh.invalidateCaches();
//! [0]    

    return EXIT_SUCCESS;
}
