#include <MRMesh/MRLaplacian.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, Laplacian )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    {
        VertBitSet vs;
        vs.autoResizeSet( 0_v );
        Laplacian laplacian( sphere );
        laplacian.init( vs, EdgeWeights::Cotan );
        laplacian.apply();

        // fix the only free vertex
        laplacian.fixVertex( 0_v );
        laplacian.apply();
    }

    {
        Laplacian laplacian( sphere );
        // no free verts
        laplacian.init( {}, EdgeWeights::Cotan );
        laplacian.apply();
    }
}

} //namespace MR
