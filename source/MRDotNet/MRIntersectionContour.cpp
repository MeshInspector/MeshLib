#include "MRIntersectionContour.h"
#include "MRMesh.h"
#include "MRMeshCollidePrecise.h"

#pragma managed( push, off )
#include <MRMesh/MRIntersectionContour.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshTopology.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

ContinousContours^ IntersectionContour::OrderIntersectionContours( Mesh^ meshA, Mesh^ meshB, PreciseCollisionResult^ intersections )
{
    if ( !meshA )
        throw gcnew System::ArgumentNullException( "meshA" );
    if ( !meshB )
        throw gcnew System::ArgumentNullException( "meshB" );
    if ( !intersections )
        throw gcnew System::ArgumentNullException( "intersections" );

    auto nativeRes = MR::orderIntersectionContours( meshA->getMesh()->topology, meshB->getMesh()->topology, *intersections->getNativeResult() );
    ContinousContours^ res = gcnew ContinousContours( int( nativeRes.size() ) );
    for ( size_t i = 0; i < nativeRes.size(); i++ )
    {
        auto contour =  gcnew ContinousContour( int( nativeRes[i].size() ) );
        for ( size_t j = 0; j < nativeRes[i].size(); j++ )
        {
            VariableEdgeTri vet;
            vet.edge = EdgeId( nativeRes[i][j].edge );
            vet.tri = FaceId( nativeRes[i][j].tri );
            vet.isEdgeATriB = nativeRes[i][j].isEdgeATriB;
            contour->Add( vet );
        }

        res->Add( contour );            
    }

    return res;
}

MR_DOTNET_NAMESPACE_END