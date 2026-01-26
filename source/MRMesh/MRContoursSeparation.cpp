#include "MRContoursSeparation.h"
#include "MRMesh.h"
#include "MRSurfacePath.h"
#include "MRRingIterator.h"
#include "MRMeshComponents.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

std::vector<FaceBitSet> separateClosedContour( const Mesh& mesh, const std::vector<Vector3f>& contour,
                                               const PathMeshEdgePointCallback& cb )
{
    MR_TIMER;

    if ( contour.size() < 3 )
        return {};

    std::vector<MeshTriPoint> projections( contour.size() );
    MR::ParallelFor( contour, [&] ( size_t i )
    {
        if ( auto projRes = mesh.projectPoint( contour[i] ) )
            projections[i] = projRes.mtp;
    } );
    for ( const auto& mtp : projections )
        if ( !mtp.e.valid() )
            return {};

    std::vector<SurfacePath> paths( projections.size() );
    ParallelFor( projections, [&] ( size_t i )
    {
        auto sp = computeSurfacePath( mesh, projections[i], projections[( i + 1 ) % projections.size()] );
        if ( sp.has_value() )
            paths[i] = std::move( sp.value() );
    } );

    FaceBitSet contourFaces( mesh.topology.getValidFaces().size() );
    auto addLeft = [&mesh, &contourFaces]( EdgeId e )
    {
        if ( FaceId f = mesh.topology.left( e ) )
            contourFaces.set( f );
    };
    auto addMEP = [&mesh,&addLeft,&cb]( const MeshEdgePoint& mep )
    {
        VertId v = mep.inVertex( mesh.topology );
        if ( v.valid() )
        {
            for ( auto e : orgRing( mesh.topology, v ) )
                addLeft( e );
        }
        else
        {
            addLeft( mep.e );
            addLeft( mep.e.sym() );
        }
        if ( cb )
            cb( mep );
    };
    for ( int i = 0; i < projections.size(); ++i )
    {
        const auto& mtp = projections[i];
        if ( auto mep = mtp.onEdge( mesh.topology ) )
            addMEP( mep );
        else
            addLeft( mtp.e );

        for ( const auto& mep2 : paths[i] )
            addMEP( mep2 );
    }

    FaceBitSet facesForSeparation = MeshComponents::getComponents( mesh, contourFaces ) - contourFaces;

    return MeshComponents::getAllComponents( { mesh,&facesForSeparation } );
}

} //namespace MR

