#include "MREmbedTerrainStructure.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MR2to3.h"
#include "MRDistanceMap.h"
#include "MRMeshFillHole.h"
#include "MRFillContours2D.h"
#include "MRRingIterator.h"

namespace MR
{

Expected<EmbeddedConeResult, std::string> createEmbeddedConeMesh(
    const Contour3f& cont, const EmbeddedConeParameters& params )
{
    MR_TIMER;

    if ( cont.size() < 4 || cont.back() != cont.front() )
        return unexpected( "Input contour should be closed" );
    float centerZ = 0;
    Box2f contBox;
    int i = 0;
    for ( ; i + 1 < cont.size(); ++i )
    {
        centerZ += cont[i].z;
        contBox.include( to2dim( cont[i] ) );
    }
    centerZ /= float( i );

    auto cutOffset = std::abs( params.maxZ - centerZ ) * std::tan( params.cutAngle );
    auto fillOffset = std::abs( params.minZ - centerZ ) * std::tan( params.fillAngle );

    bool needCutPart = params.cutBitSet.any();
    bool needFillPart = params.cutBitSet.count() + 1 != cont.size();

    float maxOffset = 0.0f;
    if ( needCutPart && needFillPart )
        maxOffset = std::max( cutOffset, fillOffset );
    else if ( needCutPart )
        maxOffset = cutOffset;
    else
        maxOffset = fillOffset;

    contBox.min -= Vector2f::diagonal( maxOffset + 3.0f * params.pixelSize );
    contBox.max += Vector2f::diagonal( maxOffset + 3.0f * params.pixelSize );

    ContourToDistanceMapParams dmParams;
    dmParams.orgPoint = contBox.min;
    dmParams.pixelSize = Vector2f::diagonal( params.pixelSize );
    dmParams.resolution = Vector2i( contBox.size() / params.pixelSize );
    dmParams.withSign = true;

    if ( dmParams.resolution.x > 2000 || dmParams.resolution.y > 2000 )
        return unexpected( "Exceed precision limit" );

    auto dm = distanceMapFromContours( Polyline2( { cont } ), dmParams );


    EmbeddedConeResult res;

    auto moveToContVert = [&] ( VertId v, bool cut )->VertId
    {
        for ( auto e : orgRing( res.mesh.topology, v ) )
        {
            auto d = res.mesh.topology.dest( e );
            if ( d < params.cutBitSet.size() && ( cut != params.cutBitSet.test( d ) ) )
                return d;
        }
        return {};
    };

    auto buildPart = [&] ( bool fill )->std::string
    {
        auto cont2f = distanceMapTo2DIsoPolyline( dm, dmParams, fill ? fillOffset : cutOffset );
        auto cont3f = cont2f.topology.convertToContours<Vector3f>( [&] ( VertId v )
        {
            return Vector3f( cont2f.points[v].x, cont2f.points[v].y, fill ? params.minZ : params.maxZ );
        } );
        auto vertSize = res.mesh.topology.lastValidVert() + 1;
        auto eBase = res.mesh.addSeparateContours( cont3f );

        buildCylinderBetweenTwoHoles( res.mesh,
            fill ? eBase.sym() : eBase,
            fill ? EdgeId{ 0 } : EdgeId{ 1 },
            { .metric = getMinTriAngleMetric( res.mesh ) } );

        auto fillRes = fillContours2D( res.mesh, { fill ? eBase : eBase.sym() } );
        if ( !fillRes.has_value() )
            return fillRes.error();
        // moves
        for ( VertId v = vertSize; v <= res.mesh.topology.lastValidVert(); ++v )
        {
            if ( auto mv = moveToContVert( v, !fill ) )
            {
                res.mesh.points[v].x = res.mesh.points[mv].x;
                res.mesh.points[v].y = res.mesh.points[mv].y;
            }
        }
        // flips
        for ( int ei = 0; ei + 1 < cont.size(); ++ei )
        {
            EdgeId e( 2 * ei );
            auto org = res.mesh.topology.org( e );
            auto dest = res.mesh.topology.dest( e );
            bool orgCut = params.cutBitSet.test( org );
            if ( orgCut == params.cutBitSet.test( dest ) )
                continue;
            EdgeId flipE;
            if ( fill )
                flipE = orgCut ? res.mesh.topology.prev( e.sym() ) : res.mesh.topology.next( e );
            else
                flipE = orgCut ? res.mesh.topology.prev( e ) : res.mesh.topology.next( e.sym() );
            if ( moveToContVert( res.mesh.topology.dest( flipE ), !fill ).valid() )
                res.mesh.topology.flipEdge( flipE );
        }
        if ( fill )
            res.fillBitSet = res.mesh.topology.getValidFaces();
        else
            res.cutBitSet = res.mesh.topology.getValidFaces() - res.fillBitSet;
        return {};
    };

    // add base contour
    res.mesh.addSeparateContours( { cont } );
    if ( needFillPart )
    {
        auto error = buildPart( true );
        if ( !error.empty() )
            return unexpected( "Building fill part failed: " + error );
    }

    if ( needCutPart )
    {
        auto error = buildPart( false );
        if ( !error.empty() )
            return unexpected( "Building cut part failed: " + error );
    }

    return res;
}

}