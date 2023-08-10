#include "MREmbedTerrainStructure.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MR2to3.h"
#include "MRDistanceMap.h"
#include "MRMeshFillHole.h"
#include "MRFillContours2D.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MRMeshBoolean.h"
#include "MRFillContour.h"
#include "MRId.h"
#include "MRMeshSave.h"

namespace MR
{

Expected<Mesh, std::string> embedStructureToTerrain( 
    const Mesh& terrain, const Mesh& structure, const EmbeddedStructureParameters& params )
{
    MR_TIMER;

    auto resMesh = terrain;
    auto globalBox = resMesh.computeBoundingBox();
    globalBox.include( structure.computeBoundingBox() );
    
    auto boxSize = globalBox.size();
    const float boxExpansion = 0.5f;
    auto mainComponent = terrain.topology.getValidFaces();
    FaceBitSet extenedFaces;
    for ( auto e : resMesh.topology.findHoleRepresentiveEdges() )
        extendHole( resMesh, e, Plane3f( Vector3f::plusZ(), globalBox.min.z - boxSize.z * boxExpansion ), &extenedFaces );

    auto structBound = findRightBoundary( structure.topology );
    if ( structBound.size() != 1 )
        return unexpected( "Structure should have only one boundary" );

    BooleanPreCutResult structPrecutRes;
    VertBitSet structCutVerts;
    boolean( resMesh, structure, BooleanOperation::InsideB, { .outPreCutB = &structPrecutRes } );
    if ( !structPrecutRes.contours.empty() )
    {
        auto structCutRes = cutMesh( structPrecutRes.mesh, structPrecutRes.contours );
        if ( structCutRes.fbsWithCountourIntersections.any() )
            return unexpected( "Intersection contour of structure and terrain has self-intersections" );
        structCutVerts = getIncidentVerts( structPrecutRes.mesh.topology,
                                                     fillContourLeft( structPrecutRes.mesh.topology, structCutRes.resultCut ) );
        structPrecutRes.mesh.topology.flip( structCutVerts );
    }
    else
    {
        auto sFace = structure.topology.getValidFaces().find_first();
        assert( sFace );

        Vector3f sPoint = structure.triCenter( sFace );
        auto signDist = resMesh.signedDistance( sPoint, FLT_MAX );
        if ( signDist && signDist < 0.0f )
            structCutVerts = structPrecutRes.mesh.topology.getValidVerts();
    }

    EmbeddedConeParameters ecParams{ params };
    ecParams.minZ = globalBox.min.z - boxSize.z * boxExpansion * 2.0f / 3.0f;
    ecParams.maxZ = globalBox.max.z + boxSize.z * boxExpansion * 2.0f / 3.0f;

    Contour3f cont( structBound[0].size() + 1 );
    ecParams.cutBitSet.resize( structBound[0].size() );
    for ( int i = 0; i + 1 < cont.size(); ++i )
    {
        auto org = structure.topology.org( structBound[0][i] );
        cont[i] = structure.points[org];
        if ( structCutVerts.test( org ) )
            ecParams.cutBitSet.set( VertId( i ) );
    }
    cont.back() = cont.front();

    auto coneMeshRes = createEmbeddedConeMesh( cont, ecParams );
    if ( !coneMeshRes.has_value() )
        return unexpected( coneMeshRes.error() );

    MeshSave::toMrmesh( coneMeshRes->mesh, "C:\\Users\\grant\\Downloads\\objects (1)\\coneMeshRes.mrmesh" );

    BooleanPreCutResult precutResTerrain;
    BooleanPreCutResult precutResCone;
    boolean( std::move( resMesh ), std::move( coneMeshRes->mesh ), BooleanOperation::Union,
        { .outPreCutA = &precutResTerrain,.outPreCutB = &precutResCone } );
    // filter excessive contours
    if ( precutResTerrain.contours.size() != 1 )
    {
        int minI = 0;
        float minDistSq = FLT_MAX;
        for ( int i = 0; i < precutResTerrain.contours.size(); ++i )
        {
            Vector3f center;
            for ( const auto& coord : precutResTerrain.contours[i].intersections )
                center += coord.coordinate;

            center /= float( precutResTerrain.contours[i].intersections.size() );
            auto distSq = ( center - structure.getBoundingBox().center() ).lengthSq();
            if ( distSq < minDistSq )
            {
                minDistSq = distSq;
                minI = i;
            }
        }
        if ( minI != 0 )
        {
            std::swap( precutResTerrain.contours[0], precutResTerrain.contours[minI] );
            std::swap( precutResCone.contours[0], precutResCone.contours[minI] );
        }
        precutResTerrain.contours.erase( precutResTerrain.contours.begin() + 1, precutResTerrain.contours.end() );
        precutResCone.contours.erase( precutResCone.contours.begin() + 1, precutResCone.contours.end() );
    }

    FaceMap newTerrainFacesMap;
    auto oldTerrainFaceSize = precutResTerrain.mesh.topology.faceSize();
    auto cutTerrainRes = cutMesh( precutResTerrain.mesh, precutResTerrain.contours, { .new2OldMap = &newTerrainFacesMap } );
    auto cutTerrainPart = fillContourLeft( precutResTerrain.mesh.topology, cutTerrainRes.resultCut );
    precutResTerrain.mesh.topology.deleteFaces( cutTerrainPart );
    precutResTerrain.mesh.invalidateCaches();
    extenedFaces.resize( precutResTerrain.mesh.topology.faceSize() );
    for ( FaceId nf = FaceId( oldTerrainFaceSize ); nf < precutResTerrain.mesh.topology.faceSize(); ++nf )
    {
        auto of = newTerrainFacesMap[nf];
        if ( !of )
            continue;
        if ( extenedFaces.test( of ) )
            extenedFaces.set( nf );
    }


    FaceMap new2oldFaces;
    auto oldConeFaceSize = precutResCone.mesh.topology.faceSize();
    auto cutConeRes = cutMesh( precutResCone.mesh, precutResCone.contours, { .new2OldMap = &new2oldFaces } );
    coneMeshRes->fillBitSet.resize( precutResCone.mesh.topology.faceSize() );
    coneMeshRes->cutBitSet.resize( precutResCone.mesh.topology.faceSize() );
    for ( FaceId nf = FaceId( oldConeFaceSize ); nf < precutResCone.mesh.topology.faceSize(); ++nf )
    {
        auto of = new2oldFaces[nf];
        if ( !of )
            continue;

        if ( coneMeshRes->fillBitSet.test( of ) )
            coneMeshRes->fillBitSet.set( nf );
        else if ( coneMeshRes->cutBitSet.test( of ) )
            coneMeshRes->cutBitSet.set( nf );
    }
    auto cutConePart = fillContourLeft( precutResCone.mesh.topology, cutConeRes.resultCut );
    auto upperPartFaces = cutConePart & coneMeshRes->cutBitSet;
    auto lowerPartFaces = ( precutResCone.mesh.topology.getValidFaces() - cutConePart ) & coneMeshRes->fillBitSet;
    precutResCone.mesh.topology.deleteFaces( upperPartFaces | lowerPartFaces );
    precutResCone.mesh.invalidateCaches();


    // merging cone parts
    std::vector<EdgeLoop> cutConeLoop;
    std::vector<EdgeLoop> fillConeLoop;
    std::vector<EdgeLoop> cutTerrainLoop;
    std::vector<EdgeLoop> fillTerrainLoop;
    bool prevCutPart = true;
    assert( cutConeRes.resultCut[0].size() == cutTerrainRes.resultCut[0].size() );
    for ( int i = 0; i < cutConeRes.resultCut[0].size(); ++i )
    {
        auto te = cutTerrainRes.resultCut[0][i];
        auto ce = cutConeRes.resultCut[0][i];
        bool thisCutPart = !precutResCone.mesh.topology.left( ce ).valid();
        if ( thisCutPart == prevCutPart && i != 0 )
        {
            auto& prevCone = thisCutPart ? cutConeLoop.back() : fillConeLoop.back();
            auto& prevTerrain = thisCutPart ? cutTerrainLoop.back() : fillTerrainLoop.back();
            prevCone.push_back( ce );
            prevTerrain.push_back( te );
        }
        else
        {
            auto& prevCone = thisCutPart ? cutConeLoop : fillConeLoop;
            auto& prevTerrain = thisCutPart ? cutTerrainLoop : fillTerrainLoop;
            prevCone.push_back( { ce } );
            prevTerrain.push_back( { te } );
            prevCutPart = thisCutPart;
        }
    }

    precutResTerrain.mesh.addPartByMask( precutResCone.mesh, precutResCone.mesh.topology.getValidFaces() & coneMeshRes->cutBitSet,
                                   true, cutTerrainLoop, cutConeLoop );
    precutResTerrain.mesh.addPartByMask( precutResCone.mesh, precutResCone.mesh.topology.getValidFaces() & coneMeshRes->fillBitSet,
                                   false, fillTerrainLoop, fillConeLoop );

    // merge with structure
    auto holes = findRightBoundary( precutResTerrain.mesh.topology );
    assert( !holes.empty() );

    auto loopStructure = findLeftBoundary( structPrecutRes.mesh.topology )[0];
    if ( loopStructure.size() != holes.back().size() )
        return unexpected( "Cannot stitch structure with terrain" );

    int i = 0;
    for ( ; i < loopStructure.size(); ++i )
    {
        if ( structPrecutRes.mesh.orgPnt( loopStructure[i] ) ==
                 precutResTerrain.mesh.orgPnt( holes.back().front() ) )
            break;
    }
    std::rotate( loopStructure.begin(), loopStructure.begin() + i, loopStructure.end() );

    precutResTerrain.mesh.addPartByMask( structPrecutRes.mesh, structPrecutRes.mesh.topology.getValidFaces(), false, { holes.back() }, { loopStructure } );
    

    // remove extenedFaces
    extenedFaces &= precutResTerrain.mesh.topology.getValidFaces();
    precutResTerrain.mesh.topology.deleteFaces( extenedFaces );
    precutResTerrain.mesh.invalidateCaches();

    return precutResTerrain.mesh;
}

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

    auto cutOffset = std::abs( params.maxZ - centerZ ) * std::clamp( std::tan( params.cutAngle ), 0.0f, 100.0f );
    auto fillOffset = std::abs( params.minZ - centerZ ) * std::clamp( std::tan( params.fillAngle ), 0.0f, 100.0f );

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
            {
                auto nd = res.mesh.topology.dest( res.mesh.topology.next( flipE ) );
                auto pd = res.mesh.topology.dest( res.mesh.topology.prev( flipE ) );
                if ( nd >= params.cutBitSet.size() || pd >= params.cutBitSet.size() )
                    res.mesh.topology.flipEdge( flipE );
            }
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