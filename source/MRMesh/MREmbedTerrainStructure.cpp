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
#include "MR2DContoursTriangulation.h"
#include "MRTriMath.h"

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

    bool needCut = ecParams.cutBitSet.any();
    bool needFill = ecParams.cutBitSet.count() + 1 != cont.size();

    auto coneMeshRes = createEmbeddedConeMesh( cont, ecParams );
    if ( !coneMeshRes.has_value() )
        return unexpected( coneMeshRes.error() );

    FaceBitSet extenedFaces;
    if ( !needFill || !needCut )
    {
        auto desiredZ = needFill ?
            globalBox.max.z + boxSize.z * boxExpansion :
            globalBox.min.z - boxSize.z * boxExpansion;
        for ( auto e : resMesh.topology.findHoleRepresentiveEdges() )
            extendHole( resMesh, e, Plane3f( Vector3f::plusZ(), desiredZ ), &extenedFaces );
    }


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

    if ( !precutResCone.contours.front().closed )
        return unexpected( "Cannot embed structure beyond terrain" );

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

    auto contNorm = [&] ( int i )
    {
        auto norm = to2dim( cont[i + 1] ) - to2dim( cont[i] );
        std::swap( norm.x, norm.y );
        norm.x = -norm.x;
        norm = norm.normalized();
        return norm;
    };

    struct OffsetBlock
    {
        Mesh mesh;
        std::vector<int> contourOffsets;
    };
    auto createOffet = [&] ( float offset )
    {
        OffsetBlock res;
        res.contourOffsets.resize( int( cont.size() ), 0 );
        Contour2f offsetCont;

        offsetCont.reserve( 3 * cont.size() );
        auto lastPoint = to2dim( cont[0] ) + offset * contNorm( int( cont.size() ) - 2 );
        for ( int i = 0; i + 1 < cont.size(); ++i )
        {
            auto orgPt = to2dim( cont[i] );
            auto destPt = to2dim( cont[i + 1] );
            auto norm = contNorm( i );

            auto nextPoint = orgPt + norm * offset;
            bool sameAsPrev = false;
            // interpolation    
            if ( offsetCont.empty() )
            {
                offsetCont.emplace_back( std::move( lastPoint ) );
                ++res.contourOffsets[i];
            }
            auto prevPoint = offsetCont.back();
            auto a = prevPoint - orgPt;
            auto b = nextPoint - orgPt;
            auto crossRes = cross( a, b );
            auto dotRes = dot( a, b );
            float ang = 0.0f;
            if ( crossRes == 0.0f )
                ang = dotRes >= 0.0f ? 0.0f : PI_F;
            else 
                ang = std::atan2( crossRes, dotRes );
                    
            sameAsPrev = std::abs( ang ) < PI_F / 360.0f;
            if ( !sameAsPrev )
            {
                int numSteps = int( std::floor( std::abs( ang ) / ( params.minAnglePrecision ) ) );
                for ( int s = 0; s < numSteps; ++s )
                {
                    float stepAng = ( ang / ( numSteps + 1 ) ) * ( s + 1 );
                    auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), orgPt );
                    offsetCont.emplace_back( rotXf( prevPoint ) );
                    ++res.contourOffsets[i];
                }
                offsetCont.emplace_back( std::move( nextPoint ) );
                ++res.contourOffsets[i];
            }

            offsetCont.emplace_back( destPt + norm * offset );
            ++res.contourOffsets[i + 1];
        }
        int prevSum = 0;
        for ( int i = 0; i + 1 < res.contourOffsets.size(); ++i )
        {
            std::swap( res.contourOffsets[i], prevSum );
            prevSum += res.contourOffsets[i];
        }

        res.contourOffsets.back() = int( offsetCont.size() ) - 1;
        res.mesh = PlanarTriangulation::triangulateContours( { offsetCont } );
        return res;
    };
    
    EmbeddedConeResult res;
    res.mesh.addSeparateContours( { cont } );
    bool needCut = params.cutBitSet.any();
    bool needFill = params.cutBitSet.count() + 1 != cont.size();

    auto findInitIndex = [] ( VertId v, const std::vector<int>& offsets, int initSize )->int
    {
        auto diff = v - initSize;
        int h = 0;
        for ( ; h + 1 < offsets.size(); ++h )
        {
            if ( diff >= offsets[h] && diff < offsets[h + 1] )
                break;
        }
        return h;
    };

    auto moveToContVert = [&] ( VertId v, const std::vector<int>& offsets, int initSize, bool cut )->VertId
    {
        auto initVert = VertId( findInitIndex( v, offsets, initSize ) );
        if ( initVert < params.cutBitSet.size() && ( cut != params.cutBitSet.test( initVert ) ) )
            return initVert;
        return {};
    };

    auto postProcess = [&] ( const EdgeLoop& baseBd, const std::vector<int>& offsets, int initSize, bool cut )->std::string
    {
        // moves
        for ( auto e : baseBd )
        {
            auto v = res.mesh.topology.org( e );
            if ( auto mv = moveToContVert( v, offsets, initSize, cut ) )
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
            if ( !cut )
                flipE = orgCut ? res.mesh.topology.prev( e.sym() ) : res.mesh.topology.next( e );
            else
                flipE = orgCut ? res.mesh.topology.prev( e ) : res.mesh.topology.next( e.sym() );
            if ( moveToContVert( res.mesh.topology.dest( flipE ), offsets, initSize, cut ).valid() )
            {
                auto nd = res.mesh.topology.dest( res.mesh.topology.next( flipE ) );
                auto pd = res.mesh.topology.dest( res.mesh.topology.prev( flipE ) );
                if ( nd >= params.cutBitSet.size() || pd >= params.cutBitSet.size() )
                    res.mesh.topology.flipEdge( flipE );
            }
        }
        if ( cut )
            res.cutBitSet = res.mesh.topology.getValidFaces();
        else
            res.fillBitSet = res.mesh.topology.getValidFaces() - res.cutBitSet;
        return {};
    };

    auto addBase = [&] ( bool cut )
    {
        auto basePart = createOffet( cut ? cutOffset : fillOffset );
        for ( auto& p : basePart.mesh.points )
            p.z = cut ? params.maxZ : params.minZ;

        if ( !cut )
            basePart.mesh.topology.flipOrientation();

        VertId vertSize = res.mesh.topology.lastValidVert() + 1;
        res.mesh.addPart( std::move( basePart.mesh ) );
        auto holes = findRightBoundary( res.mesh.topology );
        if ( cut && holes.size() != 3 )
            return false;
        else if ( !cut && holes.size() != ( needCut ? 2 : 3 ) )
            return false;

        auto ueSize = int( res.mesh.topology.undirectedEdgeSize() );
        const auto& baseHole = ( cut || !needCut ) ? holes[1] : holes[0];
        for (
            int ei = cut ? 0 : int( holes.back().size() ) - 1;
            cut ? ei < holes.back().size() : ei >= 0;
            cut ? ++ei : --ei )
        {
            auto e = cut ? holes.back()[ei] : holes.back()[( ei + 2 ) % holes.back().size()];
            auto v = res.mesh.topology.org( e );

            int h = findInitIndex( v, basePart.contourOffsets, vertSize );
            if ( h < baseHole.size() )
                makeBridgeEdge( res.mesh.topology, e,
                    cut ? res.mesh.topology.prev( baseHole[h] ) : baseHole[h] );
        }

        auto lastEdge = int( res.mesh.topology.undirectedEdgeSize() );
        FillHoleMetric metric;
        metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )->double
        {
            if ( ( a < vertSize && b < vertSize && c < vertSize ) ||
                ( a >= vertSize && b >= vertSize && c >= vertSize ) )
                return DBL_MAX;
            Vector3d aP = Vector3d( res.mesh.points[a] );
            Vector3d bP = Vector3d( res.mesh.points[b] );
            Vector3d cP = Vector3d( res.mesh.points[c] );

            return dblArea( aP, bP, cP );
        };
        for ( int ue = ueSize; ue < lastEdge; ++ue )
            fillHole( res.mesh, EdgeId( ue << 1 ), { .metric = metric } );

        postProcess( holes.back(), basePart.contourOffsets, vertSize, cut );
        return true;
    };

    if ( needCut ) // need cut part
    {
        if ( !addBase( true ) )
            return unexpected( "Cannot make cut offset contour due to precision issues" );
    }
    if ( needFill ) // need fill part
    {
        if ( !addBase( false ) )
            return unexpected( "Cannot make fill offset contour due to precision issues" );
    }

    return res;
}

}