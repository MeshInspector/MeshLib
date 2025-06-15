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
#include "MR2DContoursTriangulation.h"
#include "MRTriMath.h"
#include "MRParallelFor.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRPolyline.h"
#include "MRMapEdge.h"
#include "MRSurfacePath.h"
#include "MRExtractIsolines.h"

namespace MR
{

struct FilterBowtiesResult
{
    Contours2f contours;
    std::vector<std::vector<int>> initIndices;
};

FilterBowtiesResult filterBowties( const Contour2f& cont )
{
    auto mesh = PlanarTriangulation::getOutlineMesh( { cont } );
    auto holes = findRightBoundary( mesh.topology );
    FilterBowtiesResult res;
    res.contours.resize( holes.size() );
    res.initIndices.resize( holes.size() );
    for ( int i = 0; i < holes.size(); ++i )
    {
        const auto& hole = holes[i];
        auto& r = res.contours[i];
        auto& inds = res.initIndices[i];
        r.resize( hole.size() );
        inds.resize( hole.size() );
        for ( int j = 0; j < hole.size(); ++j )
        {
            auto org = mesh.topology.org( hole[j] );
            inds[j] = org + 1 >= cont.size() ? -1 : org.get();
            r[j] = to2dim( mesh.points[org] );
        }
    }
    return res;
}

void filterDuplicates( std::vector<MeshTriPoint>& mtps, std::vector<int>& indices )
{
    for ( int i = int( indices.size() ) - 1; i > 0; --i )
    {
        if ( indices[i] - 1 != indices[i - 1] && mtps[indices[i] - 1] == mtps[indices[i - 1]] )
        {
            // duplicates
            mtps.erase( mtps.begin() + indices[i - 1], mtps.begin() + indices[i] - 1 );
            int diff = indices[i] - indices[i - 1] - 1;
            for ( int j = i; j < indices.size(); ++j )
                indices[j] -= diff;
        }
    }
}

// class to hold intermediate results and process structure embedding
class TerrainEmbedder
{
public:
    TerrainEmbedder( const Mesh& terrain, const Mesh& structure, const EmbeddedStructureParameters& params ) :
        struct_{ structure },
        params_{ params },
        result_{terrain}
    {
    }
    Expected<Mesh> run();
private:
    // cut structure by terrain intersection contour
    Expected<VertBitSet,std::string> createCutStructure_();

    struct MarkedContour
    {
        Contour3f contour;
        VertBitSet cutBitSet;
        VertBitSet intBitSet;
    };
    // makes contour for further processing with marked vertices (cut vertices and terrain intersection vertices)
    Expected<MarkedContour> createMarkedStructureContour_( VertBitSet&& structCutVerts );

    struct MappedMeshContours
    {
        OneMeshContours contours;
        std::vector<std::vector<int>> map; // map from terrain cut edges to projected contours
        std::vector<std::vector<int>> filtBowTiesMap; // map from projected contours to no bow tie filtered contours
        std::vector<int> offsetMap; // map from filtered contours to non-project offset contours (use in findOffsetContourIndex_)
        VertBitSet intBitSet; // bit set of intersections between structure and terrain mesh
        VertBitSet cutBitSet; // bit set of cut part of structure mesh contour
    };
    // make preparation on terrain and finds contour for cut with mapping to cut structure boundary
    Expected<MappedMeshContours> prepareTerrainCut( MarkedContour&& mc );
    // cut terrain with filtered contours and remove internal part
    Expected<std::vector<EdgeLoop>> cutTerrain( const MappedMeshContours& mmc );

    // contains newly added edges from connect_ functions
    struct ConnectionEdges
    {
        EdgePath newCutEdges; // newly added edges of cut part
        EdgePath newFillEdges; // newly added edges of fill part
        EdgePath interEdges; // spliced intersection edges
    };
    // connect hole on terrain with cut structure
    ConnectionEdges connect_( std::vector<EdgeLoop>&& hole, MappedMeshContours&& mmc );
    // fill holes in connection
    void fill_( size_t oldVertSize, ConnectionEdges&& connectionInfo );

    struct OffsetBlock
    {
        Contour2f contour;
        std::vector<int> idsShifts;
    };
    // offset marked contours respecting marks
    OffsetBlock offsetContour_( const MarkedContour& mc, float cutOffset, float fillOffset );
    // return index of non-offset contour by offset index and map
    int findOffsetContourIndex_( int i, const std::vector<int>& idsShifts ) const;

    // structure mesh (only used for making cut structure)
    const Mesh& struct_;
    const EmbeddedStructureParameters& params_;
    
    // result mesh
    Mesh result_;
    // cut structure (last used in connect function, empty after it)
    Mesh cutStructure_;
    // bounds of cutStructure_
    std::vector<EdgeLoop> bounds_;
};

Expected<Mesh> TerrainEmbedder::run()
{
    auto cutBs = createCutStructure_();
    if ( !cutBs.has_value() )
        return unexpected( cutBs.error() );

    auto markedContour = createMarkedStructureContour_( std::move( *cutBs ) );
    if ( !markedContour.has_value() )
        return unexpected( markedContour.error() );

    auto prepCut = prepareTerrainCut( std::move( *markedContour ) );
    if ( !prepCut.has_value() )
        return unexpected( prepCut.error() );

    if ( prepCut->contours.size() > 1 )
        return unexpected( "Non-trivial contours are not supported yet" );

    auto cutTer = cutTerrain( *prepCut );
    if ( !cutTer.has_value() )
        return unexpected( cutTer.error() );

    auto oldVertSize = result_.topology.vertSize();
    auto connectionInfo = connect_( std::move( *cutTer ), std::move( *prepCut ) );

    fill_( oldVertSize, std::move( connectionInfo ) );

    return std::move( result_ );
}

Expected<VertBitSet> TerrainEmbedder::createCutStructure_()
{
    BooleanPreCutResult structPrecutRes;
    boolean( result_, struct_, BooleanOperation::InsideB, { .outPreCutB = &structPrecutRes } );
    VertBitSet cutVerts;
    if ( !structPrecutRes.contours.empty() )
    {
        auto structCutRes = cutMesh( structPrecutRes.mesh, structPrecutRes.contours );
        if ( structCutRes.fbsWithContourIntersections.any() )
            return unexpected( "Intersection contour of structure and terrain has self-intersections" );
        cutVerts = getIncidentVerts( structPrecutRes.mesh.topology,
                                                     fillContourLeft( structPrecutRes.mesh.topology, structCutRes.resultCut ) );
        structPrecutRes.mesh.topology.flip( cutVerts );
    }
    else
    {
        auto sFace = struct_.topology.getValidFaces().find_first();
        assert( sFace );

        Vector3f sPoint = struct_.triCenter( sFace );
        auto signDist = result_.signedDistance( sPoint, FLT_MAX );
        if ( signDist && signDist < 0.0f )
            cutVerts = structPrecutRes.mesh.topology.getValidVerts();
    }
    cutStructure_ = std::move( structPrecutRes.mesh );
    return cutVerts;
}

Expected<TerrainEmbedder::MarkedContour> TerrainEmbedder::createMarkedStructureContour_( VertBitSet&& structCutVerts )
{
    bounds_ = findRightBoundary( cutStructure_.topology );
    if ( bounds_.size() != 1 )
        return unexpected( "Structure should have only one boundary" );
    
    MarkedContour res;
    res.contour.resize( bounds_[0].size() + 1 );
    res.cutBitSet.resize( bounds_[0].size() );
    res.intBitSet.resize( bounds_[0].size() );
    auto vs = struct_.topology.vertSize();
    for ( int i = 0; i + 1 < res.contour.size(); ++i )
    {
        auto org = cutStructure_.topology.org( bounds_[0][i] );
        res.contour[i] = cutStructure_.points[org];
        if ( structCutVerts.test( org ) )
            res.cutBitSet.set( VertId( i ) );
        else if ( org >= vs )
            res.intBitSet.set( VertId( i ) );
    }
    res.contour.back() = res.contour.front();
    return res;
}

Expected<TerrainEmbedder::MappedMeshContours> TerrainEmbedder::prepareTerrainCut( MarkedContour&& mc )
{
    auto cutOffset = std::clamp( std::tan( params_.cutAngle ), 0.0f, 100.0f );
    auto fillOffset = std::clamp( std::tan( params_.fillAngle ), 0.0f, 100.0f );

    auto offCont = offsetContour_( mc, cutOffset, fillOffset );

    for ( int loneResolveTries = 0; loneResolveTries < 5; ++loneResolveTries )
    {
        std::vector<MeshTriPoint> mtps( offCont.contour.size() - 1 );
        tbb::task_group_context ctx;
        bool canceled = false;
        ParallelFor( mtps, [&] ( size_t i )
        {
            auto index = findOffsetContourIndex_( int( i ), offCont.idsShifts );
            const auto& startPoint = mc.contour[index];
            if ( mc.intBitSet.test( VertId( index ) ) )
            {
                mtps[i] = findProjection( startPoint, result_ ).mtp;
                return;
            }
            auto line =
                Line3f( startPoint,
                    to3dim( offCont.contour[i] ) + Vector3f( 0, 0, startPoint.z ) +
                    ( mc.cutBitSet.test( VertId( index ) ) ? Vector3f::plusZ() : Vector3f::minusZ() ) - startPoint );
            auto interRes = rayMeshIntersect( result_, line, -FLT_MIN ); // - FLT_MIN here to handle vertex lying in the same plane
            if ( !interRes )
            {
                if ( ctx.cancel_group_execution() )
                    canceled = true;
                return;
            }
            mtps[i] = interRes.mtp;
        } );

        if ( canceled )
            return unexpected( "Cannot embed structure beyond terrain" );

        filterDuplicates( mtps, offCont.idsShifts );

        Contour2f planarCont( mtps.size() + 1 );
        ParallelFor( mtps, [&] ( size_t i )
        {
            planarCont[i] = to2dim( result_.triPoint( mtps[i] ) );
        } );
        planarCont.back() = planarCont.front();

        auto filterBt = filterBowties( planarCont );
        std::vector<std::vector<MeshTriPoint>> noBowtiesMtps( filterBt.initIndices.size() );
        for ( int i = 0; i < noBowtiesMtps.size(); ++i )
        {
            const auto& initInds = filterBt.initIndices[i];
            const auto& noBTCont = filterBt.contours[i];
            auto& noBowtiesMtp = noBowtiesMtps[i];
            noBowtiesMtp.resize( initInds.size() );
            for ( int j = 0; j < initInds.size(); ++j )
            {
                if ( initInds[j] != -1 )
                    noBowtiesMtp[j] = mtps[initInds[j]];
                else
                {
                    auto line = Line3f( to3dim( noBTCont[j] ), Vector3f::plusZ() );
                    auto interRes = rayMeshIntersect( result_, line, -FLT_MAX, FLT_MAX );
                    if ( !interRes )
                        return unexpected( "Cannot resolve bow ties on embedded structure wall" );
                    noBowtiesMtp[j] = interRes.mtp;
                }
            }
        }

        MappedMeshContours res;
        res.filtBowTiesMap = std::move( filterBt.initIndices );
        res.contours.resize( noBowtiesMtps.size() );
        res.map.resize( noBowtiesMtps.size() );
        OneMeshContours loneContours;
        for ( int i = 0; i < res.contours.size(); ++i )
        {
            bool lone = true;
            auto cont = noBowtiesMtps[i];
            cont.push_back( cont.front() );
            auto contourRes = convertMeshTriPointsToMeshContour( result_, cont,
                [&] ( const MeshTriPoint& start, const MeshTriPoint& end, int startInd, int endInd )->Expected<SurfacePath>
            {
                auto initSMtpIndex = res.filtBowTiesMap[i][startInd];
                auto initEMtpIndex = res.filtBowTiesMap[i][endInd];
                auto baseSIndex = findOffsetContourIndex_( initSMtpIndex, offCont.idsShifts ) % bounds_[0].size();
                auto baseEIndex = findOffsetContourIndex_( initEMtpIndex, offCont.idsShifts ) % bounds_[0].size();
                auto planePoint = ( cutStructure_.orgPnt( bounds_[0][baseSIndex] ) + cutStructure_.orgPnt( bounds_[0][baseEIndex] ) ) * 0.5f;
                auto ccwPath = trackSection( result_, start, end, planePoint, true );
                auto cwPath = trackSection( result_, start, end, planePoint, false );
                if ( ccwPath.has_value() && cwPath.has_value() )
                {
                    auto ccwL = surfacePathLength( result_, *ccwPath );
                    auto cwL = surfacePathLength( result_, *cwPath );
                    if ( ccwL < cwL )
                        return ccwPath;
                    else
                        return cwPath;
                }
                else if ( ccwPath.has_value() )
                    return ccwPath;
                else if ( cwPath.has_value() )
                    return cwPath;
                else
                {
                    auto locRes = computeGeodesicPath( result_, start, end );
                    if ( !locRes.has_value() )
                        return unexpected( toString( locRes.error() ) );
                    return *locRes;
                }
            }, &res.map[i] );
            if ( !contourRes.has_value() )
                return unexpected( contourRes.error() );
            res.contours[i] = std::move( *contourRes );
            for ( int j = 0; j < res.contours[i].intersections.size(); ++j )
            {
                if ( std::holds_alternative<EdgeId>( res.contours[i].intersections[j].primitiveId ) )
                {
                    lone = false;
                    break;
                }
            }
            if ( !lone )
                continue;
            loneContours.emplace_back( std::move( res.contours[i] ) );
        }
        if ( loneContours.empty() )
        {
            res.offsetMap = std::move( offCont.idsShifts );
            res.intBitSet = std::move( mc.intBitSet );
            res.cutBitSet = std::move( mc.cutBitSet );
            return res;
        }
        subdivideLoneContours( result_, loneContours );
    }
    return unexpected( "Cannot resolve lone cut on terrain" );
}

Expected<std::vector<EdgeLoop>> TerrainEmbedder::cutTerrain( const MappedMeshContours& mmc )
{
    CutMeshParameters cutParams;
    cutParams.new2OldMap = params_.new2oldFaces;
    auto cutRes = cutMesh( result_, mmc.contours, cutParams );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "Wall contours have self-intersections" );
    auto facesToDelete = result_.topology.getValidFaces() - fillContourLeft( result_.topology, cutRes.resultCut );
    if ( params_.new2oldFaces )
    {
        for ( auto f : facesToDelete )
        {
            if ( f < params_.new2oldFaces->size() )
                ( *params_.new2oldFaces )[f] = FaceId(); // invalidate removed faces
        }
    }
    result_.topology.deleteFaces( facesToDelete );
    result_.invalidateCaches();
    return cutRes.resultCut;
}

TerrainEmbedder::ConnectionEdges TerrainEmbedder::connect_( std::vector<EdgeLoop>&& hole, MappedMeshContours&& mmc )
{
    WholeEdgeMap emap;
    auto faceNum = int( result_.topology.faceSize() );
    result_.addMesh( cutStructure_, nullptr, nullptr, &emap );
    if ( params_.outStructFaces )
    {
        params_.outStructFaces->resize( result_.topology.faceSize() );
        params_.outStructFaces->set( FaceId( faceNum ), params_.outStructFaces->size() - faceNum, true );
    }

    int prevBaseInd = 0;
    int* prevMptIndexPtr{ nullptr };
    for ( int i = 0; i < mmc.map.size(); ++i )
    {
        for ( int j = 0; j < std::min( mmc.map[i].size(), mmc.filtBowTiesMap[i].size() ); ++j )
        {
            auto cutEdgeIndex = mmc.map[i][j];
            auto initMtpIndex = mmc.filtBowTiesMap[i][j];
            if ( initMtpIndex == -1 || cutEdgeIndex == -1 )
                continue;

            auto baseEdgeIndex = findOffsetContourIndex_( initMtpIndex, mmc.offsetMap );
            if ( baseEdgeIndex + 1 >= mmc.offsetMap.size() )
                continue;

            if ( prevMptIndexPtr && baseEdgeIndex < prevBaseInd )
            {
                *prevMptIndexPtr = -1; // disable doubtable mappings
                // restart after filter
                prevMptIndexPtr = nullptr;
                i = 0;
                j = 0;
            }
            
            prevMptIndexPtr = &mmc.filtBowTiesMap[i][j];
            prevBaseInd = baseEdgeIndex;
        }
    }

    ConnectionEdges connectionInfo;
    for ( int i = 0; i < mmc.map.size(); ++i )
    {
        for ( int j = 0; j < std::min( mmc.map[i].size(), mmc.filtBowTiesMap[i].size() ); ++j )
        {
            auto cutEdgeIndex = mmc.map[i][j];
            auto initMtpIndex = mmc.filtBowTiesMap[i][j];
            if ( initMtpIndex == -1 || cutEdgeIndex == -1 )
                continue;

            auto baseEdgeIndex = findOffsetContourIndex_( initMtpIndex, mmc.offsetMap );
            if ( baseEdgeIndex + 1 >= mmc.offsetMap.size() )
                continue;

            auto e = result_.topology.prev( hole[i][cutEdgeIndex] );
            auto be = mapEdge( emap, bounds_[0][baseEdgeIndex] );
            if ( mmc.intBitSet.test( VertId( baseEdgeIndex ) ) )
            {
                auto vert = result_.topology.org( e );
                result_.topology.setOrg( e,  {} );
                result_.topology.setOrg( be, {} );
                result_.topology.splice( be, e );
                result_.topology.setOrg( e, vert );
                connectionInfo.interEdges.push_back( be );
            }
            else
            {
                auto newE = makeBridgeEdge( result_.topology, e, be );
                if ( mmc.cutBitSet.test( VertId( baseEdgeIndex ) ) )
                    connectionInfo.newCutEdges.push_back( newE );
                else
                    connectionInfo.newFillEdges.push_back( newE );
            }
        }
    }
    return connectionInfo;
}

void TerrainEmbedder::fill_( size_t oldVertSize, ConnectionEdges&& connectionInfo )
{
    auto orgMetric = getEdgeLengthFillMetric( result_ );

    FillHoleMetric metric;
    metric.edgeMetric = orgMetric.edgeMetric;
    metric.combineMetric = orgMetric.combineMetric;

    metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )
    {
        if ( ( a < oldVertSize && b < oldVertSize && c < oldVertSize ) ||
            ( a >= oldVertSize && b >= oldVertSize && c >= oldVertSize ) )
            return DBL_MAX; // no triangles on same part

        if ( orgMetric.triangleMetric )
            return orgMetric.triangleMetric( a, b, c );
        return 0.0;
    };

    FillHoleParams fhParams;
    fhParams.metric = metric;

    for ( auto edge : connectionInfo.newCutEdges )
    {
        if ( params_.outCutFaces )
            fhParams.outNewFaces = params_.outCutFaces;

        if ( !result_.topology.left( edge ) )
            fillHole( result_, edge, fhParams );
        if ( !result_.topology.left( edge.sym() ) )
            fillHole( result_, edge.sym(), fhParams );
    }

    fhParams.outNewFaces = nullptr;

    for ( auto edge : connectionInfo.newFillEdges )
    {
        if ( params_.outFillFaces )
            fhParams.outNewFaces = params_.outFillFaces;

        if ( !result_.topology.left( edge ) )
            fillHole( result_, edge, fhParams );
        if ( !result_.topology.left( edge.sym() ) )
            fillHole( result_, edge.sym(), fhParams );
    }

    // fill missed edges (no intermediate vertex)
    for ( auto edge : connectionInfo.interEdges )
    {
        if ( params_.outFillFaces )
            fhParams.outNewFaces = params_.outStructFaces;
    
        if ( !result_.topology.left( edge ) )
            fillHole( result_, edge, fhParams );
        if ( !result_.topology.left( edge.sym() ) )
            fillHole( result_, edge.sym(), fhParams );
    }
}

TerrainEmbedder::OffsetBlock TerrainEmbedder::offsetContour_( const MarkedContour& mc, float cutOffset, float fillOffset )
{
    OffsetBlock res;
    res.idsShifts.resize( int( mc.contour.size() ), 0 );

    auto contNorm = [&] ( int i )
    {
        auto norm = to2dim( mc.contour[i + 1] ) - to2dim( mc.contour[i] );
        std::swap( norm.x, norm.y );
        norm.x = -norm.x;
        norm = norm.normalized();
        return norm;
    };

    auto offset = [&] ( int i )
    {
        if ( mc.intBitSet.test( VertId( i ) ) )
            return 0.0f;
        return mc.cutBitSet.test( VertId( i ) ) ? cutOffset : fillOffset;
    };

    res.contour.reserve( 3 * mc.contour.size() );
    auto lastPoint = to2dim( mc.contour[0] ) + offset( 0 ) * contNorm( int( mc.contour.size() ) - 2 );
    for ( int i = 0; i + 1 < mc.contour.size(); ++i )
    {
        auto orgPt = to2dim( mc.contour[i] );
        auto destPt = to2dim( mc.contour[i + 1] );
        auto norm = contNorm( i );

        auto nextPoint = orgPt + norm * offset( i );
        bool sameAsPrev = false;
        // interpolation    
        if ( res.contour.empty() )
        {
            res.contour.emplace_back( std::move( lastPoint ) );
            ++res.idsShifts[i];
        }
        auto prevPoint = res.contour.back();
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
            if ( ang < 0.0f )
            {
                int numSteps = int( std::floor( std::abs( ang ) / ( params_.minAnglePrecision ) ) );
                for ( int s = 0; s < numSteps; ++s )
                {
                    float stepAng = ( ang / ( numSteps + 1 ) ) * ( s + 1 );
                    auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), orgPt );
                    res.contour.emplace_back( rotXf( prevPoint ) );
                    ++res.idsShifts[i];
                }
            }
            else
            {
                res.contour.push_back( orgPt );
                ++res.idsShifts[i];
            }
            res.contour.emplace_back( std::move( nextPoint ) );
            ++res.idsShifts[i];
        }

        res.contour.emplace_back( destPt + norm * offset( i + 1 ) );
        ++res.idsShifts[i + 1];
    }
    int prevSum = 0;
    for ( int i = 0; i + 1 < res.idsShifts.size(); ++i )
    {
        std::swap( res.idsShifts[i], prevSum );
        prevSum += res.idsShifts[i];
    }

    res.idsShifts.back() = int( res.contour.size() ) - 1;
    return res;
}

int TerrainEmbedder::findOffsetContourIndex_( int i, const std::vector<int>& idsShifts ) const
{
    int h = 0;
    for ( ; h + 1 < idsShifts.size(); ++h )
    {
        if ( i >= idsShifts[h] && i < idsShifts[h + 1] )
            break;
    }
    return h;
}

Expected<Mesh> embedStructureToTerrain( 
    const Mesh& terrain, const Mesh& structure, const EmbeddedStructureParameters& params )
{
    MR_TIMER;
    TerrainEmbedder te( terrain, structure, params );
    return te.run();
}

}