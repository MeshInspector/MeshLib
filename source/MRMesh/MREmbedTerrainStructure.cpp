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
#include "MRParallelFor.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRPolyline.h"
#include "MRLinesSave.h"
#include "MRMapEdge.h"

namespace MR
{

struct FilterBowtiesResult
{
    Contours2f contours;
    std::vector<std::vector<int>> initIndices;
};
FilterBowtiesResult filterBowties( const Contour2f& cont )
{
    auto mesh = PlanarTriangulation::triangulateContours( { cont } );
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

class TerrainEmbedder
{
public:
    TerrainEmbedder( const Mesh& terrain, const Mesh& structure, const EmbeddedStructureParameters& params ) :
        struct_{ structure },
        params_{ params },
        result{terrain}
    {
    }
    Expected<Mesh, std::string> run();
private:
    Expected<VertBitSet,std::string> createCuttedStructure_();

    struct MarkedContour
    {
        Contour3f contour;
        VertBitSet cutBitSet;
        VertBitSet intBitSet;
    };

    Expected<MarkedContour, std::string> createMarkedStructureContour_( VertBitSet&& structCutVerts );

    struct MappedMeshContours
    {
        OneMeshContours contours;
        std::vector<std::vector<int>> map;
        std::vector<std::vector<int>> filtBowTiesMap;
        std::vector<int> offsetMap;
    };

    Expected<MappedMeshContours, std::string> prepareTerrainCut( MarkedContour&& mc );
    Expected<std::vector<EdgeLoop>, std::string> cutTerrain( const MappedMeshContours& mmc );
    void connect_( std::vector<EdgeLoop>&& hole, MappedMeshContours&& mmc );
    void fill_();

    struct OffsetBlock
    {
        Contour2f contour;
        std::vector<int> idsShifts;
    };
    OffsetBlock offsetContour_( const MarkedContour& mc, float cutOffset, float fillOffset );

    int findOffsetContourIndex_( int i, const std::vector<int>& idsShifts ) const;

    const Mesh& struct_;
    const EmbeddedStructureParameters& params_;

    Mesh result;
    Mesh cuttedStructure_;
    std::vector<EdgeLoop> bounds_;
};

Expected<Mesh, std::string> TerrainEmbedder::run()
{
    auto cutBs = createCuttedStructure_();
    if ( !cutBs.has_value() )
        return unexpected( cutBs.error() );

    auto markedContour = createMarkedStructureContour_( std::move( *cutBs ) );
    if ( !markedContour.has_value() )
        return unexpected( markedContour.error() );

    auto prepCut = prepareTerrainCut( std::move( *markedContour ) );
    if ( !prepCut.has_value() )
        return unexpected( prepCut.error() );

    auto cutTer = cutTerrain( *prepCut );
    if ( !cutTer.has_value() )
        return unexpected( cutTer.error() );

    connect_( std::move( *cutTer ), std::move( *prepCut ) );

    return std::move( result );
}

Expected<VertBitSet, std::string> TerrainEmbedder::createCuttedStructure_()
{
    BooleanPreCutResult structPrecutRes;
    boolean( result, struct_, BooleanOperation::InsideB, { .outPreCutB = &structPrecutRes } );
    VertBitSet cutVerts;
    if ( !structPrecutRes.contours.empty() )
    {
        auto structCutRes = cutMesh( structPrecutRes.mesh, structPrecutRes.contours );
        if ( structCutRes.fbsWithCountourIntersections.any() )
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
        auto signDist = result.signedDistance( sPoint, FLT_MAX );
        if ( signDist && signDist < 0.0f )
            cutVerts = structPrecutRes.mesh.topology.getValidVerts();
    }
    cuttedStructure_ = std::move( structPrecutRes.mesh );
    return cutVerts;
}

Expected<TerrainEmbedder::MarkedContour, std::string> TerrainEmbedder::createMarkedStructureContour_( VertBitSet&& structCutVerts )
{
    bounds_ = findRightBoundary( cuttedStructure_.topology );
    if ( bounds_.size() != 1 )
        return unexpected( "Structure should have only one boundary" );
    
    MarkedContour res;
    res.contour.resize( bounds_[0].size() + 1 );
    res.cutBitSet.resize( bounds_[0].size() );
    res.intBitSet.resize( bounds_[0].size() );
    auto vs = struct_.topology.vertSize();
    for ( int i = 0; i + 1 < res.contour.size(); ++i )
    {
        auto org = cuttedStructure_.topology.org( bounds_[0][i] );
        res.contour[i] = cuttedStructure_.points[org];
        if ( structCutVerts.test( org ) )
            res.cutBitSet.set( VertId( i ) );
        else if ( org >= vs )
            res.intBitSet.set( VertId( i ) );
    }
    res.contour.back() = res.contour.front();
    return res;
}

Expected<TerrainEmbedder::MappedMeshContours, std::string> TerrainEmbedder::prepareTerrainCut( MarkedContour&& mc )
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
                mtps[i] = findProjection( startPoint, result ).mtp;
                return;
            }
            auto line =
                Line3f( startPoint,
                    to3dim( offCont.contour[i] ) + Vector3f( 0, 0, startPoint.z ) +
                    ( mc.cutBitSet.test( VertId( index ) ) ? Vector3f::plusZ() : Vector3f::minusZ() ) - startPoint );
            auto interRes = rayMeshIntersect( result, line );
            if ( !interRes )
            {
                if ( ctx.cancel_group_execution() )
                    canceled = true;
                return;
            }
            mtps[i] = interRes->mtp;
        } );

        if ( canceled )
            return unexpected( "Cannot embed structure beyond terrain" );

        Contour2f planarCont( mtps.size() + 1 );
        ParallelFor( mtps, [&] ( size_t i )
        {
            planarCont[i] = to2dim( result.triPoint( mtps[i] ) );
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
                    auto interRes = rayMeshIntersect( result, line, -FLT_MAX, FLT_MAX );
                    if ( !interRes )
                        return unexpected( "Cannot resolve bow ties on embedded structure wall" );
                    noBowtiesMtp[j] = interRes->mtp;
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
            res.contours[i] = convertMeshTriPointsToClosedContour( result, noBowtiesMtps[i], &res.map[i] );
            for ( int j = 0; j < res.contours[i].intersections.size(); ++j )
            {
                if ( res.contours[i].intersections[j].primitiveId.index() == OneMeshIntersection::VariantIndex::Edge )
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
            return res;
        }
        subdivideLoneContours( result, loneContours );
    }
    return unexpected( "Cannot resolve lone cut on terrain" );
}

Expected<std::vector<EdgeLoop>, std::string> TerrainEmbedder::cutTerrain( const MappedMeshContours& mmc )
{
    auto cutRes = cutMesh( result, mmc.contours );
    if ( cutRes.fbsWithCountourIntersections.any() )
        return unexpected( "Wall contours have self-intersections" );

    result.topology.deleteFaces( result.topology.getValidFaces() - fillContourLeft( result.topology, cutRes.resultCut ) );
    result.invalidateCaches();
    return cutRes.resultCut;
}

void TerrainEmbedder::connect_( std::vector<EdgeLoop>&& hole, MappedMeshContours&& mmc )
{
    WholeEdgeMap emap;
    result.addPart( cuttedStructure_, nullptr, nullptr, &emap );

    for ( int i = 0; i < mmc.map.size(); ++i )
    {
        for ( int j = 0; j < mmc.map[i].size(); ++j )
        {
            auto cutEdgeIndex = mmc.map[i][j];
            auto initMtpIndex = mmc.filtBowTiesMap[i][j];
            if ( initMtpIndex == -1 || cutEdgeIndex == -1 )
                continue;

            auto baseEdgeIndex = findOffsetContourIndex_( initMtpIndex, mmc.offsetMap );
            if ( baseEdgeIndex + 1 >= mmc.offsetMap.size() )
                continue;

            makeBridgeEdge( result.topology,
                result.topology.prev( hole[i][cutEdgeIndex] ),
                mapEdge( emap, bounds_[0][baseEdgeIndex] ) );
        }
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
            int numSteps = int( std::floor( std::abs( ang ) / ( params_.minAnglePrecision ) ) );
            for ( int s = 0; s < numSteps; ++s )
            {
                float stepAng = ( ang / ( numSteps + 1 ) ) * ( s + 1 );
                auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), orgPt );
                res.contour.emplace_back( rotXf( prevPoint ) );
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

Expected<Mesh, std::string> embedStructureToTerrain( 
    const Mesh& terrain, const Mesh& structure, const EmbeddedStructureParameters& params )
{
    MR_TIMER;
    TerrainEmbedder te( terrain, structure, params );
    return te.run();
}

}