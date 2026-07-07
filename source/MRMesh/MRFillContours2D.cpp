#include "MRFillContours2D.h"
#include "MRMesh.h"
#include "MRVector2.h"
#include "MR2DContoursTriangulation.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include "MRObjectMeshData.h"
#include "MRColor.h"
#include "MRMeshFillHole.h"

namespace MR
{

class FromOxyPlaneCalculator
{
public:
    void addLineSegm( const Vector3d & a, const Vector3d & b )
    {
        sumPts_ += a;
        sumPts_ += b;
        numPts_ += 2;
        sumCross_ += cross( a, b );
    }
    void addLineSegm( const Vector3f & a, const Vector3f & b )
    {
        addLineSegm( Vector3d( a ), Vector3d( b ) );
    }
    AffineXf3d getXf() const
    {
        if ( numPts_ <= 0 )
            return {};
        auto planeNormal = sumCross_.normalized();
        auto center = sumPts_ / double( numPts_ );
        return { Matrix3d::rotation( Vector3d::plusZ(), planeNormal ), center };
    }

private:
    Vector3d sumPts_;
    Vector3d sumCross_;
    int numPts_ = 0;
};

AffineXf3f getXfFromOxyPlane( const Mesh& mesh, const std::vector<EdgePath>& paths )
{
    FromOxyPlaneCalculator c;
    for ( const auto& path : paths )
    {
        for ( const auto& edge : path )
            c.addLineSegm( mesh.orgPnt( edge ), mesh.destPnt( edge ) );
    }
    return AffineXf3f( c.getXf() );
}

AffineXf3f getXfFromOxyPlane( const Contours3f& contours )
{
    FromOxyPlaneCalculator c;
    for ( const auto& contour : contours )
    {
        for ( int i = 0; i + 1 < contour.size(); ++i )
            c.addLineSegm( contour[i], contour[i + 1] );
    }
    return AffineXf3f( c.getXf() );
}

struct ProjectFillInput
{
    std::vector<EdgeLoop> paths;
    AffineXf3f fromPlaneXf;
    Contours2f holes2d;
};

Expected<ProjectFillInput> projectHoles( const Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER;
    assert( !holeRepresentativeEdges.empty() );
    if ( holeRepresentativeEdges.empty() )
        return unexpected( "No hole edges are given" );

    // reorder to make edges ring with hole on left side
    bool badEdge = false;
    auto& meshTopology = mesh.topology;
    for ( const auto& edge : holeRepresentativeEdges )
    {
        if ( meshTopology.left( edge ) )
        {
            badEdge = true;
            break;
        }
    }
    assert( !badEdge );
    if ( badEdge )
        return unexpected( "Some hole edges have left face" );

    ProjectFillInput res;
    // make border rings
    res.paths.resize( holeRepresentativeEdges.size() );
    for ( int i = 0; i < res.paths.size(); ++i )
        res.paths[i] = trackRightBoundaryLoop( meshTopology, holeRepresentativeEdges[i] );

    // find transformation from world to plane space and back
    res.fromPlaneXf = getXfFromOxyPlane( mesh, res.paths );
    const auto toPlane = res.fromPlaneXf.inverse();

    // make contours2D (on plane) from border rings (in world)
    res.holes2d.reserve( res.paths.size() );
    for ( const auto& path : res.paths )
    {
        res.holes2d.emplace_back();
        auto& contour = res.holes2d.back();
        contour.reserve( path.size() + 1 );
        for ( const auto& edge : path )
        {
            const auto localPoint = toPlane( mesh.orgPnt( edge ) );
            contour.emplace_back( Vector2f( localPoint.x, localPoint.y ) );
        }
        contour.emplace_back( contour.front() );
    }

    return res;
}

struct ProjectedFillMesh
{
    Mesh mesh;
    std::vector<EdgeLoop> paths;
};

Expected<ProjectedFillMesh> fillProjected( const MeshTopology& tp, const ProjectFillInput& input )
{
    MR_TIMER;
    ProjectedFillMesh res;

    auto holeVertIds = std::make_unique<PlanarTriangulation::HolesVertIds>(
        PlanarTriangulation::findHoleVertIdsByHoleEdges( tp, input.paths ) );

    auto fillResult = PlanarTriangulation::triangulateDisjointContours( input.holes2d, holeVertIds.get(), &res.paths );
    holeVertIds.reset();
    if ( !fillResult )
        return unexpected( "Cannot triangulate contours with self-intersections" );

    res.mesh = std::move( *fillResult );


    if ( input.paths.size() != res.paths.size() )
        return unexpected( "Patch surface borders size different from original mesh borders size" );

    std::vector<EdgePath> invertedHoles;
    invertedHoles.reserve( res.paths.size() );
    for ( int i = 0; i < res.paths.size(); ++i )
    {
        if ( input.paths[i].size() != res.paths[i].size() )
            return unexpected( "Patch surface borders size different from original mesh borders size" );

        // degenerate holes might invert sometimes (it is expected as far as planar triangulation does not now about input topology)
        if ( res.paths[i].empty() || res.mesh.topology.right( res.paths[i].front() ) )
            if ( !res.paths[i].empty() )
                MR::reverse( invertedHoles.emplace_back( res.paths[i] ) );
    }
    if ( !invertedHoles.empty() )
    {
        auto invertedParts = fillContourLeft( res.mesh.topology, invertedHoles );
        auto invertedEdges = getIncidentEdges( res.mesh.topology, invertedParts );
        res.mesh.topology.flipOrientation( &invertedEdges );

        // validate one more time
        for ( int i = 0; i < res.paths.size(); ++i )
            if ( res.paths[i].empty() || res.mesh.topology.right( res.paths[i].front() ) )
                if ( !res.paths[i].empty() )
                    return unexpected( "Patch surface borders are incompatible with mesh borders" );
    }
    return res;
}

Expected<void> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER;

    auto projInput = projectHoles( mesh, holeRepresentativeEdges );
    if ( !projInput.has_value() )
        return unexpected( std::move( projInput.error() ) );

    auto fillRes = fillProjected( mesh.topology, *projInput );
    if ( !fillRes.has_value() )
        return unexpected( std::move( fillRes.error() ) );

    // move patch surface border points to original position (according original mesh)
    auto& patchMeshPoints = fillRes->mesh.points;
    auto& patchMeshTopology = fillRes->mesh.topology;
    auto& meshPoints = mesh.points;
    auto& meshTopology = mesh.topology;
    for ( int i = 0; i < projInput->paths.size(); ++i )
    {
        auto& path = projInput->paths[i];
        auto& newPath = fillRes->paths[i];
        for ( int j = 0; j < path.size(); ++j )
            patchMeshPoints[patchMeshTopology.org( newPath[j] )] = meshPoints[meshTopology.org( path[j] )];
    }

    // add patch surface to original mesh
    mesh.addMeshPart( fillRes->mesh, false, projInput->paths, fillRes->paths );
    return {};
}

Expected<HoleFillPlan> fillContours2DPlan( const Mesh& mesh, EdgeId holeEdgeId )
{
    auto projInput = projectHoles( mesh, { holeEdgeId } );
    if ( !projInput.has_value() )
        return unexpected( std::move( projInput.error() ) );

    auto fillRes = fillProjected( mesh.topology, *projInput );
    if ( !fillRes.has_value() )
        return unexpected( std::move( fillRes.error() ) );

    assert( fillRes->paths.size() == 1 ); // should be validated in fillProjected

    const auto& pTp = fillRes->mesh.topology;
    const auto& ip = projInput->paths[0];
    auto& np = fillRes->paths[0];
    HoleFillPlan res;
    res.numTris = pTp.numValidFaces();
    if ( res.numTris == 1 )
        return res;
    auto size = int( np.size() );
    assert( size > 3 );
    res.items.reserve( size - 3 );

    for ( ;;)
    {
        for ( int i0 = 0; i0 < np.size(); ++i0 )
        {
            auto e0 = np[i0];
            if ( !e0 )
                continue; // skip unused/encoded 
            auto i1 = int( pTp.dest( np[i0] ) );
            if ( i1 < 0 )
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            auto e1 = np[i1];
            auto ne = pTp.next( e0 );
            auto dest = pTp.dest( ne );
            if ( dest != pTp.dest( e1 ) )
                continue;
            FillHoleItem fhi;
            int i01 = ( i0 + 1 ) % size;
            fhi.edgeCode1 = i1 == i01 ? ip[i0] : int( np[i01] );
            i1 = dest;
            e1 = np[i1];
            if ( !e1 )
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            int i11 = ( i1 + 1 ) % size;
            fhi.edgeCode2 = ( pTp.dest( e1 ) == i11 ) ? ip[i1] : int( np[i11] );
            res.items.push_back( std::move( fhi ) );
            if ( res.items.size() == size - 3 )
                return res;
            np[i0] = ne;
            np[i01] = EdgeId( -int( res.items.size() ) ); // encode newly created plan edge in free slot
        }
    }
}

Expected<void> fillPlanarHole( ObjectMeshData& data, std::vector<EdgeLoop>& holeContours )
{
    MR_TIMER;

    if ( !data.mesh )
        return unexpected( "fillPlanarHole: no input mesh" );

    auto& mesh = *data.mesh;
    auto& tp = mesh.topology;

    // take first edge from each contour and check that it is a hole boundary
    EdgePath holesEdges;
    for ( const auto& path : holeContours )
    {
        if ( path.empty() )
            continue;
        for ( auto e : path )
            if ( tp.right( e ).valid() )
                return unexpected( "fillPlanarHole: not hole contour given" );
        holesEdges.push_back( path.front().sym() );
    }

    for ( auto& loop : holeContours )
    {
        // if not closed, add edge to enclose
        if ( loop.empty() )
            continue;
        if ( tp.org( loop.front() ) == tp.dest( loop.back() ) )
            continue;
        auto newEdge = makeBridgeEdge( tp, loop.back().sym(), tp.prev( loop.front() ) );
        if ( !newEdge )
            continue;
        loop.emplace_back( newEdge );
    }

    const auto fsz0 = tp.faceSize();
    if ( !holesEdges.empty() )
    {
        auto fillSuccess = fillContours2D( mesh, holesEdges );
        if ( !fillSuccess.has_value() )
        {
            return unexpected( "Cannot fill section: " + fillSuccess.error() );
        }
    }

    const auto fsz = tp.faceSize();
    data.selectedFaces.resize( fsz );
    data.selectedFaces.set( FaceId{ fsz0 }, fsz - fsz0, true );
    data.selectedFaces &= tp.getValidFaces();

    tp.excludeLoneEdges( data.selectedEdges );
    tp.excludeLoneEdges( data.creases );

    auto& fcm = data.faceColors;
    auto& tpf = data.texturePerFace;
    if ( fcm.empty() && tpf.empty() )
        return {};
    if ( !fcm.empty() )
        fcm.resize( fsz );
    if ( !tpf.empty() )
        tpf.resize( fsz );
    for ( FaceId f = FaceId{ fsz0 }; f < fsz; ++f )
    {
        VertId v[3];
        tp.getTriVerts( f, v );
        float sumNeighColorWeight = 0;
        Vector4f sum;
        FaceId maxAreaF;
        float maxArea = 0.0f;
        for ( size_t i = 0; i < 3; ++i )
        {
            for ( auto e : orgRing( tp, v[i] ) )
            {
                const auto tmpFace = tp.left( e );
                if ( tmpFace >= fsz0 )
                    continue;

                const float area = mesh.area( tmpFace );
                if ( !tpf.empty() )
                {
                    if ( area > maxArea )
                    {
                        maxArea = area;
                        maxAreaF = tmpFace;
                    }
                }
                if ( !fcm.empty() )
                {
                    const auto& color = fcm[tmpFace];
                    sum += Vector4f( color ) * area;
                    sumNeighColorWeight += area;
                }
            }
        }
        if ( !fcm.empty() )
            fcm[f] = Color( sum / float( sumNeighColorWeight ) );
        if ( !tpf.empty() )
            tpf[f] = tpf[maxAreaF];
    }

    return {};
}

}
